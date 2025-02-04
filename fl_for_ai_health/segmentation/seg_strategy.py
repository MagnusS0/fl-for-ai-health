"""
Custom federated learning strategy with TensorBoard integration.

This module implements a customized federated averaging strategy that:
- Maintains client state across rounds
- Performs centralized model evaluation
- Provides extensive TensorBoard logging
- Implements best model checkpointing
"""

from typing import Dict, List, Tuple, Optional
import os
import datetime

import matplotlib
import matplotlib.pyplot as plt
import torch
import numpy as np
from flwr.common import EvaluateRes, FitIns, Parameters, Scalar, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from flwr.server.client_manager import ClientManager
from torch.utils.tensorboard import SummaryWriter

from fl_for_ai_health.segmentation.brats_task import load_model, set_weights

# Set non-interactive backend
# Otherwise will crash in ray multiprocessing
matplotlib.use("Agg")


class CustomFedAvg(FedAvg):
    """Federated Averaging strategy with enhanced tracking and visualization.
    
    Attributes:
        client_states: Dictionary maintaining client-specific training states
        writer: TensorBoard SummaryWriter for logging metrics
        best_acc_so_far: Best Dice score observed during training
        tb_log_dir: Root directory for TensorBoard logs
    """

    def __init__(
        self,
        *args,
        tb_log_dir: str = "tb_logs",
        tb_run_name: str = "tb_run",
        **kwargs
    ) -> None:
        """Initialize custom FedAvg strategy with TensorBoard support.
        
        Args:
            tb_log_dir: Base directory for TensorBoard logs
            tb_run_name: Experiment name for TensorBoard
            *args: Positional arguments for parent class
            **kwargs: Keyword arguments for parent class
        """
        super().__init__(*args, **kwargs)
        
        self.client_states: Dict[str, Dict] = {}
        self.best_acc_so_far: float = 0.0
        self.tb_log_dir = self._create_tb_log_dir(tb_log_dir, tb_run_name)
        self.writer = SummaryWriter(log_dir=self.tb_log_dir)

    def _create_tb_log_dir(self, base_dir: str, run_name: str) -> str:
        """Create timestamped directory for TensorBoard logs."""
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        log_dir = os.path.join(base_dir, f"{run_name}_{timestamp}")
        os.makedirs(log_dir, exist_ok=True)
        return log_dir

    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager
    ) -> List[Tuple[any, FitIns]]:
        """Configure client training with state preservation."""
        client_instructions = []
        
        for client in client_manager.all().values():
            config = self._create_client_config(client.cid)
            fit_ins = FitIns(parameters, config)
            client_instructions.append((client, fit_ins))
            
        return client_instructions

    def _create_client_config(self, cid: str) -> Dict[str, Scalar]:
        """Generate configuration for a client."""
        client_state = self.client_states.get(cid, {})
        return {
            "local_epochs": 1 if len(self.client_states) < 2 else 1,
            "scheduler_state_bytes": client_state.get("scheduler_state_bytes", b""),
        }

    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[any, EvaluateRes]],
        failures: List[BaseException]
    ) -> Tuple[Optional[Parameters], Dict[str, Scalar]]:
        """Aggregate client results and update client states."""
        aggregated_parameters, metrics = super().aggregate_fit(server_round, results, failures)
        self._update_client_states(results)
        return aggregated_parameters, metrics

    def _update_client_states(self, results: List[Tuple[any, EvaluateRes]]) -> None:
        """Update client states from training results."""
        for client, result in results:
            if result.metrics:
                self.client_states[client.cid] = {
                    "scheduler_state_bytes": result.metrics.get("scheduler_state_bytes", b""),
                }

    def evaluate(
        self, 
        server_round: int, 
        parameters: Parameters
    ) -> Optional[Tuple[float, Dict[str, Scalar]]]:
        """Perform centralized model evaluation."""
        if not self.evaluate_fn:
            return None

        parameters_ndarrays = parameters_to_ndarrays(parameters)
        loss, metrics, run_config = self.evaluate_fn(
            server_round, 
            parameters_ndarrays, 
            {},
            self.visualize_and_log_predictions
        )
        
        self._log_central_metrics(server_round, loss, metrics)
        self._update_best_model(server_round, metrics["dice"], parameters, run_config)
        
        return loss, metrics

    def _log_central_metrics(
        self,
        server_round: int,
        loss: float,
        metrics: Dict[str, float]
    ) -> None:
        """Log centralized evaluation metrics to TensorBoard."""
        self.writer.add_scalar("centralized/loss", loss, server_round)
        self.writer.add_scalar("centralized/dice", metrics["dice"], server_round)
        self.writer.add_scalar("centralized/miou", metrics["miou"], server_round)

    def _update_best_model(
        self,
        round_num: int,
        dice_score: float,
        parameters: Parameters,
        run_config: Dict[str, Scalar]
    ) -> None:
        """Save model checkpoint if current score is best."""
        if dice_score > self.best_acc_so_far:
            self.best_acc_so_far = dice_score
            print(f"🏆 New best Dice score: {self.best_acc_so_far:.4f}")
            self._save_best_model(parameters, run_config)

    def _save_best_model(
        self,
        parameters: Parameters,
        run_config: Dict[str, Scalar]
    ) -> None:
        """Save best model weights to disk."""
        model = load_model(run_config)
        set_weights(model, parameters_to_ndarrays(parameters))
        
        save_path = os.path.join(self.tb_log_dir, "best_model.pth")
        torch.save(model.state_dict(), save_path)
        print(f"💾 Saved best model to {save_path}")

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[any, EvaluateRes]],
        failures: List[BaseException]
    ) -> Tuple[float, Dict[str, Scalar]]:
        """Aggregate client evaluation results."""
        loss_aggregated, _ = super().aggregate_evaluate(server_round, results, failures)
        metrics = self._calculate_federated_metrics(results)
        self._log_federated_metrics(server_round, loss_aggregated, metrics)
        
        return loss_aggregated, metrics

    def _calculate_federated_metrics(
        self,
        results: List[Tuple[ EvaluateRes]]
    ) -> Dict[str, float]:
        """Calculate weighted average of client metrics."""
        dice_scores, iou_scores, examples = [], [], []
        
        for _, res in results:
            dice_scores.append(res.metrics["dice"] * res.num_examples)
            iou_scores.append(res.metrics["iou"] * res.num_examples)
            examples.append(res.num_examples)
            
        total_examples = sum(examples)
        return {
            "dice": sum(dice_scores) / total_examples,
            "miou": sum(iou_scores) / total_examples
        }

    def _log_federated_metrics(
        self,
        round_num: int,
        loss: float,
        metrics: Dict[str, float]
    ) -> None:
        """Log federated metrics to TensorBoard."""
        self.writer.add_scalar("federated/dice", metrics["dice"], round_num)
        self.writer.add_scalar("federated/miou", metrics["miou"], round_num)
        self.writer.add_scalar("federated/loss", loss, round_num)

    def visualize_and_log_predictions(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        outputs: torch.Tensor,
        batch_idx: int,
        round_num: int
    ) -> None:
        """Visualize and log model predictions to TensorBoard.
        
        Args:
            images: Input tensor of shape (batch_size, channels, height, width)
            labels: Ground truth segmentation masks
            outputs: Model output logits
            round_num: Current training round for logging
        """
        batch_size = images.size(0)
        fig = self._create_prediction_figure(batch_size)
        self._plot_predictions(fig, images, labels, outputs, batch_size)
        
        self.writer.add_figure(f"predictions_{batch_idx}", fig, round_num)
    
        plt.close(fig)

    def _create_prediction_figure(self, batch_size: int) -> plt.Figure:
        """Initialize prediction visualization figure."""
        fig, axes = plt.subplots(
            batch_size, 
            6, 
            figsize=(24, 3*batch_size),
            tight_layout=False
        )
        plt.subplots_adjust(wspace=0.05, hspace=0.1)
        return fig

    def _plot_predictions(
        self,
        fig: plt.Figure,
        images: torch.Tensor,
        labels: torch.Tensor,
        outputs: torch.Tensor,
        batch_size: int
    ) -> None:
        """Plot predictions for each sample in batch."""
        modality_names = ["FLAIR", "T1w", "t1gd", "T2w"]
        predictions = torch.argmax(outputs, dim=1).cpu().numpy()
        
        for i in range(batch_size):
            self._plot_modalities(images[i], modality_names, fig.axes[i*6:(i+1)*6])
            self._plot_ground_truth(labels[i], fig.axes[i*6 + 4])
            self._plot_prediction(predictions[i], fig.axes[i*6 + 5])
            
        # self._add_colorbar(fig)

    def _plot_modalities(
        self,
        image: torch.Tensor,
        modality_names: List[str],
        axes: List[plt.Axes]
    ) -> None:
        """Plot individual imaging modalities."""
        for mod_idx in range(4):
            ax = axes[mod_idx]
            ax.imshow(image[mod_idx].cpu().numpy(), cmap="gray", vmin=0, vmax=1)
            ax.axis("off")
            if ax.get_subplotspec().is_first_row():
                ax.set_title(modality_names[mod_idx], fontsize=8)

    def _plot_ground_truth(self, label: torch.Tensor, ax: plt.Axes) -> None:
        """Plot ground truth segmentation."""
        ax.imshow(label.cpu().numpy(), cmap="tab10", vmin=0, vmax=3)
        ax.axis("off")
        if ax.get_subplotspec().is_first_row():
            ax.set_title("Ground Truth", fontsize=8)

    def _plot_prediction(self, prediction: np.ndarray, ax: plt.Axes) -> None:
        """Plot model prediction."""
        ax.imshow(prediction, cmap="tab10", vmin=0, vmax=3)
        ax.axis("off")
        if ax.get_subplotspec().is_first_row():
            ax.set_title("Prediction", fontsize=8)

    def _add_colorbar(self, fig: plt.Figure) -> None:
        """Add standardized colorbar to prediction figure."""
        cax = fig.add_axes([0.92, 0.15, 0.01, 0.7])
        cbar = plt.colorbar(
            plt.cm.ScalarMappable(
                norm=plt.Normalize(0, 3),
                cmap="tab10"
            ),
            cax=cax
        )
        cbar.set_ticks([0, 1, 2, 3])
        cbar.set_ticklabels(["Background", "Edema", "Non-enhancing", "Enhancing"])
        cbar.ax.tick_params(labelsize=8)

    def __del__(self) -> None:
        """Ensure proper cleanup of TensorBoard writer."""
        if hasattr(self, "writer"):
            self.writer.close()