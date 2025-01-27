"""Custom strategy for federated learning. With TensorBoard logging."""

import os
import torch
import matplotlib.pyplot as plt
import matplotlib as mpl

mpl.use("Agg")
from rich import print
from typing import List, Tuple, Dict
from flwr.common import EvaluateRes, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from fl_for_ai_health.segmentation.brats_task import set_weights, load_model
import datetime
from torch.utils.tensorboard import SummaryWriter


class CustomFedAvg(FedAvg):
    def __init__(
        self, *args, tb_log_dir: str = "tb_logs", tb_run_name: str = "tb_run", **kwargs
    ):
        super().__init__(*args, **kwargs)

        # Initialize TensorBoard writer
        self.tb_log_dir = f"{tb_log_dir}/{tb_run_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tb_log_dir)

        # Keep track of best accuracy
        self.best_acc_so_far = 0.0

    def update_best_acc(self, round, dice, parameters, run_config):
        if dice > self.best_acc_so_far:
            self.best_acc_so_far = dice
            print(f"🏆 New best Dice score: {self.best_acc_so_far:.4f}")

            model = load_model(run_config)
            parameters_ndarrays = parameters_to_ndarrays(parameters)
            set_weights(model, parameters_ndarrays)
            save_path = os.path.join(self.tb_log_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model to {save_path}")

    def evaluate(self, server_round, parameters):
        """Run centralized evaluation if callback was passed to strategy init."""
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        loss, metrics, run_config = self.evaluate_fn(
            server_round, parameters_ndarrays, {}, self.visualize_and_log_predictions
        )
        self.writer.add_scalar("centralized/loss", loss, server_round)
        self.writer.add_scalar("centralized/dice", metrics["dice"], server_round)
        self.writer.add_scalar("centralized/miou", metrics["miou"], server_round)
        # Update best accuracy
        self.update_best_acc(server_round, metrics["dice"], parameters, run_config)

        return loss, metrics

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[int, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[float, Dict[str, float]]:
        """Aggregate evaluation results and log metrics to TensorBoard."""
        if not results:
            return 0.0, {}

        # Weighted dice score averaging
        dice_scores = [r.metrics["dice"] * r.num_examples for _, r in results]
        iou_scores = [r.metrics["iou"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        # Compute weighted average
        dice_aggregated = sum(dice_scores) / sum(examples)
        iou_aggregated = sum(iou_scores) / sum(examples)

        # Get loss through parent class
        loss_aggregated, _ = super().aggregate_evaluate(server_round, results, failures)

        self.writer.add_scalar("federated/dice", dice_aggregated, server_round)
        self.writer.add_scalar("federated/miou", iou_aggregated, server_round)
        self.writer.add_scalar("federated/loss", loss_aggregated, server_round)

        return loss_aggregated, {
            "dice": float(dice_aggregated),
            "miou": float(iou_aggregated),
        }

    def visualize_and_log_predictions(
        self,
        images: torch.Tensor,
        labels: torch.Tensor,
        outputs: torch.Tensor,
        round: int,
    ):
        """Visualize and log predictions to TensorBoard."""
        images = images.cpu().numpy()
        labels = labels.cpu().numpy()
        outputs = torch.argmax(outputs, dim=1).cpu().numpy()

        # Create figure directly without pyplot
        num_images = images.shape[0]
        fig = plt.figure(figsize=(15, 5 * num_images))

        for i in range(num_images):
            ax1 = fig.add_subplot(num_images, 3, i * 3 + 1)
            ax1.imshow(images[i, 0, :, :], cmap="gray")
            ax1.axis("off")
            ax1.set_title("Image")

            ax2 = fig.add_subplot(num_images, 3, i * 3 + 2)
            ax2.imshow(labels[i, :, :], cmap="tab10", vmin=0, vmax=3)
            ax2.axis("off")
            ax2.set_title("Label")

            ax3 = fig.add_subplot(num_images, 3, i * 3 + 3)
            ax3.imshow(outputs[i, :, :], cmap="tab10", vmin=0, vmax=3)
            ax3.axis("off")
            ax3.set_title("Prediction")

        # Log and close immediately
        self.writer.add_figure("predictions", fig, round)
        plt.close(fig)

    def __del__(self):
        """Ensure TensorBoard writer is closed properly."""
        if hasattr(self, "writer"):
            self.writer.close()
