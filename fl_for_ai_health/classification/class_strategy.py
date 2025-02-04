"""Custom strategy for classification task with TensorBoard logging."""

import os
import torch
import matplotlib as mpl

mpl.use("Agg")
from rich import print
from typing import List, Tuple, Dict
from flwr.common import EvaluateRes, parameters_to_ndarrays
from flwr.server.strategy import FedAvg
from fl_for_ai_health.classification.medmnist_task import set_weights, load_model
import datetime
from torch.utils.tensorboard import SummaryWriter


class CustomFedAvg(FedAvg):
    def __init__(
        self,
        *args,
        tb_log_dir: str = "tb_logs",
        tb_run_name: str = "classification_run",
        **kwargs,
    ):
        super().__init__(*args, **kwargs)

        # Initialize TensorBoard
        self.tb_log_dir = f"{tb_log_dir}/{tb_run_name}_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}"
        os.makedirs(tb_log_dir, exist_ok=True)
        self.writer = SummaryWriter(log_dir=self.tb_log_dir)
        self.best_accuracy_so_far = 0.0

    def update_best_model(self, round, metrics, parameters, run_config):
        """Save model if it achieves best AUC so far."""
        if metrics["accuracy"] > self.best_accuracy_so_far:
            self.best_accuracy_so_far = metrics["accuracy"]
            print(f"🏆 New best Accuracy: {self.best_accuracy_so_far:.4f}")
            print(f"With AUC: {metrics['auc']:.4f} and F1 Score: {metrics['f1_score']:.4f}")

            # Save the best model
            model = load_model(run_config)
            parameters_ndarrays = parameters_to_ndarrays(parameters)
            set_weights(model, parameters_ndarrays)
            save_path = os.path.join(self.tb_log_dir, "best_model.pth")
            torch.save(model.state_dict(), save_path)
            print(f"💾 Saved best model to {save_path}")

    def aggregate_evaluate(
        self,
        server_round: int,
        results: List[Tuple[int, EvaluateRes]],
        failures: List[BaseException],
    ) -> Tuple[float, Dict[str, float]]:
        """Aggregate evaluation metrics and log to TensorBoard."""
        if not results:
            return 0.0, {}

        # Calculate weighted averages
        accuracies = [r.metrics["accuracy"] * r.num_examples for _, r in results]
        aucs = [r.metrics["auc"] * r.num_examples for _, r in results]
        f1_scores = [r.metrics["f1_score"] * r.num_examples for _, r in results]
        examples = [r.num_examples for _, r in results]

        avg_accuracy = sum(accuracies) / sum(examples)
        avg_auc = sum(aucs) / sum(examples)
        avg_f1_score = sum(f1_scores) / sum(examples)
        loss_aggregated, _ = super().aggregate_evaluate(server_round, results, failures)

        # Log metrics to TensorBoard
        self.writer.add_scalar("federated/loss", loss_aggregated, server_round)
        self.writer.add_scalar("federated/accuracy", avg_accuracy, server_round)
        self.writer.add_scalar("federated/auc", avg_auc, server_round)
        self.writer.add_scalar("federated/f1_score", avg_f1_score, server_round)

        return loss_aggregated, {"accuracy": float(avg_accuracy), "auc": float(avg_auc), "f1_score": float(avg_f1_score)}

    def evaluate(self, server_round, parameters):
        """Run centralized evaluation and log results."""
        parameters_ndarrays = parameters_to_ndarrays(parameters)
        loss, metrics, run_config = self.evaluate_fn(
            server_round, parameters_ndarrays, {}
        )

        # Log centralized metrics
        self.writer.add_scalar("centralized/loss", loss, server_round)
        self.writer.add_scalar(
            "centralized/accuracy", metrics["accuracy"], server_round
        )
        self.writer.add_scalar("centralized/auc", metrics["auc"], server_round)
        self.writer.add_scalar("centralized/f1_score", metrics["f1_score"], server_round)

        # Update best model
        self.update_best_model(server_round, metrics, parameters, run_config)

        return loss, metrics

    def __del__(self):
        """Ensure proper cleanup of TensorBoard writer."""
        if hasattr(self, "writer"):
            self.writer.close()
