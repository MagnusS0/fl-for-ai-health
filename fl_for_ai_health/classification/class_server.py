"""fl-for-AI-health: A Flower / PyTorch app."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from fl_for_ai_health.classification.medmnist_task import (
    get_weights,
    load_model,
    set_weights,
    test,
    load_data,
)
from typing import Dict
from fl_for_ai_health.classification.class_strategy import CustomFedAvg
from torch.utils.data import DataLoader
import torch


def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    run_config: Dict,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config):
        net = load_model(run_config)
        set_weights(net, parameters_ndarrays)
        loss, accuracy, auc = test(net, testloader, device)
        return loss, {"accuracy": accuracy, "auc": auc}, run_config

    return evaluate


def server_fn(context: Context):
    # Read from config
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load test dataset
    testloader, _ = load_data(partition_id=0, num_partitions=1, split="test")

    # Initialize model parameters
    net = load_model(context.run_config)
    parameters = ndarrays_to_parameters(get_weights(net))
    # Define strategy with custom class
    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        tb_log_dir="tb_logs",
        tb_run_name=context.run_config["tb_run_name"],
        evaluate_fn=gen_evaluate_fn(testloader, device, context.run_config),
    )
    config = ServerConfig(num_rounds=num_rounds)

    return ServerAppComponents(strategy=strategy, config=config)


# Create ServerApp
app = ServerApp(server_fn=server_fn)
