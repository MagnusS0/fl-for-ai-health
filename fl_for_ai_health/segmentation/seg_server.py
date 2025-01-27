"""BRATS segmentation server implementation for Flower."""

from flwr.common import Context, ndarrays_to_parameters
from flwr.server import ServerApp, ServerAppComponents, ServerConfig
from fl_for_ai_health.segmentation.brats_task import get_weights, set_weights, test, load_data, load_model
from fl_for_ai_health.segmentation.seg_strategy import CustomFedAvg
from typing import Dict, Tuple
from torch.utils.data import DataLoader
import torch

def gen_evaluate_fn(
    testloader: DataLoader,
    device: torch.device,
    run_config: Dict,
):
    """Generate the function for centralized evaluation."""

    def evaluate(server_round, parameters_ndarrays, config, viz_fn=None, run_config=run_config) -> Tuple[float, Dict[str, float], Dict]:
        """Evaluate global model on centralized test set."""
        net = load_model(run_config)
        set_weights(net, parameters_ndarrays)
        net.to(device)
        loss, dice_score, iou_score = test(
            net, 
            testloader, 
            device=device, 
            viz_fn=viz_fn, 
            run_config=run_config,
            server_round=server_round
        )
        return loss, {"dice": dice_score, "miou": iou_score}, run_config

    return evaluate


def server_fn(context: Context):
    num_rounds = context.run_config["num-server-rounds"]
    fraction_fit = context.run_config["fraction-fit"]
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Load test dataset
    testloader, _ = load_data(
        partition_id=0, 
        num_partitions=1, 
        run_config=context.run_config, 
        global_test_set=True
    )

    # Initialize model parameters
    net = load_model(context.run_config)
    parameters = ndarrays_to_parameters(get_weights(net))

    strategy = CustomFedAvg(
        fraction_fit=fraction_fit,
        fraction_evaluate=1.0,
        min_available_clients=2,
        initial_parameters=parameters,
        tb_log_dir="tb_logs",
        tb_run_name=context.run_config["tb_run_name"],
        evaluate_fn=gen_evaluate_fn(testloader, device, context.run_config)
    )
    
    config = ServerConfig(num_rounds=num_rounds)
    return ServerAppComponents(strategy=strategy, config=config)

# Create ServerApp
app = ServerApp(server_fn=server_fn) 