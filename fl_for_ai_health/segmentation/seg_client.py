"""BRATS segmentation client implementation for Flower."""

import torch
from flwr.client import ClientApp, NumPyClient
from flwr.common import Context, ConfigsRecord
from fl_for_ai_health.segmentation.brats_task import get_weights, load_data, set_weights, test, train, load_model

class FlowerClient(NumPyClient):
    def __init__(self, net, trainloader, valloader, local_epochs, context: Context):
        self.net = net
        self.client_state = context.state
        self.trainloader = trainloader
        self.valloader = valloader
        self.local_epochs = local_epochs
        self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        self.net.to(self.device)
        self.run_config = context.run_config

        if "fit_metrics" not in self.client_state.configs_records:
            self.client_state.configs_records["fit_metrics"] = ConfigsRecord()

    def fit(self, parameters, config):
        set_weights(self.net, parameters)

        train_loss = train(
            self.net,
            self.trainloader,
            self.local_epochs,
            self.device,
            self.run_config
        )

        fit_metrics = self.client_state.configs_records["fit_metrics"]
        if "train_loss_hist" not in fit_metrics:
            fit_metrics["train_loss_hist"] = [train_loss]
        else:
            fit_metrics["train_loss_hist"].append(train_loss)
    
        return (
            get_weights(self.net),
            len(self.trainloader.dataset),
            {"train_loss": train_loss},
        )

    def evaluate(self, parameters, config):
        set_weights(self.net, parameters)
        loss, dice_score, iou_score = test(self.net, self.valloader, self.device, self.run_config)
        
        return loss, len(self.valloader.dataset), {
            "dice": dice_score, 
            "iou": iou_score
        }

def client_fn(context: Context):
    # Load data
    partition_id = context.node_config["partition-id"]
    num_partitions = context.node_config["num-partitions"]
    trainloader, valloader = load_data(partition_id, num_partitions, context.run_config)
    
    # Initialize model with context parameters
    net = load_model(context.run_config)
    
    local_epochs = context.run_config["local-epochs"]
    return FlowerClient(net, trainloader, valloader, local_epochs, context).to_client()

# Flower ClientApp
app = ClientApp(
    client_fn,
)