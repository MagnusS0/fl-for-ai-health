"""fl-for-AI-health: A Flower / PyTorch app."""

from collections import OrderedDict

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.amp import autocast, GradScaler
from flwr_datasets import FederatedDataset
from flwr_datasets.partitioner import IidPartitioner
from torch.utils.data import DataLoader
from torchvision.transforms import Compose, Normalize, ToTensor
from sklearn.metrics import roc_auc_score
import numpy as np
from typing import Dict
from models.resnet.resnet18 import ResNet18
from models.tiny_vit.tiny_vit import tiny_vit_5m_224


def load_model(run_config: Dict) -> nn.Module:
    """Load the model specified in the run_config."""
    if run_config["model"] == "resnet-18":
        return ResNet18(
            in_channels=run_config["in-channels"],
            num_classes=run_config["num-classes"]
        )
    elif run_config["model"] == "tiny-vit":
        return tiny_vit_5m_224(
            img_size=run_config["img-size"],
            in_chans=run_config["in-channels"],
            num_classes=run_config["num-classes"]
        )
    else:
        raise ValueError(f"Model {run_config['model']} not supported")


fds = None  # Cache FederatedDataset



def load_data(partition_id: int, num_partitions: int, split: str = "train"):
    """Load partition MedMNIST data."""
    # Only initialize `FederatedDataset` once
    global fds
    if fds is None:
        partitioner = IidPartitioner(num_partitions=num_partitions)
        fds = FederatedDataset(
            dataset="MagnusSa/medmnist",
            partitioners={split: partitioner},
        )
    partition = fds.load_partition(partition_id)
    # Divide data on each node: 80% train, 20% test
    partition_train_test = partition.train_test_split(test_size=0.2, seed=42)
    pytorch_transforms = Compose(
        [ToTensor(), Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    )

    def apply_transforms(batch):
        """Apply transforms to the partition from FederatedDataset."""
        batch["image"] = [pytorch_transforms(img) for img in batch["image"]]
        return batch

    partition_train_test = partition_train_test.with_transform(apply_transforms)
    trainloader = DataLoader(partition_train_test["train"], batch_size=32, shuffle=True)
    testloader = DataLoader(partition_train_test["test"], batch_size=32)
    return trainloader, testloader


scaler = GradScaler() # Gradient scaling for mixed precision training

def train(net, trainloader, epochs, device):
    """Train the model on the training set."""
    net.to(device)  # move model to GPU if available
    criterion = torch.nn.CrossEntropyLoss().to(device)
    optimizer = torch.optim.SGD(net.parameters(), lr=0.001, momentum=0.9)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.001, total_steps=len(trainloader) * epochs)
    net.train()
    running_loss = 0.0
    for _ in range(epochs):
        for batch in trainloader:
            images = batch["image"]
            labels = batch["label"]
            optimizer.zero_grad()
            # Mixed precision training
            dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
            with autocast(device.type, dtype=dtype):
                loss = criterion(net(images.to(device)), labels.to(device))
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            scheduler.step()
            running_loss += loss.item()

    avg_trainloss = running_loss / len(trainloader)
    return avg_trainloss


def test(net, testloader, device):
    """Validate the model on the test set."""
    net.to(device)
    criterion = torch.nn.CrossEntropyLoss()
    correct, loss = 0, 0.0
    all_probs = []
    all_labels = []
    with torch.no_grad():
        for batch in testloader:
            images = batch["image"].to(device)
            labels = batch["label"].to(device)
            outputs = net(images)
            probs = F.softmax(outputs, dim=1)
            loss += criterion(outputs, labels).item()
            correct += (torch.max(outputs.data, 1)[1] == labels).sum().item()
            
            # Store probabilities and labels for AUC calculation
            all_probs.extend(probs.cpu().numpy())
            all_labels.extend(labels.cpu().numpy())
            
    accuracy = correct / len(testloader.dataset)
    loss = loss / len(testloader)
    accuracy = correct / len(testloader.dataset)

    # Calculate AUC (handles both binary and multiclass cases)
    try:
        if len(np.unique(all_labels)) == 2: 
            auc = roc_auc_score(all_labels, np.array(all_probs)[:, 1])
        else:  # Multi-class classification
            auc = roc_auc_score(all_labels, all_probs, multi_class='ovr')
    except ValueError:
        auc = float('nan')
        
    return loss, accuracy, auc


def get_weights(net):
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
