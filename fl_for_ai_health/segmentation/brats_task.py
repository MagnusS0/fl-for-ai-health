"""BRATS segmentation task implementation for Flower."""

from collections import OrderedDict
import torch
import torch.nn as nn
from torchvision.transforms import Resize
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader
from torchmetrics.segmentation import DiceScore
from torchmetrics import JaccardIndex
from data.brats import BRATSDataset2D
from typing import Tuple, Dict
from models.u_net.u_net import UNet
from models.tiny_segformer.segmed import SegMed
from utils.losses import DiceFocalLoss


def load_model(run_config: Dict) -> nn.Module:
    """Load the model specified in the run_config."""
    if run_config["model"] == "u-net":
        return UNet(
            in_channels=run_config["in-channels"], num_classes=run_config["num-classes"]
        )
    elif run_config["model"] == "segformer":
        return SegMed(
            img_size=run_config["img-size"],
            in_chans=run_config["in-channels"],
            num_classes=run_config["num-classes"],
        )
    else:
        raise ValueError(f"Model {run_config['model']} not supported")


full_dataset = None  # Cache full dataset


def load_data(
    partition_id: int,
    num_partitions: int,
    run_config: Dict,
    global_test_set: bool = False,
) -> Tuple[DataLoader, DataLoader]:
    """Load and partition BRATS data."""
    # Define common dataset parameters
    data_dir = run_config["data-dir"]
    dataset_json_path = run_config["dataset-json-path"]

    # Create full dataset
    global full_dataset
    if full_dataset is None:
        full_dataset = BRATSDataset2D(
            data_dir=data_dir,
            dataset_json_path=dataset_json_path,
            modality_to_use="FLAIR",
            slice_direction="axial",
            transform=Resize((run_config["img-size"], run_config["img-size"])),
            split="train" if not global_test_set else "test",
        )

    if global_test_set:
        testloader = DataLoader(
            full_dataset,
            batch_size=run_config["batch-size"],
            shuffle=False,
            num_workers=run_config["num-workers"],
            pin_memory=True,
        )
        return testloader, None

    # Calculate partition sizes
    total_size = len(full_dataset)
    partition_size = total_size // num_partitions
    start_idx = partition_id * partition_size
    end_idx = (
        start_idx + partition_size if partition_id < num_partitions - 1 else total_size
    )

    # Create train/val split for this partition (80/20)
    partition_indices = list(range(start_idx, end_idx))
    train_size = int(run_config["val-split"] * len(partition_indices))

    train_indices = partition_indices[:train_size]
    val_indices = partition_indices[train_size:]

    # Create train and validation datasets
    train_dataset = torch.utils.data.Subset(full_dataset, train_indices)
    val_dataset = torch.utils.data.Subset(full_dataset, val_indices)

    # Create data loaders with reduced number of workers
    trainloader = DataLoader(
        train_dataset,
        batch_size=run_config["batch-size"],
        shuffle=True,
        num_workers=run_config["num-workers"],
        pin_memory=True,
    )

    valloader = DataLoader(
        val_dataset,
        batch_size=run_config["batch-size"],
        shuffle=False,
        num_workers=run_config["num-workers"],
        pin_memory=True,
    )

    return trainloader, valloader


def train(net, trainloader, epochs, device, run_config: Dict):
    """Train the segmentation model."""
    net.to(device)
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=run_config["learning-rate"],
        weight_decay=run_config["weight-decay"],
    )
    # scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, epochs*len(trainloader))
    loss_fn = DiceFocalLoss(lambda_dice=1, lambda_focal=2)

    scaler = GradScaler()
    net.train()
    total_loss = 0.0
    GRAD_CLIP = 1.0

    # Initialize metrics
    train_dice = DiceScore(
        num_classes=run_config["num-classes"],
        include_background=False,
        input_format="index",
    )
    train_iou = JaccardIndex(
        task="multiclass", num_classes=run_config["num-classes"], ignore_index=0
    )
    train_dice.to(device)
    train_iou.to(device)

    for epoch in range(epochs):
        epoch_loss = 0.0
        train_dice.reset()
        train_iou.reset()
        for batch_idx, (images, labels) in enumerate(trainloader):
            images, labels = images.to(device), labels.to(device)

            # Forward pass with mixed precision
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = net(images)
                loss = loss_fn(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(net.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            # scheduler.step()
            # Calculate metrics
            epoch_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_dice.update(preds, labels)
            train_iou.update(preds, labels)

        avg_dice = train_dice.compute()
        avg_iou = train_iou.compute()
        print(f"Epoch {epoch + 1}/{epochs}")
        print(
            f"Loss: {epoch_loss / len(trainloader):.4f}, Dice: {avg_dice:.4f}, IoU: {avg_iou:.4f}"
        )
        total_loss += epoch_loss / len(trainloader)

    return total_loss / epochs


def test(net, testloader, device, run_config: Dict, viz_fn=None, server_round=None):
    """Evaluate the model."""
    net.eval()
    total_loss = 0.0
    loss_fn = DiceFocalLoss(lambda_dice=0.4, lambda_focal=0.6)
    val_dice = DiceScore(
        num_classes=run_config["num-classes"],
        include_background=False,
        input_format="index",
    )
    val_iou = JaccardIndex(
        task="multiclass", num_classes=run_config["num-classes"], ignore_index=0
    )
    val_dice.to(device)
    val_iou.to(device)

    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(testloader):
            images, masks = images.to(device), masks.to(device)

            with autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = net(images)
                loss = loss_fn(outputs, masks)

            total_loss += loss.item()
            # Evaluate metrics
            preds = torch.argmax(outputs, dim=1)
            val_dice.update(preds, masks)
            val_iou.update(preds, masks)
            if viz_fn and batch_idx <= 3:
                viz_fn(images, masks, outputs, server_round or batch_idx)

    avg_loss = total_loss / len(testloader)
    avg_dice = val_dice.compute().item()
    avg_iou = val_iou.compute().item()

    return avg_loss, avg_dice, avg_iou


def get_weights(net):
    """Get model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net, parameters):
    """Set model weights from a list of NumPy arrays."""
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
