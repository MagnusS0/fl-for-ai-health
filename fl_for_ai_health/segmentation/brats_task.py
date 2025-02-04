"""
BRATS segmentation task implementation for Flower using PyTorch.

This module handles model loading, data preparation, training, and evaluation
for brain tumor segmentation tasks using the BRATS dataset.
"""

from collections import OrderedDict
from typing import Tuple, Dict, Optional, Any

import torch
import torch.nn as nn
from torchvision.transforms import Resize, Compose
from torch.amp import autocast, GradScaler
from torch.utils.data import DataLoader, Subset
from torchmetrics.segmentation import DiceScore
from torchmetrics import JaccardIndex

from data.brats import BRATSDataset2D
from models.u_net.u_net import UNet
from models.tiny_segformer.segmed import SegMed
from utils.losses import DiceFocalLoss

class BRATSDatasetCache:
    """Class-based cache for BRATS dataset to avoid global state."""
    
    def __init__(self):
        self._full_dataset = None
        self._run_config = None
    
    def initialize(self, run_config: Dict[str, Any], global_test_set: bool) -> None:
        """Initialize the dataset cache if not already loaded."""
        if self._full_dataset is None or self._run_config != run_config:
            self._run_config = run_config
            self._full_dataset = BRATSDataset2D(
                data_dir=run_config["data-dir"],
                dataset_json_path=run_config["dataset-json-path"],
                modality_to_use=["FLAIR", "T1w", "t1gd", "T2w"],
                slice_direction="axial",
                transform_image=Compose([
                    Resize((run_config["img-size"], run_config["img-size"]))
                ]),
                transform_label=Compose([
                    Resize((run_config["img-size"], run_config["img-size"]))
                ]),
                split="test" if global_test_set else "train"
            )
    
    def get_dataset(self) -> Subset:
        """Get the cached dataset."""
        if self._full_dataset is None:
            raise RuntimeError("Dataset cache not initialized")
        return self._full_dataset
    
    def create_loaders(self, partition_id: int, num_partitions: int, run_config: Dict[str, Any]) -> Tuple[DataLoader, DataLoader]:
        """Create partitioned data loaders from cached dataset."""
        # Shuffle and partition
        indices = torch.randperm(len(self._full_dataset)).tolist()
        shuffled_dataset = Subset(self._full_dataset, indices)
        
        # Calculate partition boundaries
        total_size = len(shuffled_dataset)
        partition_size = total_size // num_partitions
        start_idx = partition_id * partition_size
        end_idx = (start_idx + partition_size if partition_id < num_partitions - 1 
                   else total_size)

        # Create train/val split
        partition_indices = list(range(start_idx, end_idx))
        train_size = int(run_config["val-split"] * len(partition_indices))
        
        train_dataset = Subset(shuffled_dataset, partition_indices[:train_size])
        val_dataset = Subset(shuffled_dataset, partition_indices[train_size:])

        return (
            DataLoader(
                train_dataset,
                batch_size=run_config["batch-size"],
                shuffle=True,
                num_workers=run_config["num-workers"],
                pin_memory=True,
            ),
            DataLoader(
                val_dataset,
                batch_size=run_config["batch-size"],
                shuffle=False,
                num_workers=run_config["num-workers"],
                pin_memory=True,
            )
        )

# Module-level cache instance
_dataset_cache = BRATSDatasetCache()

def load_model(run_config: Dict[str, Any]) -> nn.Module:
    """Load the segmentation model based on run configuration.
    
    Args:
        run_config: Configuration dictionary

    Returns:
        Initialized neural network model

    Raises:
        ValueError: If unsupported model type is specified
    """
    model_type = run_config["model"].lower()
    
    if model_type == "u-net":
        return UNet(
            in_channels=run_config["in-channels"],
            num_classes=run_config["num-classes"]
        )
    if model_type == "segformer":
        return SegMed(
            img_size=run_config["img-size"],
            in_chans=run_config["in-channels"],
            num_classes=run_config["num-classes"],
        )
    
    raise ValueError(f"Unsupported model type: {model_type}. "
                     "Supported options: 'u-net', 'segformer'")


def load_data(
    partition_id: int,
    num_partitions: int,
    run_config: Dict[str, Any],
    global_test_set: bool = False
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """Load and partition BRATS data with proper shuffling (updated to use class-based cache)."""
    # Initialize dataset cache
    _dataset_cache.initialize(run_config, global_test_set)
    
    if global_test_set:
        return DataLoader(
            _dataset_cache.get_dataset(),
            batch_size=run_config["batch-size"],
            shuffle=False,
            num_workers=run_config["num-workers"],
            pin_memory=True,
        ), None

    return _dataset_cache.create_loaders(partition_id, num_partitions, run_config)


def train(
    net: nn.Module,
    trainloader: DataLoader,
    epochs: int,
    device: torch.device,
    run_config: Dict[str, Any],
    scheduler_state: Optional[Dict[str, Any]] = None
) -> Tuple[float, Dict[str, Any]]:
    """Train the segmentation model with mixed precision and gradient clipping.
    
    Args:
        net: Model to train
        trainloader: Training data loader
        epochs: Number of training epochs
        device: Target device (CPU/GPU)
        run_config: Training configuration dictionary containing:
            - 'num-classes': Number of segmentation classes
            - 'learning-rate': Initial learning rate
            - 'weight-decay': Weight decay for optimizer
            - 'num-server-rounds': Total federated rounds
            - 'local-epochs': Local epochs per round
        scheduler_state: Optional scheduler state dict for resuming training

    Returns:
        Tuple of (average loss per epoch, scheduler state dictionary)
    """
    net.to(device)
    net.train()

    # Initialize training components
    optimizer = torch.optim.AdamW(
        net.parameters(),
        lr=run_config["learning-rate"],
        weight_decay=run_config["weight-decay"],
    )
    
    loss_fn = DiceFocalLoss(
        lambda_dice=1,
        lambda_focal=2,
        num_classes=run_config["num-classes"],
        ignore_index=0,
        device=device
    )

    # Configure learning rate scheduler
    total_steps = (run_config["num-server-rounds"] 
                   * run_config["local-epochs"] 
                   * len(trainloader))
    scheduler = _configure_scheduler(optimizer, total_steps, scheduler_state)
    scaler = GradScaler()

    # Initialize metrics
    train_dice = DiceScore(
        num_classes=run_config["num-classes"],
        include_background=False,
        input_format="index",
    ).to(device)
    
    train_iou = JaccardIndex(
        task="multiclass",
        num_classes=run_config["num-classes"],
        ignore_index=0
    ).to(device)

    # Training loop
    total_loss = 0.0
    for epoch in range(epochs):
        epoch_loss = 0.0
        train_dice.reset()
        train_iou.reset()

        for images, labels in trainloader:
            images, labels = images.to(device), labels.to(device)
            epoch_loss += _train_step(
                net, images, labels, optimizer, scaler, loss_fn, 
                train_dice, train_iou, scheduler, device
            )

        # Log epoch metrics
        avg_loss = epoch_loss / len(trainloader)
        total_loss += avg_loss
        print(f"Epoch {epoch+1}/{epochs}: "
              f"Loss: {avg_loss:.4f}, "
              f"Dice: {train_dice.compute():.4f}, "
              f"IoU: {train_iou.compute():.4f}")

    return total_loss / epochs, scheduler.state_dict()


def _configure_scheduler(
    optimizer: torch.optim.Optimizer,
    total_steps: int,
    scheduler_state: Optional[Dict[str, Any]] = None
) -> torch.optim.lr_scheduler.LRScheduler:
    """Configure learning rate scheduler with warmup."""
    scheduler = torch.optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            torch.optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,
                end_factor=1.0,
                total_iters=int(0.1 * total_steps),
            ),
            torch.optim.lr_scheduler.PolynomialLR(
                optimizer,
                total_iters=int(0.9 * total_steps),
                power=1.0,
            ),
        ],
        milestones=[int(0.1 * total_steps)],
    )
    
    if scheduler_state:
        scheduler.load_state_dict(scheduler_state)
        optimizer.param_groups[0]["lr"] = scheduler.get_last_lr()[0]
    
    return scheduler


def _train_step(
    net: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    optimizer: torch.optim.Optimizer,
    scaler: GradScaler,
    loss_fn: nn.Module,
    dice_metric: DiceScore,
    iou_metric: JaccardIndex,
    scheduler: torch.optim.lr_scheduler.LRScheduler,
    device: torch.device
) -> float:
    """Execute single training step with mixed precision."""
    # Forward pass
    optimizer.zero_grad()
    dtype = torch.bfloat16 if device.type == "cuda" else torch.float32
    
    with autocast(device_type=device.type, dtype=dtype):
        outputs = net(images)
        loss = loss_fn(outputs, labels)

    # Backward pass
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
    scaler.step(optimizer)
    scaler.update()
    scheduler.step()

    # Update metrics
    preds = torch.argmax(outputs, dim=1)
    dice_metric.update(preds, labels)
    iou_metric.update(preds, labels)
    
    return loss.item()


def test(
    net: nn.Module,
    testloader: DataLoader,
    device: torch.device,
    run_config: Dict[str, Any],
    viz_fn: Optional[Any] = None,
    server_round: Optional[int] = None
) -> Tuple[float, float, float]:
    """Evaluate model performance on test set.
    
    Args:
        net: Model to evaluate
        testloader: Test data loader
        device: Target device (CPU/GPU)
        run_config: Configuration dictionary containing 'num-classes'
        viz_fn: Optional visualization function (not implemented in current version)
        server_round: Optional server round number for logging

    Returns:
        Tuple of (average loss, dice score, iou score)
    """
    net.eval()
    loss_fn = DiceFocalLoss(
        lambda_dice=1,
        lambda_focal=2,
        num_classes=run_config["num-classes"],
        ignore_index=0,
        device=device
    )
    
    val_dice = DiceScore(
        num_classes=run_config["num-classes"],
        include_background=False,
        input_format="index",
    ).to(device)
    
    val_iou = JaccardIndex(
        task="multiclass",
        num_classes=run_config["num-classes"],
        ignore_index=0
    ).to(device)

    total_loss = 0.0
    with torch.no_grad():
        for batch_idx, (images, masks) in enumerate(testloader):
            images, masks = images.to(device), masks.to(device)
            
            with autocast(device_type=device.type, dtype=torch.bfloat16):
                outputs = net(images)
                loss = loss_fn(outputs, masks)
            
            total_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            val_dice.update(preds, masks)
            val_iou.update(preds, masks)
            if viz_fn and batch_idx % 200 == 0:
                viz_fn(images, masks, outputs, batch_idx, server_round)

    return (
        total_loss / len(testloader),
        val_dice.compute().item(),
        val_iou.compute().item()
    )


def get_weights(net):
    """Get model weights as a list of NumPy arrays."""
    return [val.cpu().numpy() for _, val in net.state_dict().items()]


def set_weights(net: nn.Module, parameters: list[torch.Tensor]) -> None:
    """
    Set model weights from list of tensors.
    """
    params_dict = zip(net.state_dict().keys(), parameters)
    state_dict = OrderedDict({k: torch.tensor(v) for k, v in params_dict})
    net.load_state_dict(state_dict, strict=True)
