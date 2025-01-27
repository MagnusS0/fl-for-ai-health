import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import os
from torchmetrics import JaccardIndex, Dice
from models.tiny_segformer.segmed import SegMed
from utils.losses import DiceFocalLoss
from data.brats import BRATSDataset2D
from torchvision.transforms import Compose, Resize
from torch.amp import GradScaler, autocast
import matplotlib.pyplot as plt
from tqdm import tqdm


# Configuration
class Config:
    data_dir = "/home/magnus/Datasets/Images/MedicalDecathlon/Task01_BrainTumour"
    dataset_json_path = (
        "/home/magnus/Datasets/Images/MedicalDecathlon/Task01_BrainTumour/dataset.json"
    )
    save_dir = "checkpoints"
    model_name = "UNet"

    # Model parameters
    in_channels = 1  # FLAIR is single-channel
    num_classes = 4
    img_size = 128

    # Training parameters
    epochs = 100
    batch_size = 64
    lr = 4e-4
    weight_decay = 1e-6
    val_split = 0.2
    num_workers = 8
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")


def visualize_predictions(images, labels, outputs):
    # Convert images, labels, and outputs to numpy arrays
    plt.switch_backend("Agg")

    outputs = torch.argmax(outputs, dim=1)
    images = images.cpu().numpy()
    labels = labels.cpu().numpy()
    outputs = outputs.cpu().numpy()

    for i in range(images.shape[0]):
        fig, axes = plt.subplots(1, 3, figsize=(10, 5))
        axes[0].imshow(images[i, 0, :, :], cmap="gray")
        axes[0].axis("off")
        axes[0].set_title("Image")
        axes[1].imshow(labels[i, :, :], cmap="tab10", vmin=0, vmax=3)
        axes[1].axis("off")
        axes[1].set_title("Label")
        axes[2].imshow(outputs[i, :, :], cmap="tab10", vmin=0, vmax=3)
        axes[2].axis("off")
        axes[2].set_title("Prediction")
        plt.savefig(f"predictions/prediction_{i}.png")
        plt.close()


scaler = GradScaler()


def train():
    # Initialize dataset and dataloader
    train_dataset = BRATSDataset2D(
        data_dir=Config.data_dir,
        dataset_json_path=Config.dataset_json_path,
        modality_to_use="FLAIR",
        slice_direction="axial",
        transform=Compose(
            [
                Resize((Config.img_size, Config.img_size)),
            ]
        ),
        split="train",
    )

    val_dataset = BRATSDataset2D(
        data_dir=Config.data_dir,
        dataset_json_path=Config.dataset_json_path,
        modality_to_use="FLAIR",
        slice_direction="axial",
        transform=Compose(
            [
                Resize((Config.img_size, Config.img_size)),
            ]
        ),
        split="test",
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=Config.batch_size,
        shuffle=True,
        num_workers=Config.num_workers,
        pin_memory=True,
        prefetch_factor=2,
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=Config.batch_size,
        shuffle=False,
        num_workers=Config.num_workers,
        pin_memory=True,
    )

    # Initialize model, loss, and optimizer
    model = SegMed(
        img_size=Config.img_size,
        in_chans=Config.in_channels,
        num_classes=Config.num_classes,
    )
    criterion = DiceFocalLoss(lambda_dice=1, lambda_focal=2)
    # criterion = nn.CrossEntropyLoss()
    optimizer = optim.AdamW(
        model.parameters(), lr=Config.lr, weight_decay=Config.weight_decay
    )
    lr_scheduler = optim.lr_scheduler.SequentialLR(
        optimizer,
        schedulers=[
            optim.lr_scheduler.LinearLR(
                optimizer,
                start_factor=0.01,  # Start at 4e-6 (4e-4 * 0.01 = 4e-6)
                end_factor=1.0,  # End at 4e-4
                total_iters=int(
                    0.1 * Config.epochs * len(train_loader)
                ),  # 10% of total steps for warmup
            ),
            optim.lr_scheduler.PolynomialLR(
                optimizer,
                total_iters=int(
                    0.9 * Config.epochs * len(train_loader)
                ),  # Remaining 90% of steps
                power=2.0,
            ),
        ],
        milestones=[
            int(0.1 * Config.epochs * len(train_loader))
        ],  # Switch from warmup to decay at 10% of training
    )

    # Metrics
    train_jaccard = JaccardIndex(
        num_classes=Config.num_classes, task="multiclass", ignore_index=0
    )
    val_jaccard = JaccardIndex(
        num_classes=Config.num_classes, task="multiclass", ignore_index=0
    )
    train_dice = Dice(num_classes=Config.num_classes, ignore_index=0)
    val_dice = Dice(num_classes=Config.num_classes, ignore_index=0)

    # Move model and metrics to device
    model = model.to(Config.device)

    # Initialize best metrics
    best_jaccard = 0.0
    best_epoch = 0
    GRAD_CLIP = 1.0

    # Training loop
    for epoch in range(1, Config.epochs + 1):
        print(f"Epoch {epoch}/{Config.epochs}")
        print("------------------------")

        train_jaccard.reset()
        train_dice.reset()
        val_jaccard.reset()
        val_dice.reset()

        # Training phase
        model.train()
        train_loss = 0.0
        viz_counter = 0
        train_pbar = tqdm(train_loader, desc=f"Training Epoch {epoch}", leave=True)
        for batch_idx, (images, labels) in enumerate(train_pbar):
            images, labels = images.to(Config.device), labels.to(Config.device)

            # Forward pass
            with autocast(device_type=Config.device.type, dtype=torch.bfloat16):
                outputs = model(images)
                loss = criterion(outputs, labels)

            # Backward pass and optimize
            optimizer.zero_grad()
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), GRAD_CLIP)
            scaler.step(optimizer)
            scaler.update()
            lr_scheduler.step()

            # Calculate metrics
            train_loss += loss.item()
            preds = torch.argmax(outputs, dim=1)
            train_jaccard.update(preds.cpu(), labels.cpu())
            train_dice.update(preds.cpu(), labels.cpu())

            # Update progress bar
            train_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

        avg_train_loss = train_loss / len(train_loader)
        avg_train_jaccard = train_jaccard.compute()
        avg_train_dice = train_dice.compute()

        print(f"\nTrain Loss: {avg_train_loss:.4f}")
        print(f"Train Jaccard Index: {avg_train_jaccard:.4f}")
        print(f"Train Dice Coefficient: {avg_train_dice:.4f}")
        train_jaccard.reset()
        train_dice.reset()

        # Validation phase
        model.eval()
        val_loss = 0.0

        with torch.no_grad():
            val_pbar = tqdm(val_loader, desc="Validation", leave=True)
            for images, labels in val_pbar:
                viz_counter += 1
                images, labels = images.to(Config.device), labels.to(Config.device)

                # Forward pass
                outputs = model(images)

                # Compute loss
                loss = criterion(outputs, labels)
                val_loss += loss.item()

                # Calculate metrics
                preds = torch.argmax(outputs, dim=1)
                val_jaccard.update(preds.cpu(), labels.cpu())
                val_dice.update(preds.cpu(), labels.cpu())

                # Update progress bar
                val_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

                # Visualize predictions
                if viz_counter % 40 == 0:
                    visualize_predictions(images, labels, outputs)

        avg_val_loss = val_loss / len(val_loader)
        avg_val_jaccard = val_jaccard.compute()
        avg_val_dice = val_dice.compute()

        print(f"Validation Loss: {avg_val_loss:.4f}")
        print(f"Validation Jaccard Index: {avg_val_jaccard:.4f}")
        print(f"Validation Dice Coefficient: {avg_val_dice:.4f}")
        val_jaccard.reset()
        val_dice.reset()

        # Save best model
        if avg_val_jaccard > best_jaccard:
            best_jaccard = avg_val_jaccard
            best_epoch = epoch
            save_model(model, epoch, avg_train_jaccard, avg_val_jaccard)

    print(
        f"Best model achieved at epoch {best_epoch} with Jaccard Index: {best_jaccard:.4f}"
    )
    return model


def save_model(model, epoch, train_jaccard, val_jaccard):
    os.makedirs(Config.save_dir, exist_ok=True)
    filename = f"{Config.model_name}_epoch{epoch}_train_jaccard{train_jaccard:.4f}_val_jaccard{val_jaccard:.4f}.pth"
    path = os.path.join(Config.save_dir, filename)
    torch.save(model.state_dict(), path)
    print(f"Saved model to {path}")


if __name__ == "__main__":
    train()
