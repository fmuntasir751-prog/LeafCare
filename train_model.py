import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "split"
OUTPUT = BASE / "artifacts"
EPOCHS = 8
BATCH_SIZE = 32


def main():
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)

    OUTPUT.mkdir(exist_ok=True)
    checkpoint = OUTPUT / "leafcare_best.pt"

    if checkpoint.exists():
        raise SystemExit(
            "leafcare_best.pt already exists. "
            "Rename it before starting another training run."
        )

    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    preprocess = weights.transforms()

    train_transform = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomRotation(15),
        preprocess,
    ])

    train_data = datasets.ImageFolder(
        DATA / "train", transform=train_transform
    )
    val_data = datasets.ImageFolder(
        DATA / "val", transform=preprocess
    )

    if train_data.class_to_idx != val_data.class_to_idx:
        raise SystemExit("Training and validation classes do not match.")

    train_loader = DataLoader(
        train_data,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=0,
    )

    print("Device: CPU", flush=True)
    print("Classes:", train_data.classes, flush=True)
    print("Loading pretrained model...", flush=True)

    model = models.mobilenet_v3_small(weights=weights)

    # Freeze pretrained layers; train only the new output layer.
    for parameter in model.parameters():
        parameter.requires_grad = False

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(train_data.classes),
    )

    optimizer = torch.optim.Adam(
        model.classifier[3].parameters(), lr=0.001
    )
    criterion = nn.CrossEntropyLoss()

    best_val_loss = float("inf")
    history = []

    for epoch in range(1, EPOCHS + 1):
        # Keep frozen BatchNorm layers fixed.
        model.eval()
        model.classifier.train()

        train_loss = 0.0
        train_correct = 0

        for step, (images, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            train_loss += loss.item() * labels.size(0)
            train_correct += (
                outputs.argmax(1) == labels
            ).sum().item()

            if step % 20 == 0 or step == len(train_loader):
                print(
                    f"Epoch {epoch}/{EPOCHS} | "
                    f"Batch {step}/{len(train_loader)}",
                    flush=True,
                )

        model.eval()
        val_loss = 0.0
        val_correct = 0

        with torch.inference_mode():
            for images, labels in val_loader:
                outputs = model(images)
                loss = criterion(outputs, labels)

                val_loss += loss.item() * labels.size(0)
                val_correct += (
                    outputs.argmax(1) == labels
                ).sum().item()

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss / len(train_data),
            "train_accuracy": train_correct / len(train_data),
            "val_loss": val_loss / len(val_data),
            "val_accuracy": val_correct / len(val_data),
        }
        history.append(metrics)

        print(
            f"Epoch {epoch}: "
            f"Train accuracy={metrics['train_accuracy']:.2%} | "
            f"Val accuracy={metrics['val_accuracy']:.2%} | "
            f"Val loss={metrics['val_loss']:.4f}",
            flush=True,
        )

        if metrics["val_loss"] < best_val_loss:
            best_val_loss = metrics["val_loss"]

            torch.save({
                "architecture": "mobilenet_v3_small",
                "model_state_dict": model.state_dict(),
                "classes": train_data.classes,
                "weights": "IMAGENET1K_V1",
                "epoch": epoch,
                "val_loss": best_val_loss,
                "val_accuracy": metrics["val_accuracy"],
            }, checkpoint)

            print("Best model saved.", flush=True)

        (OUTPUT / "history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    print("\nTraining finished.", flush=True)
    print("Model: artifacts/leafcare_best.pt", flush=True)
    print("History: artifacts/history.json", flush=True)


if __name__ == "__main__":
    main()