import json
import random
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models, transforms


BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "leaf_check_v2"
OUTPUT = BASE / "artifacts"

EPOCHS = 10
BATCH_SIZE = 32


def main():
    random.seed(42)
    torch.manual_seed(42)
    torch.set_num_threads(4)

    OUTPUT.mkdir(exist_ok=True)
    checkpoint_path = OUTPUT / "leaf_check_v2_best.pt"

    if checkpoint_path.exists():
        raise SystemExit(
            "leaf_check_v2_best.pt already exists. "
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

    expected = ["not_tomato_leaf", "tomato_leaf"]

    if train_data.classes != expected:
        raise SystemExit(f"Unexpected classes: {train_data.classes}")

    if train_data.class_to_idx != val_data.class_to_idx:
        raise SystemExit("Training and validation classes differ.")

    counts = torch.bincount(
        torch.tensor(train_data.targets),
        minlength=len(expected),
    )

    if (counts == 0).any():
        raise SystemExit("A training class has no images.")

    class_weights = (
        counts.sum().float() / (len(expected) * counts.float())
    )

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

    model = models.mobilenet_v3_small(weights=weights)

    for parameter in model.parameters():
        parameter.requires_grad = False

    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(expected),
    )

    optimizer = torch.optim.Adam(
        model.classifier[3].parameters(),
        lr=0.001,
    )

    # Sum reduction allows correct weighted epoch-loss averaging.
    train_criterion = nn.CrossEntropyLoss(
        weight=class_weights,
        reduction="sum",
    )
    val_criterion = nn.CrossEntropyLoss(reduction="sum")

    best_balanced_accuracy = -1.0
    best_val_loss = float("inf")
    history = []

    print("Device: CPU", flush=True)
    print("Classes:", train_data.classes, flush=True)
    print("Training counts:", counts.tolist(), flush=True)
    print("Class weights:", class_weights.tolist(), flush=True)

    for epoch in range(1, EPOCHS + 1):
        model.eval()
        model.classifier.train()

        train_loss_sum = 0.0
        train_weight_sum = 0.0
        train_correct = 0

        for step, (images, labels) in enumerate(train_loader, 1):
            optimizer.zero_grad(set_to_none=True)

            outputs = model(images)
            loss_sum = train_criterion(outputs, labels)
            weight_sum = class_weights[labels].sum()
            loss = loss_sum / weight_sum

            loss.backward()
            optimizer.step()

            train_loss_sum += loss_sum.item()
            train_weight_sum += weight_sum.item()
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
        val_loss_sum = 0.0
        matrix = torch.zeros(2, 2, dtype=torch.long)

        with torch.inference_mode():
            for images, labels in val_loader:
                outputs = model(images)
                predictions = outputs.argmax(1)

                val_loss_sum += val_criterion(
                    outputs, labels
                ).item()

                matrix += torch.bincount(
                    labels * 2 + predictions,
                    minlength=4,
                ).reshape(2, 2)

        recall = matrix.diag().float() / matrix.sum(1).clamp_min(1)
        balanced_accuracy = recall.mean().item()
        val_loss = val_loss_sum / len(val_data)
        val_accuracy = matrix.diag().sum().item() / len(val_data)

        metrics = {
            "epoch": epoch,
            "train_loss": train_loss_sum / train_weight_sum,
            "train_accuracy": train_correct / len(train_data),
            "val_loss": val_loss,
            "val_accuracy": val_accuracy,
            "val_balanced_accuracy": balanced_accuracy,
            "not_tomato_recall": recall[0].item(),
            "tomato_recall": recall[1].item(),
        }
        history.append(metrics)

        print(
            f"Epoch {epoch}: "
            f"Train={metrics['train_accuracy']:.2%} | "
            f"Val={val_accuracy:.2%} | "
            f"Balanced={balanced_accuracy:.2%}",
            flush=True,
        )
        print(
            f"Other-image rejection={recall[0].item():.2%} | "
            f"Tomato acceptance={recall[1].item():.2%}",
            flush=True,
        )

        improved = (
            balanced_accuracy > best_balanced_accuracy
            or (
                balanced_accuracy == best_balanced_accuracy
                and val_loss < best_val_loss
            )
        )

        if improved:
            best_balanced_accuracy = balanced_accuracy
            best_val_loss = val_loss

            torch.save({
                "architecture": "mobilenet_v3_small",
                "model_state_dict": model.state_dict(),
                "classes": train_data.classes,
                "weights": "IMAGENET1K_V1",
                "epoch": epoch,
                "val_loss": val_loss,
                "val_accuracy": val_accuracy,
                "val_balanced_accuracy": balanced_accuracy,
                "class_weights": class_weights.tolist(),
                "dataset": "leaf_check_v2",
                "decision_rule": "argmax",
            }, checkpoint_path)

            print("Best V2 checker saved.", flush=True)

        (OUTPUT / "leaf_check_v2_history.json").write_text(
            json.dumps(history, indent=2),
            encoding="utf-8",
        )

    print("\nTraining finished.")
    print("Model: artifacts/leaf_check_v2_best.pt")
    print("History: artifacts/leaf_check_v2_history.json")


if __name__ == "__main__":
    main()