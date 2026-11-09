import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")

import matplotlib.pyplot as plt
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    ConfusionMatrixDisplay,
    confusion_matrix,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models


BASE = Path(__file__).resolve().parent
OUTPUT = BASE / "artifacts"


def main():
    torch.set_num_threads(4)

    checkpoint = torch.load(
        OUTPUT / "leafcare_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    classes = checkpoint["classes"]

    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    test_data = datasets.ImageFolder(
        BASE / "data" / "split" / "test",
        transform=weights.transforms(),
    )

    if test_data.classes != classes:
        raise SystemExit("Test classes do not match model classes.")

    loader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features, len(classes)
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    actual = []
    predicted = []

    with torch.inference_mode():
        for images, labels in loader:
            outputs = model(images)
            actual.extend(labels.tolist())
            predicted.extend(outputs.argmax(1).tolist())

    indices = list(range(len(classes)))
    names = [name.replace("Tomato___", "") for name in classes]

    accuracy = accuracy_score(actual, predicted)
    report = classification_report(
        actual,
        predicted,
        labels=indices,
        target_names=names,
        output_dict=True,
        zero_division=0,
    )
    matrix = confusion_matrix(actual, predicted, labels=indices)

    print(f"Saved model epoch: {checkpoint['epoch']}")
    print(f"Test images: {len(actual)}")
    print(f"Test accuracy: {accuracy:.2%}\n")
    print(classification_report(
        actual,
        predicted,
        labels=indices,
        target_names=names,
        digits=4,
        zero_division=0,
    ))

    (OUTPUT / "test_metrics.json").write_text(
        json.dumps({
            "epoch": checkpoint["epoch"],
            "test_images": len(actual),
            "accuracy": accuracy,
            "classification_report": report,
            "confusion_matrix": matrix.tolist(),
            "classes": classes,
        }, indent=2),
        encoding="utf-8",
    )

    fig, ax = plt.subplots(figsize=(8, 6))
    ConfusionMatrixDisplay(
        confusion_matrix=matrix,
        display_labels=names,
    ).plot(ax=ax, cmap="Greens", colorbar=False)

    ax.set_title("LeafCare — Test Confusion Matrix")
    plt.xticks(rotation=25, ha="right")
    fig.tight_layout()
    fig.savefig(OUTPUT / "confusion_matrix.png", dpi=160)
    plt.close(fig)

    print("Saved: artifacts/test_metrics.json")
    print("Saved: artifacts/confusion_matrix.png")


if __name__ == "__main__":
    main()