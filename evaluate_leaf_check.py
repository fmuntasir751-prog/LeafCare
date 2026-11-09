import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
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
        OUTPUT / "leaf_check_best.pt",
        map_location="cpu",
        weights_only=True,
    )
    classes = checkpoint["classes"]

    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1

    test_data = datasets.ImageFolder(
        BASE / "data" / "leaf_check_split" / "test",
        transform=weights.transforms(),
    )

    if classes != ["not_tomato_leaf", "tomato_leaf"]:
        raise SystemExit(f"Unexpected model classes: {classes}")

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
        model.classifier[3].in_features,
        len(classes),
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
    matrix = confusion_matrix(
        actual, predicted, labels=indices
    )

    report = classification_report(
        actual,
        predicted,
        labels=indices,
        target_names=classes,
        output_dict=True,
        zero_division=0,
    )

    accuracy = accuracy_score(actual, predicted)

    # Rows: actual; columns: predicted.
    false_accepts = int(matrix[0, 1])
    false_rejects = int(matrix[1, 0])

    print(f"Selected epoch: {checkpoint['epoch']}")
    print(f"Test images: {len(actual)}")
    print(f"Test accuracy: {accuracy:.2%}\n")

    print(classification_report(
        actual,
        predicted,
        labels=indices,
        target_names=classes,
        digits=4,
        zero_division=0,
    ))

    print(
        "Other images incorrectly accepted as tomato:",
        false_accepts,
    )
    print(
        "Tomato images incorrectly rejected:",
        false_rejects,
    )

    metrics = {
        "epoch": checkpoint["epoch"],
        "test_images": len(actual),
        "accuracy": accuracy,
        "classes": classes,
        "classification_report": report,
        "confusion_matrix": matrix.tolist(),
        "false_accepts": false_accepts,
        "false_rejects": false_rejects,
        "decision_rule": "argmax",
        "scope": "PlantVillage tomato vs selected Imagenette images",
    }

    path = OUTPUT / "leaf_check_test_metrics.json"
    path.write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )

    print("\nSaved: artifacts/leaf_check_test_metrics.json")


if __name__ == "__main__":
    main()