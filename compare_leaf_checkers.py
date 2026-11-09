import csv
import json
from pathlib import Path

import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
)
from torch import nn
from torch.utils.data import DataLoader
from torchvision import datasets, models


BASE = Path(__file__).resolve().parent
DATA = BASE / "data" / "leaf_check_v2"
OUTPUT = BASE / "artifacts"

CLASSES = ["not_tomato_leaf", "tomato_leaf"]


def main():
    torch.set_num_threads(4)

    weights = models.MobileNet_V3_Small_Weights.IMAGENET1K_V1
    test_data = datasets.ImageFolder(
        DATA / "test",
        transform=weights.transforms(),
    )

    if test_data.classes != CLASSES:
        raise SystemExit("Unexpected test classes.")

    # Match each image to its source group.
    with (DATA / "manifest.csv").open(
        encoding="utf-8", newline=""
    ) as file:
        groups = {
            row["pixel_sha256"]: row["source_group"]
            for row in csv.DictReader(file)
            if row["split"] == "test"
        }

    source_groups = [
        groups[Path(path).stem]
        for path, _ in test_data.samples
    ]

    loader = DataLoader(
        test_data,
        batch_size=32,
        shuffle=False,
        num_workers=0,
    )

    results = {}

    for version, filename in [
        ("v1", "leaf_check_best.pt"),
        ("v2", "leaf_check_v2_best.pt"),
    ]:
        checkpoint = torch.load(
            OUTPUT / filename,
            map_location="cpu",
            weights_only=True,
        )

        if checkpoint["classes"] != CLASSES:
            raise SystemExit(f"Unexpected classes in {filename}")

        model = models.mobilenet_v3_small(weights=None)
        model.classifier[3] = nn.Linear(
            model.classifier[3].in_features, 2
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

        matrix = confusion_matrix(
            actual, predicted, labels=[0, 1]
        )

        def subgroup_stats(indices):
            total = len(indices)
            wrong = sum(predicted[i] == 1 for i in indices)

            return {
                "images": total,
                "incorrectly_accepted": wrong,
                "false_acceptance_rate": (
                    wrong / total if total else None
                ),
            }

        other_leaf_indices = [
            i for i, label in enumerate(actual)
            if label == 0
            and source_groups[i] != "previous_checker"
        ]

        object_indices = [
            i for i, label in enumerate(actual)
            if label == 0
            and source_groups[i] == "previous_checker"
        ]

        other_leaves = subgroup_stats(other_leaf_indices)
        objects = subgroup_stats(object_indices)

        result = {
            "epoch": checkpoint["epoch"],
            "accuracy": accuracy_score(actual, predicted),
            "balanced_accuracy": balanced_accuracy_score(
                actual, predicted
            ),
            "confusion_matrix": matrix.tolist(),
            "tomato_images": int(matrix[1].sum()),
            "tomato_false_rejections": int(matrix[1, 0]),
            "tomato_acceptance_rate": (
                float(matrix[1, 1] / matrix[1].sum())
            ),
            "other_leaves": other_leaves,
            "objects": objects,
        }

        results[version] = result

        print(f"\n{version.upper()} — epoch {result['epoch']}")
        print(f"Test accuracy: {result['accuracy']:.2%}")
        print(
            "Balanced accuracy: "
            f"{result['balanced_accuracy']:.2%}"
        )
        print(
            "Tomato acceptance: "
            f"{result['tomato_acceptance_rate']:.2%}"
        )
        print(
            "Other leaves incorrectly accepted: "
            f"{other_leaves['incorrectly_accepted']}"
            f"/{other_leaves['images']}"
        )
        print(
            "Objects incorrectly accepted: "
            f"{objects['incorrectly_accepted']}"
            f"/{objects['images']}"
        )

    (OUTPUT / "leaf_checker_comparison.json").write_text(
        json.dumps(
            {
                "dataset": "leaf_check_v2/test",
                "decision_rule": "argmax",
                "results": results,
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    print("\nSaved: artifacts/leaf_checker_comparison.json")


if __name__ == "__main__":
    main()