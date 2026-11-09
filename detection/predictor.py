from functools import lru_cache

import torch
from django.conf import settings
from PIL import Image
from torch import nn
from torchvision import models


LABELS = {
    "Tomato___Early_blight": "Tomato — Early blight",
    "Tomato___Late_blight": "Tomato — Late blight",
    "Tomato___Leaf_Mold": "Tomato — Leaf mold",
    "Tomato___healthy": "Tomato — Healthy",
}


class NotTomatoLeafError(ValueError):
    pass


def build_model(filename):
    checkpoint = torch.load(
        settings.BASE_DIR / "artifacts" / filename,
        map_location="cpu",
        weights_only=True,
    )

    classes = checkpoint["classes"]

    model = models.mobilenet_v3_small(weights=None)
    model.classifier[3] = nn.Linear(
        model.classifier[3].in_features,
        len(classes),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model.eval()

    preprocess = (
        models.MobileNet_V3_Small_Weights.IMAGENET1K_V1.transforms()
    )

    return model, classes, preprocess


@lru_cache(maxsize=1)
def load_model():
    return build_model("leafcare_best.pt")


@lru_cache(maxsize=1)
def load_checker():
    model, classes, preprocess = build_model(
        "leaf_check_v2_best.pt"
    )

    if classes != ["not_tomato_leaf", "tomato_leaf"]:
        raise ValueError("Unexpected leaf checker classes.")

    return model, classes, preprocess


def predict_leaf(upload):
    upload.seek(0)

    with Image.open(upload) as image:
        rgb = image.convert("RGB")

    checker, checker_classes, checker_preprocess = load_checker()
    checker_input = checker_preprocess(rgb).unsqueeze(0)

    with torch.inference_mode():
        checker_scores = checker(checker_input)[0]
        checker_index = int(checker_scores.argmax().item())

    # Same argmax decision rule used during checker evaluation.
    if checker_classes[checker_index] != "tomato_leaf":
        raise NotTomatoLeafError(
            "This image was not accepted as a tomato leaf. "
            "Please upload a clear photo of a single tomato leaf. "
            "The checker can also make mistakes."
        )

    model, classes, preprocess = load_model()
    tensor = preprocess(rgb).unsqueeze(0)

    with torch.inference_mode():
        scores = torch.softmax(model(tensor), dim=1)[0]

    ranked = sorted(
        [
            {
                "label": LABELS.get(name, name),
                "score": float(scores[index]) * 100,
            }
            for index, name in enumerate(classes)
        ],
        key=lambda item: item["score"],
        reverse=True,
    )

    return {
        "label": ranked[0]["label"],
        "score": ranked[0]["score"],
        "ranked": ranked,
    }