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


@lru_cache(maxsize=1)
def load_model():
    checkpoint = torch.load(
        settings.BASE_DIR / "artifacts" / "leafcare_best.pt",
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


def predict_leaf(upload):
    model, classes, preprocess = load_model()

    upload.seek(0)
    with Image.open(upload) as image:
        tensor = preprocess(image.convert("RGB")).unsqueeze(0)

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