import csv
import random
import shutil
from hashlib import sha256
from pathlib import Path

from PIL import Image


BASE = Path(__file__).resolve().parent
TOMATO = BASE / "data" / "split"
OTHER = BASE / "data" / "external" / "imagenette2-160"
OUTPUT = BASE / "data" / "leaf_check_split"

EXTENSIONS = {".jpg", ".jpeg", ".png"}

TOMATO_CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
]


def main():
    if OUTPUT.exists():
        raise SystemExit(
            "data/leaf_check_split already exists. "
            "Stopping to avoid mixing datasets."
        )

    rng = random.Random(42)
    seen = set()
    records = []

    def select_images(folder, count, split, label):
        if not folder.is_dir():
            raise SystemExit(f"Missing folder: {folder}")

        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        rng.shuffle(files)

        selected = 0

        for path in files:
            try:
                with Image.open(path) as image:
                    rgb = image.convert("RGB")
                    digest = sha256(
                        str(rgb.size).encode() + rgb.tobytes()
                    ).hexdigest()
            except (OSError, ValueError, Image.DecompressionBombError):
                continue

            # Exclude identical pixels across all selected splits.
            if digest in seen:
                continue

            seen.add(digest)
            records.append({
                "split": split,
                "label": label,
                "source": path.relative_to(BASE).as_posix(),
                "pixel_sha256": digest,
            })

            selected += 1
            if selected == count:
                break

        if selected != count:
            raise SystemExit(
                f"Not enough valid unique images: {folder}\n"
                f"Needed {count}, found {selected}."
            )

    # Preserve the existing tomato split assignments.
    for split, per_class in [
        ("train", 250),
        ("val", 50),
        ("test", 50),
    ]:
        for class_name in TOMATO_CLASSES:
            select_images(
                TOMATO / split / class_name,
                per_class,
                split,
                "tomato_leaf",
            )

    other_classes = sorted(
        p.name for p in (OTHER / "train").iterdir()
        if p.is_dir()
    )

    if len(other_classes) != 10:
        raise SystemExit(
            f"Expected 10 Imagenette classes, found {len(other_classes)}."
        )

    for class_name in other_classes:
        # Training and validation both come from Imagenette train.
        # 'seen' ensures these selections do not overlap.
        select_images(
            OTHER / "train" / class_name,
            100,
            "train",
            "not_tomato_leaf",
        )
        select_images(
            OTHER / "train" / class_name,
            20,
            "val",
            "not_tomato_leaf",
        )

        # Reserve Imagenette val for our test split.
        select_images(
            OTHER / "val" / class_name,
            20,
            "test",
            "not_tomato_leaf",
        )

    # Copy only after all selections pass validation.
    for record in records:
        source = BASE / record["source"]
        destination = (
            OUTPUT
            / record["split"]
            / record["label"]
            / f"{record['pixel_sha256']}{source.suffix.lower()}"
        )
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    with (OUTPUT / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split", "label", "source", "pixel_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(records)

    for split in ["train", "val", "test"]:
        for label in ["tomato_leaf", "not_tomato_leaf"]:
            count = sum(
                r["split"] == split and r["label"] == label
                for r in records
            )
            print(f"{split:5} | {label:16} | {count}")

    print("\nSaved: data/leaf_check_split")
    print("Original datasets unchanged.")


if __name__ == "__main__":
    main()