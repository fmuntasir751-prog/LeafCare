import csv
import random
import shutil
from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from PIL import Image

BASE = Path(__file__).resolve().parent
SOURCE = BASE / "data" / "raw"
OUTPUT = BASE / "data" / "split"

CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
]


def image_hash(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return sha256(
            str(rgb.size).encode() + rgb.tobytes()
        ).hexdigest()


def main():
    if OUTPUT.exists():
        raise SystemExit(
            "data/split already exists. Stopping to avoid mixing splits."
        )

    groups = defaultdict(list)

    # Check all source images before creating output.
    for class_name in CLASSES:
        folder = SOURCE / class_name
        files = sorted(
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )

        if not files:
            raise SystemExit(f"No images found: {folder}")

        for path in files:
            groups[image_hash(path)].append(path)

    unique = defaultdict(list)

    for digest, paths in groups.items():
        labels = {path.parent.name for path in paths}

        if len(labels) != 1:
            raise SystemExit(f"Conflicting labels: {paths}")

        # Keep one representative of each identical-pixel group.
        unique[paths[0].parent.name].append((digest, paths[0]))

    rng = random.Random(42)
    manifest = []
    removed = sum(len(paths) - 1 for paths in groups.values())

    print(f"{'Class':28} {'Train':>7} {'Val':>7} {'Test':>7}")

    for class_name in CLASSES:
        items = sorted(unique[class_name])
        rng.shuffle(items)

        n_train = int(len(items) * 0.70)
        n_val = int(len(items) * 0.15)

        parts = {
            "train": items[:n_train],
            "val": items[n_train:n_train + n_val],
            "test": items[n_train + n_val:],
        }

        for split_name, entries in parts.items():
            destination = OUTPUT / split_name / class_name
            destination.mkdir(parents=True, exist_ok=True)

            for digest, source in entries:
                target = destination / source.name
                shutil.copy2(source, target)
                manifest.append({
                    "split": split_name,
                    "class": class_name,
                    "source": source.relative_to(BASE).as_posix(),
                    "destination": target.relative_to(BASE).as_posix(),
                    "pixel_sha256": digest,
                })

        print(
            f"{class_name:28} "
            f"{len(parts['train']):7} "
            f"{len(parts['val']):7} "
            f"{len(parts['test']):7}"
        )

    with (OUTPUT / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "split", "class", "source",
                "destination", "pixel_sha256",
            ],
        )
        writer.writeheader()
        writer.writerows(manifest)

    print(f"\nDuplicate copies excluded: {removed}")
    print(f"Total unique images: {len(manifest)}")
    print("Saved to: data/split")
    print("Original images unchanged.")


if __name__ == "__main__":
    main()