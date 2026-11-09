import argparse
import csv
import random
import shutil
from hashlib import sha256
from pathlib import Path

from PIL import Image


BASE = Path(__file__).resolve().parent
OLD = BASE / "data" / "leaf_check_split"
OUTPUT = BASE / "data" / "leaf_check_v2"
FAILURE_REPORT = BASE / "artifacts" / "other_leaves_results.csv"
EXTENSIONS = {".jpg", ".jpeg", ".png"}


def pixel_hash(path):
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return sha256(
            str(rgb.size).encode() + rgb.tobytes()
        ).hexdigest()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()

    if OUTPUT.exists():
        raise SystemExit(
            "data/leaf_check_v2 already exists. Stopping."
        )

    if not args.source.is_dir():
        raise SystemExit("PlantVillage color folder not found.")

    if not FAILURE_REPORT.is_file():
        raise SystemExit("other_leaves_results.csv not found.")

    rng = random.Random(73)
    blocked = set()
    seen = set()
    records = []

    # Exclude the previously inspected images and exact copies.
    with FAILURE_REPORT.open(
        encoding="utf-8-sig", newline=""
    ) as file:
        for row in csv.DictReader(file):
            path = (
                args.source
                / row["source_class"]
                / row["filename"]
            )

            if not path.is_file():
                raise SystemExit(f"Missing inspected image: {path}")

            blocked.add(pixel_hash(path))

    def add_record(path, split, label, source_group):
        digest = pixel_hash(path)

        if digest in blocked or digest in seen:
            return False

        seen.add(digest)
        records.append({
            "split": split,
            "label": label,
            "source_group": source_group,
            "source": str(path.resolve()),
            "pixel_sha256": digest,
        })
        return True

    # Preserve the previous checker split assignments.
    for split in ["train", "val", "test"]:
        for label in ["tomato_leaf", "not_tomato_leaf"]:
            folder = OLD / split / label

            if not folder.is_dir():
                raise SystemExit(f"Missing folder: {folder}")

            for path in sorted(folder.iterdir()):
                if path.suffix.lower() in EXTENSIONS:
                    if not add_record(
                        path, split, label, "previous_checker"
                    ):
                        raise SystemExit(
                            f"Unexpected duplicate in old dataset: {path}"
                        )

    folders = sorted(
        p for p in args.source.iterdir()
        if p.is_dir()
        and "___" in p.name
        and not p.name.lower().startswith("tomato")
    )

    if not folders:
        raise SystemExit("No other plant classes found.")

    # Add new non-tomato leaves to each split.
    for folder in folders:
        files = sorted(
            p for p in folder.iterdir()
            if p.is_file() and p.suffix.lower() in EXTENSIONS
        )
        rng.shuffle(files)
        cursor = 0

        for split, required in [
            ("train", 20),
            ("val", 5),
            ("test", 5),
        ]:
            added = 0

            while added < required and cursor < len(files):
                path = files[cursor]
                cursor += 1

                try:
                    success = add_record(
                        path,
                        split,
                        "not_tomato_leaf",
                        folder.name,
                    )
                except (
                    OSError,
                    ValueError,
                    Image.DecompressionBombError,
                ):
                    continue

                if success:
                    added += 1

            if added != required:
                raise SystemExit(
                    f"Not enough unique images: {folder.name}, {split}"
                )

        print(f"Prepared: {folder.name}", flush=True)

    # Copy only after all source selections succeed.
    for record in records:
        source = Path(record["source"])
        target = (
            OUTPUT
            / record["split"]
            / record["label"]
            / f"{record['pixel_sha256']}{source.suffix.lower()}"
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)

    with (OUTPUT / "manifest.csv").open(
        "w", newline="", encoding="utf-8"
    ) as file:
        writer = csv.DictWriter(
            file, fieldnames=list(records[0])
        )
        writer.writeheader()
        writer.writerows(records)

    print("\nSplit | Class | Images")

    for split in ["train", "val", "test"]:
        for label in ["tomato_leaf", "not_tomato_leaf"]:
            count = sum(
                row["split"] == split and row["label"] == label
                for row in records
            )
            print(f"{split} | {label} | {count}")

    print("\nSaved: data/leaf_check_v2")
    print("Previously inspected images excluded.")
    print("Original data and models unchanged.")


if __name__ == "__main__":
    main()