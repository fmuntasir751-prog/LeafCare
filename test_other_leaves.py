import argparse
import csv
import os
import random
from pathlib import Path

os.environ.setdefault("DJANGO_SETTINGS_MODULE", "config.settings")

import django
django.setup()

import torch
from PIL import Image

from detection.predictor import load_checker


BASE = Path(__file__).resolve().parent


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path)
    args = parser.parse_args()

    if not args.source.is_dir():
        raise SystemExit(f"Folder not found: {args.source}")

    folders = sorted(
        p for p in args.source.iterdir()
        if p.is_dir()
        and "___" in p.name
        and not p.name.lower().startswith("tomato")
    )

    if not folders:
        raise SystemExit("No non-tomato class folders found.")

    torch.set_num_threads(4)
    model, classes, preprocess = load_checker()

    rng = random.Random(42)
    rows = []

    for folder in folders:
        files = sorted(
            p for p in folder.iterdir()
            if p.is_file()
            and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
        )
        rng.shuffle(files)

        accepted = 0
        checked = 0

        for path in files[:5]:
            try:
                with Image.open(path) as image:
                    tensor = preprocess(
                        image.convert("RGB")
                    ).unsqueeze(0)

                with torch.inference_mode():
                    index = int(model(tensor)[0].argmax().item())

                is_accepted = classes[index] == "tomato_leaf"

                checked += 1
                accepted += int(is_accepted)

                rows.append({
                    "source_class": folder.name,
                    "filename": path.name,
                    "checker_result": classes[index],
                    "incorrectly_accepted": int(is_accepted),
                    "error": "",
                })

            except (OSError, ValueError, Image.DecompressionBombError) as error:
                rows.append({
                    "source_class": folder.name,
                    "filename": path.name,
                    "checker_result": "",
                    "incorrectly_accepted": "",
                    "error": str(error),
                })

        print(
            f"{folder.name}: "
            f"incorrectly accepted {accepted}/{checked}",
            flush=True,
        )

    valid = [row for row in rows if not row["error"]]
    wrong = sum(row["incorrectly_accepted"] for row in valid)

    output = BASE / "artifacts" / "other_leaves_results.csv"
    output.parent.mkdir(exist_ok=True)

    with output.open("w", newline="", encoding="utf-8-sig") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "source_class",
                "filename",
                "checker_result",
                "incorrectly_accepted",
                "error",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print(f"\nImages checked: {len(valid)}")
    print(f"Incorrectly accepted as tomato: {wrong}")

    if valid:
        print(f"False acceptance rate: {wrong / len(valid):.2%}")

    print(f"Unreadable images: {len(rows) - len(valid)}")
    print("Saved: artifacts/other_leaves_results.csv")


if __name__ == "__main__":
    main()