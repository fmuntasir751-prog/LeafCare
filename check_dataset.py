from collections import defaultdict
from hashlib import sha256
from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parent / "data" / "raw"
CLASSES = [
    "Tomato___Early_blight",
    "Tomato___Late_blight",
    "Tomato___Leaf_Mold",
    "Tomato___healthy",
]

hashes = defaultdict(list)
invalid = []
total = 0

for name in CLASSES:
    folder = ROOT / name

    if not folder.is_dir():
        raise SystemExit(f"Missing folder: {folder}")

    files = sorted(
        p for p in folder.iterdir()
        if p.is_file() and p.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )

    print(f"{name}: {len(files)} images")
    total += len(files)

    for path in files:
        try:
            with Image.open(path) as image:
                image.verify()

            with Image.open(path) as image:
                rgb = image.convert("RGB")
                digest = sha256(
                    str(rgb.size).encode() + rgb.tobytes()
                ).hexdigest()

            hashes[digest].append(path)
        except (OSError, ValueError, Image.DecompressionBombError) as error:
            invalid.append((path, str(error)))

duplicates = [paths for paths in hashes.values() if len(paths) > 1]
conflicts = [
    paths for paths in duplicates
    if len({p.parent.name for p in paths}) > 1
]

print(f"\nTotal images: {total}")
print(f"Invalid images: {len(invalid)}")
print(f"Duplicate groups: {len(duplicates)}")
print(f"Conflicting-label groups: {len(conflicts)}")

for path, error in invalid:
    print("INVALID:", path, error)

for paths in conflicts:
    print("CONFLICT:", [str(p.relative_to(ROOT)) for p in paths])