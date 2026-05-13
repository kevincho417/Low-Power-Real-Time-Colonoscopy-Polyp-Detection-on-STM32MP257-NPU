from __future__ import annotations

import argparse
from pathlib import Path


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
SPLITS = ("train", "val", "test", "external")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Summarize and sanity-check processed YOLO datasets.")
    parser.add_argument("--root", default=Path("data/processed"), type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    rows: list[dict[str, object]] = []
    invalid: list[str] = []

    for dataset_root in sorted(path for path in args.root.iterdir() if path.is_dir()):
        for split in SPLITS:
            image_dir = dataset_root / "images" / split
            label_dir = dataset_root / "labels" / split
            if not image_dir.exists() and not label_dir.exists():
                continue

            images = [path for path in image_dir.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES]
            labels = sorted(label_dir.glob("*.txt")) if label_dir.exists() else []
            image_stems = {path.stem for path in images}
            label_stems = {path.stem for path in labels}
            boxes = 0
            empty_labels = 0

            for label_path in labels:
                lines = [line.strip() for line in label_path.read_text(encoding="utf-8").splitlines() if line.strip()]
                if not lines:
                    empty_labels += 1
                    continue
                for line_no, line in enumerate(lines, start=1):
                    parts = line.split()
                    if len(parts) != 5:
                        invalid.append(f"{label_path}:{line_no}: expected 5 fields")
                        continue
                    try:
                        class_id = int(parts[0])
                        values = [float(value) for value in parts[1:]]
                    except ValueError:
                        invalid.append(f"{label_path}:{line_no}: non-numeric label")
                        continue
                    if class_id != 0:
                        invalid.append(f"{label_path}:{line_no}: class_id={class_id}")
                    x, y, width, height = values
                    if not all(0.0 <= value <= 1.0 for value in values):
                        invalid.append(f"{label_path}:{line_no}: value outside [0,1]")
                    if width <= 0.0 or height <= 0.0:
                        invalid.append(f"{label_path}:{line_no}: non-positive box size")
                    boxes += 1

            rows.append(
                {
                    "dataset": dataset_root.name,
                    "split": split,
                    "images": len(images),
                    "labels": len(labels),
                    "boxes": boxes,
                    "empty": empty_labels,
                    "missing_labels": len(image_stems - label_stems),
                    "orphan_labels": len(label_stems - image_stems),
                }
            )

    print("dataset,split,images,labels,boxes,empty_labels,missing_labels,orphan_labels")
    for row in rows:
        print(
            f"{row['dataset']},{row['split']},{row['images']},{row['labels']},"
            f"{row['boxes']},{row['empty']},{row['missing_labels']},{row['orphan_labels']}"
        )

    if invalid:
        print("\nInvalid labels:")
        for item in invalid[:50]:
            print(item)
        raise SystemExit(f"Found {len(invalid)} invalid label issue(s).")
    print("\nAll checked labels are valid.")


if __name__ == "__main__":
    main()
