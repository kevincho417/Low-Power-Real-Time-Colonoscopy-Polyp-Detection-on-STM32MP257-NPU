from __future__ import annotations

import argparse
import json
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    stem: str
    image_path: Path
    lines: list[str]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert Hyper-Kvasir segmented-images bounding-boxes.json to YOLO detection labels."
    )
    parser.add_argument("--name", default="hyper-kvasir-segmented")
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--bboxes", required=True, type=Path)
    parser.add_argument("--out", default=Path("data/processed"), type=Path)
    parser.add_argument("--splits", default="train=0.8,val=0.1,test=0.1")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--class-name", default="polyp")
    parser.add_argument(
        "--min-box-size",
        default=2.0,
        type=float,
        help="Skip boxes with width or height smaller than this many pixels.",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output_root = args.out / args.name
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)

    image_by_stem = {path.stem: path for path in _iter_images(args.images)}
    annotations = json.loads(args.bboxes.read_text(encoding="utf-8"))
    samples, stats = _build_samples(annotations, image_by_stem, args.min_box_size)
    if not samples:
        raise SystemExit("No valid Hyper-Kvasir samples were created.")

    random.Random(args.seed).shuffle(samples)
    assigned = _assign_splits(samples, _parse_splits(args.splits))
    for split, split_samples in assigned.items():
        image_out = output_root / "images" / split
        label_out = output_root / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for sample in tqdm(split_samples, desc=f"{args.name}:{split}"):
            dst_image = image_out / f"{sample.stem}{sample.image_path.suffix.lower()}"
            dst_label = label_out / f"{sample.stem}.txt"
            shutil.copy2(sample.image_path, dst_image)
            dst_label.write_text("\n".join(sample.lines), encoding="utf-8")

    yaml_path = output_root / f"{args.name}.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(output_root.resolve()).replace("\\", "/"),
                "train": "images/train",
                "val": "images/val",
                "test": "images/test",
                "names": {0: args.class_name},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    split_stats = {
        split: {
            "images": len(split_samples),
            "boxes": sum(len(sample.lines) for sample in split_samples),
        }
        for split, split_samples in assigned.items()
    }
    print(f"YOLO dataset written to: {output_root}")
    print(f"Dataset yaml: {yaml_path}")
    print(f"Conversion stats: {stats}")
    print(f"Split stats: {split_stats}")


def _build_samples(
    annotations: dict,
    image_by_stem: dict[str, Path],
    min_box_size: float,
) -> tuple[list[Sample], dict[str, int]]:
    samples: list[Sample] = []
    stats = {
        "json_entries": len(annotations),
        "missing_images": 0,
        "images": 0,
        "boxes": 0,
        "skipped_boxes": 0,
    }

    for stem, item in sorted(annotations.items()):
        image_path = image_by_stem.get(stem)
        if image_path is None:
            stats["missing_images"] += 1
            continue

        image_width, image_height = _image_size(image_path)
        width = int(item.get("width") or image_width)
        height = int(item.get("height") or image_height)
        if width != image_width or height != image_height:
            width, height = image_width, image_height

        lines: list[str] = []
        for box in item.get("bbox", []):
            line = _box_to_yolo_line(box, width, height, min_box_size)
            if line is None:
                stats["skipped_boxes"] += 1
                continue
            lines.append(line)

        if not lines:
            continue
        samples.append(Sample(stem=stem, image_path=image_path, lines=lines))
        stats["images"] += 1
        stats["boxes"] += len(lines)

    return samples, stats


def _box_to_yolo_line(
    box: dict,
    image_width: int,
    image_height: int,
    min_box_size: float,
) -> str | None:
    if box.get("label") != "polyp":
        return None

    xmin = max(0.0, min(float(box["xmin"]), float(image_width)))
    ymin = max(0.0, min(float(box["ymin"]), float(image_height)))
    xmax = max(0.0, min(float(box["xmax"]), float(image_width)))
    ymax = max(0.0, min(float(box["ymax"]), float(image_height)))
    box_width = xmax - xmin
    box_height = ymax - ymin
    if box_width < min_box_size or box_height < min_box_size:
        return None

    x_center = (xmin + xmax) / 2.0 / image_width
    y_center = (ymin + ymax) / 2.0 / image_height
    return f"0 {x_center:.6f} {y_center:.6f} {box_width / image_width:.6f} {box_height / image_height:.6f}"


def _iter_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


def _image_size(path: Path) -> tuple[int, int]:
    with Image.open(path) as image:
        return image.size


def _parse_splits(spec: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in spec.split(","):
        key, value = item.split("=", 1)
        result[key.strip()] = float(value)
    total = sum(result.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")
    return result


def _assign_splits(samples: list[Sample], split_spec: dict[str, float]) -> dict[str, list[Sample]]:
    assigned: dict[str, list[Sample]] = {}
    cursor = 0
    names = list(split_spec)
    for idx, name in enumerate(names):
        if idx == len(names) - 1:
            assigned[name] = samples[cursor:]
            break
        count = int(round(len(samples) * split_spec[name]))
        assigned[name] = samples[cursor : cursor + count]
        cursor += count
    return assigned


if __name__ == "__main__":
    main()
