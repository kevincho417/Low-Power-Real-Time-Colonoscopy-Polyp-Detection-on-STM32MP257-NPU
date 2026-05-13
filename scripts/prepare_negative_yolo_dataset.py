from __future__ import annotations

import argparse
import os
import random
import shutil
from dataclasses import dataclass
from pathlib import Path

import yaml
from tqdm import tqdm


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


@dataclass(frozen=True)
class Sample:
    source: Path
    stem: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a YOLO dataset of negative images with empty label files."
    )
    parser.add_argument("--name", required=True)
    parser.add_argument("--images-dir", action="append", required=True, type=Path)
    parser.add_argument("--out", default=Path("data/processed"), type=Path)
    parser.add_argument("--splits", default="train=0.8,val=0.1,test=0.1")
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--class-name", default="polyp")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    samples = _collect_samples(args.images_dir)
    if not samples:
        raise SystemExit("No images found for negative dataset.")

    output_root = args.out / args.name
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)

    random.Random(args.seed).shuffle(samples)
    assigned = _assign_splits(samples, _parse_splits(args.splits))
    for split, split_samples in assigned.items():
        image_out = output_root / "images" / split
        label_out = output_root / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for sample in tqdm(split_samples, desc=f"{args.name}:{split}"):
            dst_image = image_out / f"{sample.stem}{sample.source.suffix.lower()}"
            dst_label = label_out / f"{sample.stem}.txt"
            shutil.copy2(sample.source, dst_image)
            dst_label.write_text("", encoding="utf-8")

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

    split_stats = {split: {"images": len(split_samples), "empty_labels": len(split_samples)} for split, split_samples in assigned.items()}
    print(f"YOLO negative dataset written to: {output_root}")
    print(f"Dataset yaml: {yaml_path}")
    print(f"Total images: {len(samples)}")
    print(f"Split stats: {split_stats}")


def _collect_samples(image_dirs: list[Path]) -> list[Sample]:
    common_root = Path(os.path.commonpath([str(path.resolve()) for path in image_dirs]))
    samples: list[Sample] = []
    used_stems: set[str] = set()
    for image_dir in image_dirs:
        for image_path in _iter_images(image_dir):
            rel = image_path.resolve().relative_to(common_root)
            stem = _safe_stem(rel.with_suffix(""))
            if stem in used_stems:
                stem = _safe_stem(Path(f"{len(used_stems)}_{rel.with_suffix('')}"))
            used_stems.add(stem)
            samples.append(Sample(source=image_path, stem=stem))
    return samples


def _safe_stem(path: Path) -> str:
    return "_".join(part.replace(" ", "_").replace("-", "_") for part in path.parts)


def _iter_images(directory: Path) -> list[Path]:
    return sorted(
        path for path in directory.rglob("*") if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
    )


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
