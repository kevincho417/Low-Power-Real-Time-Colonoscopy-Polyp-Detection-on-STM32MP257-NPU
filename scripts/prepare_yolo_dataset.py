from __future__ import annotations

import argparse
import random
import shutil
import sys
from pathlib import Path

import yaml
from PIL import Image
from tqdm import tqdm

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyp_edge.masks import (
    connected_component_boxes,
    load_binary_mask,
    pair_images_and_masks,
    whole_mask_box,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert colonoscopy polyp segmentation masks to a YOLO detection dataset."
    )
    parser.add_argument("--name", required=True, help="Dataset name, e.g. kvasir-seg")
    parser.add_argument("--images", required=True, type=Path, help="Directory containing images")
    parser.add_argument("--masks", required=True, type=Path, help="Directory containing binary masks")
    parser.add_argument(
        "--out",
        default=Path("data/processed"),
        type=Path,
        help="Output root. A subfolder named by --name will be created.",
    )
    parser.add_argument(
        "--splits",
        default="train=0.8,val=0.1,test=0.1",
        help="Split fractions. Use external=1.0 for an external-only dataset.",
    )
    parser.add_argument(
        "--split-file",
        action="append",
        default=[],
        metavar="SPLIT=PATH",
        help=(
            "Optional fixed split file containing one image stem per line. "
            "Can be provided multiple times, e.g. --split-file train=train.txt "
            "--split-file val=val.txt. Overrides --splits."
        ),
    )
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--class-name", default="polyp")
    parser.add_argument("--threshold", default=0, type=int, help="Mask pixel threshold")
    parser.add_argument("--min-area", default=25, type=int, help="Minimum mask area in pixels")
    parser.add_argument(
        "--box-mode",
        choices=["whole", "components"],
        default="whole",
        help="whole is recommended for Kvasir-SEG and CVC-ClinicDB.",
    )
    parser.add_argument(
        "--keep-negative",
        action="store_true",
        help="Keep images whose masks contain no valid polyp area.",
    )
    parser.add_argument(
        "--recursive-sequence-pairs",
        action="store_true",
        help=(
            "Pair every */images folder with its sibling */masks folder. "
            "Useful for PolypGen sequence folders and preserves the sequence name in output filenames."
        ),
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Remove the output dataset folder before writing new files.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    pairs = (
        _pair_recursive_sequence_images_and_masks(args.images)
        if args.recursive_sequence_pairs
        else _pair_flat_images_and_masks(args.images, args.masks)
    )
    if not pairs:
        raise SystemExit(
            f"No image/mask pairs found. Check stems under {args.images} and {args.masks}."
        )

    output_root = args.out / args.name
    if args.overwrite and output_root.exists():
        shutil.rmtree(output_root)
    if args.split_file:
        assigned = _assign_fixed_splits(pairs, args.split_file)
    else:
        split_spec = _parse_splits(args.splits)
        random.Random(args.seed).shuffle(pairs)
        assigned = _assign_splits(pairs, split_spec)

    stats = {"images": 0, "labeled_images": 0, "boxes": 0, "skipped_empty": 0}
    for split, split_pairs in assigned.items():
        image_out = output_root / "images" / split
        label_out = output_root / "labels" / split
        image_out.mkdir(parents=True, exist_ok=True)
        label_out.mkdir(parents=True, exist_ok=True)

        for pair in tqdm(split_pairs, desc=f"{args.name}:{split}"):
            image_path, mask_path, output_stem = pair
            mask = load_binary_mask(mask_path, threshold=args.threshold)
            if args.box_mode == "whole":
                boxes = whole_mask_box(mask, min_area=args.min_area)
            else:
                boxes = connected_component_boxes(mask, min_area=args.min_area)

            if not boxes and not args.keep_negative:
                stats["skipped_empty"] += 1
                continue

            image_ext = _normalized_image_ext(image_path)
            dst_image = image_out / f"{output_stem}{image_ext}"
            dst_label = label_out / f"{output_stem}.txt"
            _copy_or_convert_image(image_path, dst_image)
            dst_label.write_text("\n".join(box.to_yolo_line() for box in boxes), encoding="utf-8")

            stats["images"] += 1
            stats["boxes"] += len(boxes)
            if boxes:
                stats["labeled_images"] += 1

    yaml_path = output_root / f"{args.name}.yaml"
    yaml_path.write_text(
        yaml.safe_dump(
            {
                "path": str(output_root.resolve()).replace("\\", "/"),
                "train": "images/train" if (output_root / "images" / "train").exists() else "",
                "val": "images/val" if (output_root / "images" / "val").exists() else "",
                "test": "images/test" if (output_root / "images" / "test").exists() else "",
                "names": {0: args.class_name},
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    print(f"YOLO dataset written to: {output_root}")
    print(f"Dataset yaml: {yaml_path}")
    print(f"Stats: {stats}")


def _parse_splits(spec: str) -> dict[str, float]:
    result: dict[str, float] = {}
    for item in spec.split(","):
        key, value = item.split("=", 1)
        result[key.strip()] = float(value)
    total = sum(result.values())
    if abs(total - 1.0) > 1e-6:
        raise ValueError(f"Split fractions must sum to 1.0, got {total}")
    return result


Pair = tuple[Path, Path, str]


def _pair_flat_images_and_masks(images: Path, masks: Path) -> list[Pair]:
    return [(image_path, mask_path, image_path.stem) for image_path, mask_path in pair_images_and_masks(images, masks)]


def _pair_recursive_sequence_images_and_masks(root: Path) -> list[Pair]:
    pairs: list[Pair] = []
    for image_dir in sorted(path for path in root.rglob("images") if path.is_dir()):
        sequence_dir = image_dir.parent
        mask_dir = sequence_dir / "masks"
        if not mask_dir.exists():
            continue

        sequence_name = _safe_stem(root, sequence_dir)
        sequence_pairs = pair_images_and_masks(image_dir, mask_dir)
        for image_path, mask_path in sequence_pairs:
            pairs.append((image_path, mask_path, f"{sequence_name}_{image_path.stem}"))
    return pairs


def _safe_stem(root: Path, path: Path) -> str:
    rel = path.relative_to(root)
    return "_".join(part.replace(" ", "_") for part in rel.parts)


def _assign_splits(
    pairs: list[Pair],
    split_spec: dict[str, float],
) -> dict[str, list[Pair]]:
    assigned: dict[str, list[Pair]] = {}
    cursor = 0
    names = list(split_spec)
    for idx, name in enumerate(names):
        if idx == len(names) - 1:
            assigned[name] = pairs[cursor:]
            break
        count = int(round(len(pairs) * split_spec[name]))
        assigned[name] = pairs[cursor : cursor + count]
        cursor += count
    return assigned


def _assign_fixed_splits(
    pairs: list[Pair],
    split_files: list[str],
) -> dict[str, list[Pair]]:
    by_stem = {output_stem: (image_path, mask_path, output_stem) for image_path, mask_path, output_stem in pairs}
    by_image_stem = {image_path.stem: (image_path, mask_path, output_stem) for image_path, mask_path, output_stem in pairs}
    used: set[str] = set()
    assigned: dict[str, list[Pair]] = {}

    for spec in split_files:
        split, path_text = spec.split("=", 1)
        split = split.strip()
        split_path = Path(path_text.strip())
        if not split_path.exists():
            raise FileNotFoundError(f"Split file not found: {split_path}")

        selected: list[Pair] = []
        missing: list[str] = []
        for line in split_path.read_text(encoding="utf-8").splitlines():
            stem = Path(line.strip()).stem
            if not stem:
                continue
            pair = by_stem.get(stem, by_image_stem.get(stem))
            if pair is None:
                missing.append(stem)
                continue
            selected.append(pair)
            used.add(pair[2])

        assigned[split] = selected
        if missing:
            print(f"Warning: {len(missing)} stems from {split_path} were not found.")

    leftovers = [pair for pair in pairs if pair[2] not in used]
    if leftovers:
        assigned["test"] = leftovers
        print(f"Assigned {len(leftovers)} unlisted pair(s) to test split.")

    return assigned


def _normalized_image_ext(path: Path) -> str:
    if path.suffix.lower() in {".jpg", ".jpeg", ".png", ".bmp"}:
        return path.suffix.lower()
    return ".png"


def _copy_or_convert_image(src: Path, dst: Path) -> None:
    if src.suffix.lower() == dst.suffix.lower():
        shutil.copy2(src, dst)
        return
    Image.open(src).convert("RGB").save(dst)


if __name__ == "__main__":
    main()
