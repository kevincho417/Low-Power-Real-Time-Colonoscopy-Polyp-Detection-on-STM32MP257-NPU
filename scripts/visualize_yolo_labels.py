from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

from PIL import Image, ImageDraw

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyp_edge.masks import iter_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Visualize YOLO labels for manual QA.")
    parser.add_argument("--dataset-root", required=True, type=Path, help="Processed YOLO dataset root")
    parser.add_argument("--split", default="train")
    parser.add_argument("--out", default=Path("runs/label_check"), type=Path)
    parser.add_argument("--max-images", default=24, type=int)
    parser.add_argument("--seed", default=42, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    image_dir = args.dataset_root / "images" / args.split
    label_dir = args.dataset_root / "labels" / args.split
    args.out.mkdir(parents=True, exist_ok=True)

    images = list(iter_images(image_dir))
    if not images:
        raise SystemExit(f"No images found under {image_dir}")
    random.Random(args.seed).shuffle(images)

    for image_path in images[: args.max_images]:
        image = Image.open(image_path).convert("RGB")
        draw = ImageDraw.Draw(image)
        label_path = label_dir / f"{image_path.stem}.txt"
        if label_path.exists():
            for line in label_path.read_text(encoding="utf-8").splitlines():
                if line.strip():
                    draw_box(draw, image.size, line)
        image.save(args.out / image_path.name)

    print(f"Saved label visualizations to {args.out}")


def draw_box(draw: ImageDraw.ImageDraw, size: tuple[int, int], line: str) -> None:
    width, height = size
    class_id, x_center, y_center, box_w, box_h = line.split()[:5]
    x_center = float(x_center) * width
    y_center = float(y_center) * height
    box_w = float(box_w) * width
    box_h = float(box_h) * height
    x1 = x_center - box_w / 2
    y1 = y_center - box_h / 2
    x2 = x_center + box_w / 2
    y2 = y_center + box_h / 2
    draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=3)
    label_top = y1 - 16
    label_bottom = y1
    if label_top < 0:
        label_top = y1
        label_bottom = y1 + 16
    draw.rectangle([x1, label_top, x1 + 70, label_bottom], fill=(255, 0, 0))
    draw.text((x1 + 3, label_top + 1), f"class {class_id}", fill=(255, 255, 255))


if __name__ == "__main__":
    main()
