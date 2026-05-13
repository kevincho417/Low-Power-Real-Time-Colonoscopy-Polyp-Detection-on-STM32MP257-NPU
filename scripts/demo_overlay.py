from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate annotated polyp detection outputs.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--source", required=True, type=Path, help="Image folder, image file, or video")
    parser.add_argument("--imgsz", default=416, type=int)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/demo")
    parser.add_argument("--name", default="polyp_overlay")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    results = model.predict(
        source=str(args.source),
        imgsz=args.imgsz,
        conf=args.conf,
        device=args.device,
        project=args.project,
        name=args.name,
        save=True,
        save_txt=True,
        save_conf=True,
        stream=False,
    )
    print(f"Saved {len(results)} annotated result(s) under {args.project}/{args.name}")


if __name__ == "__main__":
    main()
