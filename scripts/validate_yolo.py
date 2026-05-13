from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate a YOLO polyp detector.")
    parser.add_argument("--model", required=True, type=Path, help="Path to best.pt or exported model")
    parser.add_argument("--data", required=True, type=Path, help="Ultralytics dataset yaml")
    parser.add_argument("--split", default="val", choices=["val", "test", "train"])
    parser.add_argument("--imgsz", default=416, type=int)
    parser.add_argument("--conf", default=0.001, type=float)
    parser.add_argument("--iou", default=0.6, type=float)
    parser.add_argument("--device", default=None)
    parser.add_argument("--project", default="runs/val")
    parser.add_argument("--name", default="polyp_eval")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    metrics = model.val(
        data=str(args.data),
        split=args.split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=args.project,
        name=args.name,
        plots=True,
        save_json=True,
    )
    print("Validation complete.")
    print(metrics)


if __name__ == "__main__":
    main()

