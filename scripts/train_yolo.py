from __future__ import annotations

import argparse
from pathlib import Path

import yaml
from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train a lightweight YOLO polyp detector.")
    parser.add_argument("--data", required=True, type=Path, help="Ultralytics dataset yaml")
    parser.add_argument("--config", default=Path("configs/training_defaults.yaml"), type=Path)
    parser.add_argument("--model", default=None, help="YOLO checkpoint, e.g. yolo11n.pt or yolo11s.pt")
    parser.add_argument("--imgsz", default=None, type=int)
    parser.add_argument("--epochs", default=None, type=int)
    parser.add_argument("--batch", default=None, type=int)
    parser.add_argument("--device", default=None, help="e.g. 0, cpu, mps")
    parser.add_argument("--project", default=None)
    parser.add_argument("--name", default=None)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    defaults = _load_defaults(args.config)
    opts = {**defaults}
    for key in ("model", "imgsz", "epochs", "batch", "project", "name"):
        value = getattr(args, key)
        if value is not None:
            opts[key] = value
    if args.device is not None:
        opts["device"] = args.device

    model = YOLO(opts.pop("model"))
    results = model.train(data=str(args.data), resume=args.resume, **opts)
    print("Training complete.")
    print(results)


def _load_defaults(path: Path) -> dict:
    if not path.exists():
        return {}
    return yaml.safe_load(path.read_text(encoding="utf-8")) or {}


if __name__ == "__main__":
    main()

