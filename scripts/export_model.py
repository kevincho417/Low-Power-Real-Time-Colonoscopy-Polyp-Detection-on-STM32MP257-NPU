from __future__ import annotations

import argparse
from pathlib import Path

from ultralytics import YOLO


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export YOLO model for edge deployment.")
    parser.add_argument("--model", required=True, type=Path, help="Path to trained best.pt")
    parser.add_argument("--format", default="onnx", choices=["onnx", "tflite", "openvino", "engine"])
    parser.add_argument("--imgsz", default=416, type=int)
    parser.add_argument("--data", type=Path, help="Dataset yaml for INT8 calibration")
    parser.add_argument("--int8", action="store_true", help="Enable INT8 export when supported")
    parser.add_argument("--dynamic", action="store_true", help="Dynamic input shape")
    parser.add_argument("--simplify", action="store_true", help="Simplify ONNX graph when supported")
    parser.add_argument("--opset", default=13, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    model = YOLO(str(args.model))
    kwargs = {
        "format": args.format,
        "imgsz": args.imgsz,
        "int8": args.int8,
        "dynamic": args.dynamic,
    }
    if args.format == "onnx":
        kwargs["opset"] = args.opset
        kwargs["simplify"] = args.simplify
    if args.data is not None:
        kwargs["data"] = str(args.data)
    exported = model.export(**kwargs)
    print(f"Exported model: {exported}")


if __name__ == "__main__":
    main()

