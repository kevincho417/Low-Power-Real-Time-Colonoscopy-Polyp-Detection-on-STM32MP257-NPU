from __future__ import annotations

import argparse
import random
from pathlib import Path

import cv2
import numpy as np
import yaml
from onnxruntime.quantization import (
    CalibrationDataReader,
    CalibrationMethod,
    QuantFormat,
    QuantType,
    quant_pre_process,
    quantize_static,
)


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


class ImageCalibrationReader(CalibrationDataReader):
    def __init__(self, input_name: str, image_paths: list[Path], imgsz: int) -> None:
        self.input_name = input_name
        self.image_paths = image_paths
        self.imgsz = imgsz
        self.index = 0

    def get_next(self) -> dict[str, np.ndarray] | None:
        if self.index >= len(self.image_paths):
            return None
        image_path = self.image_paths[self.index]
        self.index += 1
        return {self.input_name: preprocess(image_path, self.imgsz)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Create an INT8 QDQ ONNX model with image calibration.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--data", required=True, type=Path, help="YOLO dataset yaml used to collect calibration images")
    parser.add_argument("--output", required=True, type=Path)
    parser.add_argument("--imgsz", default=416, type=int)
    parser.add_argument("--samples", default=512, type=int)
    parser.add_argument("--seed", default=42, type=int)
    parser.add_argument("--input-name", default="images")
    parser.add_argument(
        "--op-types",
        default="Conv,MatMul",
        help="Comma-separated op types to quantize. Keep conservative for ST Edge AI compatibility.",
    )
    parser.add_argument(
        "--activation-type",
        choices=["uint8", "int8"],
        default="uint8",
        help="Activation quantization type.",
    )
    parser.add_argument(
        "--per-channel",
        action="store_true",
        help="Enable per-channel weight quantization. Per-tensor is often more compatible with STM32MPU tools.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output.parent.mkdir(parents=True, exist_ok=True)

    calibration_images = collect_calibration_images(args.data, args.samples, args.seed)
    if not calibration_images:
        raise SystemExit("No calibration images found.")

    preprocessed_model = args.output.with_name(args.output.stem + "_preprocessed.onnx")
    quant_pre_process(
        input_model_path=str(args.model),
        output_model_path=str(preprocessed_model),
        skip_optimization=False,
        skip_onnx_shape=False,
        skip_symbolic_shape=False,
    )

    reader = ImageCalibrationReader(args.input_name, calibration_images, args.imgsz)
    op_types = [item.strip() for item in args.op_types.split(",") if item.strip()]
    quantize_static(
        model_input=str(preprocessed_model),
        model_output=str(args.output),
        calibration_data_reader=reader,
        quant_format=QuantFormat.QDQ,
        activation_type=QuantType.QInt8 if args.activation_type == "int8" else QuantType.QUInt8,
        weight_type=QuantType.QInt8,
        calibrate_method=CalibrationMethod.MinMax,
        op_types_to_quantize=op_types,
        per_channel=args.per_channel,
        extra_options={
            "ActivationSymmetric": args.activation_type == "int8",
            "WeightSymmetric": True,
        },
    )

    print(f"Calibration images: {len(calibration_images)}")
    print(f"Preprocessed ONNX: {preprocessed_model}")
    print(f"INT8 QDQ ONNX: {args.output}")


def collect_calibration_images(data_yaml: Path, samples: int, seed: int) -> list[Path]:
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    root = Path(config["path"])
    image_dirs = config["train"]
    if isinstance(image_dirs, str):
        image_dirs = [image_dirs]

    images: list[Path] = []
    for image_dir in image_dirs:
        path = root / image_dir
        images.extend(
            image_path
            for image_path in path.rglob("*")
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_SUFFIXES
        )
    random.Random(seed).shuffle(images)
    return images[:samples]


def preprocess(image_path: Path, imgsz: int) -> np.ndarray:
    data = np.fromfile(str(image_path), dtype=np.uint8)
    image = cv2.imdecode(data, cv2.IMREAD_COLOR)
    if image is None:
        raise FileNotFoundError(f"Could not read image: {image_path}")
    height, width = image.shape[:2]
    scale = min(imgsz / height, imgsz / width)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    left = int(round((imgsz - new_width) / 2.0 - 0.1))
    top = int(round((imgsz - new_height) / 2.0 - 0.1))
    canvas[top : top + new_height, left : left + new_width] = resized

    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    return np.transpose(rgb, (2, 0, 1))[None, ...]


if __name__ == "__main__":
    main()
