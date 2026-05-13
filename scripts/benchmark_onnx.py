from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import numpy as np
import onnxruntime as ort
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyp_edge.masks import iter_images


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Benchmark ONNX inference latency on image folders.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--images", required=True, type=Path)
    parser.add_argument("--imgsz", default=416, type=int)
    parser.add_argument("--runs", default=200, type=int)
    parser.add_argument("--warmup", default=20, type=int)
    parser.add_argument(
        "--providers",
        default="CPUExecutionProvider",
        help="Comma-separated ONNX Runtime providers.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    providers = [item.strip() for item in args.providers.split(",") if item.strip()]
    session = ort.InferenceSession(str(args.model), providers=providers)
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name
    input_shape = input_meta.shape
    input_type = input_meta.type

    images = list(iter_images(args.images))
    if not images:
        raise SystemExit(f"No images found under {args.images}")

    tensors = [
        preprocess(path, args.imgsz, input_shape=input_shape, input_type=input_type)
        for path in images[: min(len(images), args.runs)]
    ]
    if not tensors:
        raise SystemExit("No tensors prepared.")

    for idx in range(args.warmup):
        session.run(None, {input_name: tensors[idx % len(tensors)]})

    latencies_ms: list[float] = []
    for idx in range(args.runs):
        tensor = tensors[idx % len(tensors)]
        start = time.perf_counter()
        session.run(None, {input_name: tensor})
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    arr = np.asarray(latencies_ms)
    print(f"model={args.model}")
    print(f"provider={session.get_providers()}")
    print(f"input={input_name} shape={input_shape} type={input_type}")
    print(f"runs={args.runs} warmup={args.warmup}")
    print(f"latency_ms mean={arr.mean():.2f} p50={np.percentile(arr, 50):.2f} p95={np.percentile(arr, 95):.2f}")
    print(f"fps_mean={1000.0 / arr.mean():.2f}")


def preprocess(path: Path, imgsz: int, input_shape: list, input_type: str) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.asarray(image)
    nchw = _expects_nchw(input_shape)

    if "uint8" in input_type:
        tensor = arr.astype(np.uint8)
    else:
        tensor = arr.astype(np.float32) / 255.0

    if nchw:
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    else:
        tensor = tensor[None, ...]
    return tensor


def _expects_nchw(shape: list) -> bool:
    if len(shape) != 4:
        return True
    second = shape[1]
    last = shape[-1]
    if second == 3:
        return True
    if last == 3:
        return False
    return True


if __name__ == "__main__":
    main()
