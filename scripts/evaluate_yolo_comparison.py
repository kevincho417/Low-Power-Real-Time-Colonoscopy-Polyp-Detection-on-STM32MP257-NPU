from __future__ import annotations

import argparse
import csv
import json
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import onnxruntime as ort
from PIL import Image
from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from polyp_edge.masks import iter_images


CANDIDATES = (
    ("YOLO26n", "cmp_yolo26n_416"),
    ("YOLO11n", "cmp_yolo11n_416"),
    ("YOLOv8n", "cmp_yolov8n_416"),
    ("YOLOv8s", "cmp_yolov8s_416"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate trained YOLO candidates for the polyp project.")
    parser.add_argument("--train-root", default=Path("runs/detect/runs/compare_train"), type=Path)
    parser.add_argument(
        "--candidates",
        default=None,
        help=(
            "Comma-separated display=run_name list. "
            "Example: YOLO26n-Aug=aug_yolo26n_416,YOLOv8n-Aug=aug_yolov8n_416"
        ),
    )
    parser.add_argument("--kvasir-data", default=Path("data/processed/kvasir-seg/kvasir-seg.yaml"), type=Path)
    parser.add_argument("--cvc-data", default=Path("data/processed/cvc-clinicdb/cvc-clinicdb.yaml"), type=Path)
    parser.add_argument("--bench-images", default=Path("data/processed/kvasir-seg/images/val"), type=Path)
    parser.add_argument("--output-dir", default=Path("runs/compare_eval"), type=Path)
    parser.add_argument("--imgsz", default=416, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", default=0.001, type=float)
    parser.add_argument("--iou", default=0.6, type=float)
    parser.add_argument("--opset", default=13, type=int)
    parser.add_argument("--runs", default=200, type=int)
    parser.add_argument("--warmup", default=30, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for display_name, run_name in parse_candidates(args.candidates):
        weight = args.train_root / run_name / "weights" / "best.pt"
        if not weight.exists():
            raise FileNotFoundError(f"Missing trained weight: {weight}")

        print(f"Evaluating {display_name}: {weight}")
        model = YOLO(str(weight))
        params = sum(parameter.numel() for parameter in model.model.parameters())
        gflops = safe_gflops(model, args.imgsz)

        kvasir = run_validation(
            model=model,
            data=args.kvasir_data,
            split="val",
            output_dir=args.output_dir,
            name=f"{run_name}_kvasir_val",
            args=args,
        )
        cvc = run_validation(
            model=model,
            data=args.cvc_data,
            split="test",
            output_dir=args.output_dir,
            name=f"{run_name}_cvc_test",
            args=args,
        )
        onnx_path = export_onnx(model, args.imgsz, args.opset)
        bench = benchmark_onnx(
            model_path=onnx_path,
            images_dir=args.bench_images,
            imgsz=args.imgsz,
            runs=args.runs,
            warmup=args.warmup,
        )

        row = {
            "model": display_name,
            "run_name": run_name,
            "weight_path": str(weight),
            "onnx_path": str(onnx_path),
            "params": params,
            "params_m": params / 1_000_000,
            "gflops": gflops,
            "pt_size_mb": weight.stat().st_size / (1024 * 1024),
            "onnx_size_mb": onnx_path.stat().st_size / (1024 * 1024),
            **prefix("kvasir", kvasir),
            **prefix("cvc", cvc),
            **prefix("onnx_cpu", bench),
        }
        rows.append(row)
        write_outputs(rows, args.output_dir)
        print(
            f"{display_name}: Kvasir mAP50-95={row['kvasir_map50_95']:.3f}, "
            f"CVC mAP50-95={row['cvc_map50_95']:.3f}, "
            f"params={row['params_m']:.2f}M, CPU mean={row['onnx_cpu_mean_ms']:.2f} ms"
        )

    write_outputs(rows, args.output_dir)
    print(f"Wrote comparison metrics to {args.output_dir}")


def safe_gflops(model: YOLO, imgsz: int) -> float:
    try:
        return float(get_flops(model.model, imgsz=imgsz))
    except Exception:
        return float("nan")


def parse_candidates(raw: str | None) -> tuple[tuple[str, str], ...]:
    if raw is None:
        return CANDIDATES
    candidates: list[tuple[str, str]] = []
    for item in raw.split(","):
        item = item.strip()
        if not item:
            continue
        if "=" not in item:
            raise ValueError(f"Candidate must use display=run_name format: {item}")
        display_name, run_name = item.split("=", 1)
        candidates.append((display_name.strip(), run_name.strip()))
    if not candidates:
        raise ValueError("No candidates were provided.")
    return tuple(candidates)


def run_validation(
    model: YOLO,
    data: Path,
    split: str,
    output_dir: Path,
    name: str,
    args: argparse.Namespace,
) -> dict[str, float]:
    metrics = model.val(
        data=str(data),
        split=split,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        device=args.device,
        project=str((output_dir / "val").resolve()),
        name=name,
        plots=True,
        save_json=True,
        verbose=False,
    )
    results = metrics.results_dict
    speed = metrics.speed
    return {
        "precision": float(results["metrics/precision(B)"]),
        "recall": float(results["metrics/recall(B)"]),
        "map50": float(results["metrics/mAP50(B)"]),
        "map50_95": float(results["metrics/mAP50-95(B)"]),
        "fitness": float(results["fitness"]),
        "speed_preprocess_ms": float(speed["preprocess"]),
        "speed_inference_ms": float(speed["inference"]),
        "speed_postprocess_ms": float(speed["postprocess"]),
    }


def export_onnx(model: YOLO, imgsz: int, opset: int) -> Path:
    exported = model.export(format="onnx", imgsz=imgsz, opset=opset, simplify=False)
    return Path(exported)


def benchmark_onnx(
    model_path: Path,
    images_dir: Path,
    imgsz: int,
    runs: int,
    warmup: int,
) -> dict[str, float]:
    session = ort.InferenceSession(str(model_path), providers=["CPUExecutionProvider"])
    input_meta = session.get_inputs()[0]
    input_name = input_meta.name

    images = list(iter_images(images_dir))
    if not images:
        raise FileNotFoundError(f"No benchmark images found under {images_dir}")

    tensors = [
        preprocess(path, imgsz, input_shape=input_meta.shape, input_type=input_meta.type)
        for path in images[: min(len(images), runs)]
    ]

    for idx in range(warmup):
        session.run(None, {input_name: tensors[idx % len(tensors)]})

    latencies_ms: list[float] = []
    for idx in range(runs):
        start = time.perf_counter()
        session.run(None, {input_name: tensors[idx % len(tensors)]})
        latencies_ms.append((time.perf_counter() - start) * 1000.0)

    arr = np.asarray(latencies_ms, dtype=np.float64)
    mean_ms = float(arr.mean())
    return {
        "mean_ms": mean_ms,
        "p50_ms": float(np.percentile(arr, 50)),
        "p95_ms": float(np.percentile(arr, 95)),
        "fps": float(1000.0 / mean_ms),
    }


def preprocess(path: Path, imgsz: int, input_shape: list[Any], input_type: str) -> np.ndarray:
    image = Image.open(path).convert("RGB").resize((imgsz, imgsz), Image.BILINEAR)
    arr = np.asarray(image)
    tensor: np.ndarray
    if "uint8" in input_type:
        tensor = arr.astype(np.uint8)
    else:
        tensor = arr.astype(np.float32) / 255.0
    if expects_nchw(input_shape):
        tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
    else:
        tensor = tensor[None, ...]
    return tensor


def expects_nchw(shape: list[Any]) -> bool:
    if len(shape) != 4:
        return True
    if shape[1] == 3:
        return True
    if shape[-1] == 3:
        return False
    return True


def prefix(name: str, values: dict[str, Any]) -> dict[str, Any]:
    return {f"{name}_{key}": value for key, value in values.items()}


def write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    json_path = output_dir / "yolo_model_comparison.json"
    csv_path = output_dir / "yolo_model_comparison.csv"
    json_path.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not rows:
        return
    fieldnames = list(rows[0].keys())
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
