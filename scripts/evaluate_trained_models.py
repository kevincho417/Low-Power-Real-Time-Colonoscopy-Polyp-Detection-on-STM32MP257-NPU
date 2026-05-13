from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

from ultralytics import YOLO
from ultralytics.utils.torch_utils import get_flops


DEFAULT_MODELS = {
    "YOLO11n-MultiData": Path("runs/detect/runs/multidata_train/multidata_yolo11n_416/weights/best.pt"),
    "YOLO26n-MultiData": Path("runs/detect/runs/multidata_train/multidata_yolo26n_416/weights/best.pt"),
    "YOLOv8n-MultiData": Path("runs/detect/runs/multidata_train/multidata_yolov8n_416/weights/best.pt"),
    "YOLOv8s-MultiData": Path("runs/detect/runs/multidata_train/multidata_yolov8s_416/weights/best.pt"),
}

DEFAULT_DATASETS = {
    "internal_holdout": (Path("data/processed/polyp-multidata/polyp-multidata-local.yaml"), "test"),
    "cvc_clinicdb": (Path("data/processed/cvc-clinicdb/cvc-clinicdb.yaml"), "test"),
    "cvc_colondb": (Path("data/processed/cvc-colondb/cvc-colondb.yaml"), "test"),
    "etis": (Path("data/processed/etis/etis.yaml"), "test"),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate multidata YOLO models on internal and external test sets.")
    parser.add_argument("--output-dir", default=Path("runs/multidata_eval"), type=Path)
    parser.add_argument("--imgsz", default=416, type=int)
    parser.add_argument("--device", default="0")
    parser.add_argument("--conf", default=0.001, type=float)
    parser.add_argument("--iou", default=0.6, type=float)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    rows: list[dict[str, Any]] = []
    for model_name, weight_path in DEFAULT_MODELS.items():
        if not weight_path.exists():
            raise FileNotFoundError(f"Missing model weight: {weight_path}")
        model = YOLO(str(weight_path))
        params = sum(parameter.numel() for parameter in model.model.parameters())
        gflops = _safe_gflops(model, args.imgsz)

        for dataset_name, (data_path, split) in DEFAULT_DATASETS.items():
            metrics = model.val(
                data=str(data_path),
                split=split,
                imgsz=args.imgsz,
                conf=args.conf,
                iou=args.iou,
                device=args.device,
                project=str(args.output_dir / "val"),
                name=f"{model_name}_{dataset_name}",
                plots=True,
                save_json=True,
                verbose=False,
            )
            result = metrics.results_dict
            speed = metrics.speed
            row = {
                "model": model_name,
                "weight_path": str(weight_path),
                "dataset": dataset_name,
                "split": split,
                "params": params,
                "params_m": params / 1_000_000,
                "gflops": gflops,
                "precision": float(result["metrics/precision(B)"]),
                "recall": float(result["metrics/recall(B)"]),
                "map50": float(result["metrics/mAP50(B)"]),
                "map50_95": float(result["metrics/mAP50-95(B)"]),
                "fitness": float(result["fitness"]),
                "preprocess_ms": float(speed["preprocess"]),
                "inference_ms": float(speed["inference"]),
                "postprocess_ms": float(speed["postprocess"]),
            }
            rows.append(row)
            print(
                f"{model_name} {dataset_name}: "
                f"P={row['precision']:.3f}, R={row['recall']:.3f}, "
                f"mAP50={row['map50']:.3f}, mAP50-95={row['map50_95']:.3f}"
            )
            _write_outputs(rows, args.output_dir)

    _write_outputs(rows, args.output_dir)
    print(f"Wrote metrics to {args.output_dir}")


def _safe_gflops(model: YOLO, imgsz: int) -> float:
    try:
        return float(get_flops(model.model, imgsz=imgsz))
    except Exception:
        return float("nan")


def _write_outputs(rows: list[dict[str, Any]], output_dir: Path) -> None:
    (output_dir / "multidata_eval.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    if not rows:
        return
    with (output_dir / "multidata_eval.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    main()
