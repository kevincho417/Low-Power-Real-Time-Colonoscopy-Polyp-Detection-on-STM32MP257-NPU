from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
import onnxruntime as ort


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run exported YOLO ONNX video inference with CPU ONNX Runtime.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--imgsz", default=416, type=int)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--max-frames", default=0, type=int, help="0 means full video")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    session = ort.InferenceSession(str(args.model), providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name

    cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = cv2.VideoWriter(str(args.out), fourcc, fps, (width, height))

    frame_count = 0
    detection_frames = 0
    detections = 0
    infer_ms: list[float] = []
    wall_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_count >= args.max_frames:
            break

        tensor, meta = preprocess(frame, args.imgsz)
        start = time.perf_counter()
        output = session.run(None, {input_name: tensor})[0]
        infer_ms.append((time.perf_counter() - start) * 1000.0)

        boxes = decode(output, meta, args.conf)
        if boxes:
            detection_frames += 1
            detections += len(boxes)
        draw_boxes(frame, boxes)
        writer.write(frame)
        frame_count += 1

    cap.release()
    writer.release()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    infer_arr = np.asarray(infer_ms, dtype=np.float32)

    print(f"model={args.model}")
    print(f"video={args.video}")
    print(f"out={args.out}")
    print(f"source_frames={total} processed_frames={frame_count}")
    print(f"detection_frames={detection_frames} detections={detections}")
    if len(infer_arr):
        print(
            "inference_ms "
            f"mean={infer_arr.mean():.2f} "
            f"p50={np.percentile(infer_arr, 50):.2f} "
            f"p95={np.percentile(infer_arr, 95):.2f}"
        )
        print(f"inference_fps={1000.0 / infer_arr.mean():.2f}")
    print(f"wall_fps={frame_count / (wall_ms / 1000.0):.2f}")


def preprocess(frame: np.ndarray, imgsz: int) -> tuple[np.ndarray, dict[str, float]]:
    height, width = frame.shape[:2]
    scale = min(imgsz / height, imgsz / width)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (imgsz - new_width) / 2.0
    pad_y = (imgsz - new_height) / 2.0
    canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))
    canvas[top : top + new_height, left : left + new_width] = resized
    rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    tensor = np.transpose(rgb, (2, 0, 1))[None, ...]
    return tensor, {"scale": scale, "pad_x": float(left), "pad_y": float(top), "width": width, "height": height}


def decode(output: np.ndarray, meta: dict[str, float], conf: float) -> list[tuple[int, int, int, int, float]]:
    arr = np.squeeze(output)
    if arr.ndim != 2 or arr.shape[-1] < 6:
        return []
    boxes: list[tuple[int, int, int, int, float]] = []
    for row in arr:
        score = float(row[4])
        if score < conf:
            continue
        x1, y1, x2, y2 = [float(value) for value in row[:4]]
        x1 = (x1 - meta["pad_x"]) / meta["scale"]
        x2 = (x2 - meta["pad_x"]) / meta["scale"]
        y1 = (y1 - meta["pad_y"]) / meta["scale"]
        y2 = (y2 - meta["pad_y"]) / meta["scale"]
        x1 = int(np.clip(x1, 0, meta["width"] - 1))
        x2 = int(np.clip(x2, 0, meta["width"] - 1))
        y1 = int(np.clip(y1, 0, meta["height"] - 1))
        y2 = int(np.clip(y2, 0, meta["height"] - 1))
        if x2 <= x1 or y2 <= y1:
            continue
        boxes.append((x1, y1, x2, y2, score))
    return boxes


def draw_boxes(frame: np.ndarray, boxes: list[tuple[int, int, int, int, float]]) -> None:
    for x1, y1, x2, y2, score in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"polyp {score:.2f}"
        cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + 120, y1), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1 + 4, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


if __name__ == "__main__":
    main()
