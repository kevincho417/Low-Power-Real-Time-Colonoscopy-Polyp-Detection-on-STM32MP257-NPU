from __future__ import annotations

import argparse
import time
from pathlib import Path

import cv2
import numpy as np
from stai_mpu import stai_mpu_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run YOLO .nb video inference with STAI MPU NPU backend.")
    parser.add_argument("--model", required=True, type=Path)
    parser.add_argument("--video", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--imgsz", default=416, type=int)
    parser.add_argument("--conf", default=0.25, type=float)
    parser.add_argument("--iou", default=0.45, type=float)
    parser.add_argument("--max-frames", default=0, type=int, help="0 means full video")
    parser.add_argument("--debug-frames", default=0, type=int)
    parser.add_argument("--max-candidates", default=100, type=int)
    parser.add_argument("--gst-resize", action="store_true", help="Use GStreamer to decode and resize frames to imgsz before Python.")
    parser.add_argument("--gst-rgb", action="store_true", help="Use RGB frames from the GStreamer resize pipeline.")
    parser.add_argument("--no-output", action="store_true", help="Skip drawing and video writing for throughput measurement.")
    return parser.parse_args()


def tensor_dtype(info) -> np.dtype:
    dtype_map = {
        "float16": np.float16,
        "float32": np.float32,
        "int8": np.int8,
        "uint8": np.uint8,
    }
    return np.dtype(dtype_map.get(str(info.get_dtype()), np.float32))


def quant_params(info) -> tuple[float, int]:
    try:
        return float(info.get_scale()), int(info.get_zero_point())
    except Exception:
        return 0.0, 0


def main() -> None:
    args = parse_args()
    args.out.parent.mkdir(parents=True, exist_ok=True)

    network = stai_mpu_network(model_path=str(args.model), use_hw_acceleration=True)
    input_info = network.get_input_infos()[0]
    output_info = network.get_output_infos()[0]
    input_shape = tuple(input_info.get_shape())
    input_dtype = tensor_dtype(input_info)
    input_scale, input_zp = quant_params(input_info)
    output_scale, output_zp = quant_params(output_info)

    if args.gst_resize:
        gst_format = "RGB" if args.gst_rgb else "BGR"
        pipeline = (
            f"filesrc location={args.video} ! decodebin ! videoconvert ! videoscale ! "
            f"video/x-raw,format={gst_format},width={args.imgsz},height={args.imgsz} ! appsink sync=false"
        )
        cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
    else:
        cap = cv2.VideoCapture(str(args.video))
    if not cap.isOpened():
        raise SystemExit(f"Could not open video: {args.video}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = args.imgsz if args.gst_resize else int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = args.imgsz if args.gst_resize else int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fourcc = cv2.VideoWriter_fourcc(*"XVID")
    writer = None if args.no_output else cv2.VideoWriter(str(args.out), fourcc, fps, (width, height))

    frame_count = 0
    detection_frames = 0
    detections = 0
    infer_ms: list[float] = []
    preprocess_ms: list[float] = []
    decode_ms: list[float] = []
    draw_write_ms: list[float] = []
    wall_start = time.perf_counter()

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        if args.max_frames and frame_count >= args.max_frames:
            break

        step = time.perf_counter()
        if args.gst_resize:
            tensor, meta = preprocess_resized(frame, input_shape, input_dtype, input_scale, input_zp, args.gst_rgb)
        else:
            tensor, meta = preprocess(frame, args.imgsz, input_shape, input_dtype, input_scale, input_zp)
        preprocess_ms.append((time.perf_counter() - step) * 1000.0)
        start = time.perf_counter()
        network.set_input(0, tensor)
        network.run()
        output = network.get_output(0)
        infer_ms.append((time.perf_counter() - start) * 1000.0)
        if frame_count < args.debug_frames:
            print(output_summary(output, output_scale, output_zp))

        step = time.perf_counter()
        boxes = decode(output, meta, args.conf, args.iou, output_scale, output_zp, args.max_candidates)
        decode_ms.append((time.perf_counter() - step) * 1000.0)
        if boxes:
            detection_frames += 1
            detections += len(boxes)
        step = time.perf_counter()
        if writer is not None:
            if args.gst_rgb:
                frame = cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)
            draw_boxes(frame, boxes)
            writer.write(frame)
        draw_write_ms.append((time.perf_counter() - step) * 1000.0)
        frame_count += 1

    cap.release()
    if writer is not None:
        writer.release()
    wall_ms = (time.perf_counter() - wall_start) * 1000.0
    infer_arr = np.asarray(infer_ms, dtype=np.float32)

    print(f"model={args.model}")
    print(f"video={args.video}")
    print(f"out={args.out}")
    print(f"input_shape={input_shape} input_dtype={input_dtype} input_scale={input_scale} input_zp={input_zp}")
    print(f"output_shape={tuple(output_info.get_shape())} output_dtype={tensor_dtype(output_info)} output_scale={output_scale} output_zp={output_zp}")
    print(f"gst_resize={args.gst_resize} gst_rgb={args.gst_rgb} no_output={args.no_output}")
    print(f"source_frames={total} processed_frames={frame_count}")
    print(f"detection_frames={detection_frames} detections={detections}")
    if len(infer_arr):
        pre_arr = np.asarray(preprocess_ms, dtype=np.float32)
        dec_arr = np.asarray(decode_ms, dtype=np.float32)
        draw_arr = np.asarray(draw_write_ms, dtype=np.float32)
        print(f"preprocess_ms mean={pre_arr.mean():.2f} p95={np.percentile(pre_arr, 95):.2f}")
        print(
            "inference_ms "
            f"mean={infer_arr.mean():.2f} "
            f"p50={np.percentile(infer_arr, 50):.2f} "
            f"p95={np.percentile(infer_arr, 95):.2f}"
        )
        print(f"decode_nms_ms mean={dec_arr.mean():.2f} p95={np.percentile(dec_arr, 95):.2f}")
        print(f"draw_write_ms mean={draw_arr.mean():.2f} p95={np.percentile(draw_arr, 95):.2f}")
        print(f"inference_fps={1000.0 / infer_arr.mean():.2f}")
    print(f"wall_fps={frame_count / (wall_ms / 1000.0):.2f}")


def preprocess(
    frame: np.ndarray,
    imgsz: int,
    input_shape: tuple[int, ...],
    input_dtype: np.dtype,
    input_scale: float,
    input_zp: int,
) -> tuple[np.ndarray, dict[str, float]]:
    height, width = frame.shape[:2]
    scale = min(imgsz / height, imgsz / width)
    new_width = int(round(width * scale))
    new_height = int(round(height * scale))
    resized = cv2.resize(frame, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
    pad_x = (imgsz - new_width) / 2.0
    pad_y = (imgsz - new_height) / 2.0
    left = int(round(pad_x - 0.1))
    top = int(round(pad_y - 0.1))

    if (
        input_dtype == np.dtype(np.int8)
        and abs(input_scale - (1.0 / 255.0)) < 1e-6
        and input_zp == -128
        and len(input_shape) == 4
        and input_shape[-1] == 3
    ):
        tensor = np.full((imgsz, imgsz, 3), -14, dtype=np.int8)
        target = tensor[top : top + new_height, left : left + new_width]
        target[..., 0] = np.bitwise_xor(resized[..., 2], 128).view(np.int8)
        target[..., 1] = np.bitwise_xor(resized[..., 1], 128).view(np.int8)
        target[..., 2] = np.bitwise_xor(resized[..., 0], 128).view(np.int8)
        tensor = tensor[None, ...]
    elif np.issubdtype(input_dtype, np.integer):
        if input_scale <= 0:
            raise ValueError("Quantized input requires a positive scale.")
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        canvas[top : top + new_height, left : left + new_width] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = np.round((rgb.astype(np.float32) / 255.0) / input_scale + input_zp)
        limits = np.iinfo(input_dtype)
        tensor = np.clip(tensor, limits.min, limits.max).astype(input_dtype)
        if len(input_shape) == 4 and input_shape[1] == 3:
            tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        else:
            tensor = tensor[None, ...]
    else:
        canvas = np.full((imgsz, imgsz, 3), 114, dtype=np.uint8)
        canvas[top : top + new_height, left : left + new_width] = resized
        rgb = cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB)
        tensor = (rgb.astype(np.float32) / 255.0).astype(input_dtype)
        if len(input_shape) == 4 and input_shape[1] == 3:
            tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        else:
            tensor = tensor[None, ...]
    return tensor, {
        "scale": scale,
        "pad_x": float(left),
        "pad_y": float(top),
        "width": width,
        "height": height,
        "imgsz": imgsz,
    }


def preprocess_resized(
    frame: np.ndarray,
    input_shape: tuple[int, ...],
    input_dtype: np.dtype,
    input_scale: float,
    input_zp: int,
    frame_is_rgb: bool,
) -> tuple[np.ndarray, dict[str, float]]:
    height, width = frame.shape[:2]
    if (
        input_dtype == np.dtype(np.int8)
        and abs(input_scale - (1.0 / 255.0)) < 1e-6
        and input_zp == -128
        and len(input_shape) == 4
        and input_shape[-1] == 3
    ):
        if frame_is_rgb:
            tensor = np.bitwise_xor(frame, 128).view(np.int8)
        else:
            tensor = np.empty((height, width, 3), dtype=np.int8)
            tensor[..., 0] = np.bitwise_xor(frame[..., 2], 128).view(np.int8)
            tensor[..., 1] = np.bitwise_xor(frame[..., 1], 128).view(np.int8)
            tensor[..., 2] = np.bitwise_xor(frame[..., 0], 128).view(np.int8)
        tensor = tensor[None, ...]
    else:
        rgb = frame if frame_is_rgb else cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        if np.issubdtype(input_dtype, np.integer):
            if input_scale <= 0:
                raise ValueError("Quantized input requires a positive scale.")
            tensor = np.round((rgb.astype(np.float32) / 255.0) / input_scale + input_zp)
            limits = np.iinfo(input_dtype)
            tensor = np.clip(tensor, limits.min, limits.max).astype(input_dtype)
        else:
            tensor = (rgb.astype(np.float32) / 255.0).astype(input_dtype)
        if len(input_shape) == 4 and input_shape[1] == 3:
            tensor = np.transpose(tensor, (2, 0, 1))[None, ...]
        else:
            tensor = tensor[None, ...]
    return tensor, {"scale": 1.0, "pad_x": 0.0, "pad_y": 0.0, "width": width, "height": height, "imgsz": width}


def dequantize(output: np.ndarray, scale: float, zero_point: int) -> np.ndarray:
    if np.issubdtype(output.dtype, np.integer) and scale > 0:
        return (output.astype(np.float32) - float(zero_point)) * float(scale)
    return output.astype(np.float32)


def decode(
    output: np.ndarray,
    meta: dict[str, float],
    conf: float,
    iou: float,
    output_scale: float,
    output_zp: int,
    max_candidates: int,
) -> list[tuple[int, int, int, int, float]]:
    arr = np.squeeze(dequantize(output, output_scale, output_zp))
    if arr.ndim != 2:
        return []
    if arr.shape[0] == 5:
        arr = arr.T
    if arr.shape[-1] < 5:
        return []

    scores_arr = arr[:, 4]
    keep = np.flatnonzero(scores_arr >= conf)
    if keep.size == 0:
        return []
    if max_candidates > 0 and keep.size > max_candidates:
        top = np.argpartition(scores_arr[keep], -max_candidates)[-max_candidates:]
        keep = keep[top]

    selected = arr[keep, :5].astype(np.float32, copy=True)
    coords = selected[:, :4]
    if float(coords.max()) <= 2.0:
        coords *= float(meta["imgsz"])

    cx = coords[:, 0]
    cy = coords[:, 1]
    bw = coords[:, 2]
    bh = coords[:, 3]
    x1 = (cx - bw / 2.0 - meta["pad_x"]) / meta["scale"]
    y1 = (cy - bh / 2.0 - meta["pad_y"]) / meta["scale"]
    x2 = (cx + bw / 2.0 - meta["pad_x"]) / meta["scale"]
    y2 = (cy + bh / 2.0 - meta["pad_y"]) / meta["scale"]
    x1 = np.clip(x1, 0, meta["width"] - 1).astype(np.int32)
    x2 = np.clip(x2, 0, meta["width"] - 1).astype(np.int32)
    y1 = np.clip(y1, 0, meta["height"] - 1).astype(np.int32)
    y2 = np.clip(y2, 0, meta["height"] - 1).astype(np.int32)

    valid = np.flatnonzero((x2 > x1) & (y2 > y1))
    if valid.size == 0:
        return []

    raw_boxes = np.stack((x1[valid], y1[valid], x2[valid] - x1[valid], y2[valid] - y1[valid]), axis=1).tolist()
    scores = selected[valid, 4].tolist()

    boxes: list[tuple[int, int, int, int, float]] = []
    for idx in nms(raw_boxes, scores, iou):
        x, y, w, h = raw_boxes[int(idx)]
        boxes.append((x, y, x + w, y + h, scores[int(idx)]))
    return boxes


def output_summary(output: np.ndarray, scale: float, zero_point: int) -> str:
    arr = np.squeeze(dequantize(output, scale, zero_point))
    lines = [f"debug_output shape={arr.shape} min={float(arr.min()):.4f} max={float(arr.max()):.4f}"]
    if arr.ndim == 2:
        channels = arr if arr.shape[0] <= arr.shape[1] else arr.T
        for idx in range(min(5, channels.shape[0])):
            values = channels[idx]
            lines.append(
                f"channel_{idx} min={float(values.min()):.4f} "
                f"max={float(values.max()):.4f} mean={float(values.mean()):.4f}"
            )
    return "\n".join(lines)


def nms(boxes: list[list[int]], scores: list[float], iou_threshold: float) -> list[int]:
    if not boxes:
        return []
    xywh = np.asarray(boxes, dtype=np.float32)
    x1 = xywh[:, 0]
    y1 = xywh[:, 1]
    x2 = xywh[:, 0] + xywh[:, 2]
    y2 = xywh[:, 1] + xywh[:, 3]
    areas = np.maximum(0.0, x2 - x1) * np.maximum(0.0, y2 - y1)
    order = np.asarray(scores, dtype=np.float32).argsort()[::-1]

    keep: list[int] = []
    while order.size > 0:
        i = int(order[0])
        keep.append(i)
        if order.size == 1:
            break

        rest = order[1:]
        xx1 = np.maximum(x1[i], x1[rest])
        yy1 = np.maximum(y1[i], y1[rest])
        xx2 = np.minimum(x2[i], x2[rest])
        yy2 = np.minimum(y2[i], y2[rest])
        inter_w = np.maximum(0.0, xx2 - xx1)
        inter_h = np.maximum(0.0, yy2 - yy1)
        inter = inter_w * inter_h
        union = areas[i] + areas[rest] - inter
        ious = np.divide(inter, union, out=np.zeros_like(inter), where=union > 0)
        order = rest[ious <= iou_threshold]
    return keep


def draw_boxes(frame: np.ndarray, boxes: list[tuple[int, int, int, int, float]]) -> None:
    for x1, y1, x2, y2, score in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        label = f"polyp {score:.2f}"
        cv2.rectangle(frame, (x1, max(0, y1 - 22)), (x1 + 120, y1), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1 + 4, max(15, y1 - 6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)


if __name__ == "__main__":
    main()
