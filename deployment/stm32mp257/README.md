# STM32MP257 NPU Deployment Notes

This directory documents the deployment path for the STM32MP257 NPU portion of
the gastrointestinal polyp detection final project.

The exact commands can vary by OpenSTLinux image, X-LINUX-AI version, and ST Edge
AI tool version. Treat this file as the controlled deployment checklist for the
final report.

## Deployment Principle

The embedded system should run only the neural-network graph on the NPU:

```text
frame -> resize/normalize -> NPU YOLO graph -> CPU NMS -> bounding-box overlay
```

Keep post-processing out of the NPU graph. YOLO NMS and visualization are easier
to maintain on CPU and reduce the chance of conversion failure.

## Model Export on Host PC

Export fixed-shape ONNX first:

```powershell
python scripts\export_model.py `
  --model runs\train\yolo11n_polyp_416\weights\best.pt `
  --format onnx `
  --imgsz 416 `
  --simplify `
  --opset 13
```

Export INT8 TFLite if needed by the ST conversion flow:

```powershell
python scripts\export_model.py `
  --model runs\train\yolo11n_polyp_416\weights\best.pt `
  --format tflite `
  --imgsz 416 `
  --int8 `
  --data data\processed\kvasir-seg\kvasir-seg.yaml
```

## ST Edge AI Conversion

Depending on installed ST tools, the conversion path may use `stedgeai` to
validate and generate an NPU-compatible network binary graph. Example:

```powershell
stedgeai validate `
  -m runs\train\yolo11n_polyp_416\weights\best.onnx `
  --target stm32mp25
```

```powershell
stedgeai generate `
  -m runs\train\yolo11n_polyp_416\weights\best.onnx `
  --target stm32mp25 `
  --st-neural-art default
```

Expected output is an NBG/NPU-ready artifact when all operators are supported.
If conversion fails, document the unsupported operator and apply one of these
fallbacks:

1. Export a smaller or older YOLO variant.
2. Use a fixed input size and static graph.
3. Keep all NMS/post-processing outside the exported graph.
4. Use TFLite INT8 as an intermediate format if supported better by the toolchain.

## Board-Side Benchmark Checklist

Record these in the final report:

| Item | Required measurement |
| --- | --- |
| Model-only latency | Mean, p50, p95 in ms |
| End-to-end latency | Resize + inference + NMS + overlay |
| FPS | Sustained FPS on test frames |
| CPU vs NPU | Same model/input where possible |
| Model size | FP32 and INT8 files |
| Accuracy drop | FP32 mAP/recall vs INT8 mAP/recall |

## Success Criteria

Minimum:

- YOLO11n or YOLOv8n runs at 416 or 320 input size.
- NPU or board-side runtime produces bounding boxes from colonoscopy frames.
- End-to-end latency is reported honestly, including post-processing.

Target:

- At least 15 FPS end-to-end.
- Less than 5 percentage point absolute drop in mAP50 after quantization.
- External validation on CVC-ClinicDB is included.

