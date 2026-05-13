# Current Experiment Summary

Date: 2026-05-12

## Data Prepared

| Dataset | Role | Images | Labels | Split |
| --- | --- | ---: | ---: | --- |
| Kvasir-SEG | Training / validation | 1000 | 1000 polyp boxes | train 880, val 120 |
| CVC-ClinicDB | External validation | 612 | 612 polyp boxes | test 612 |

Kvasir-SEG uses the downloaded `train.txt` and `val.txt` files for reproducible
splitting. CVC-ClinicDB is kept as an external test dataset to support the
clinical generalization argument.

## Baseline Model

| Item | Value |
| --- | --- |
| Architecture | YOLO11n detection |
| Input size | 416 |
| Epochs | 60 |
| Batch size | 32 |
| GPU | NVIDIA GeForce RTX 4070 SUPER |
| Parameters | 2.58M |
| Compute | ~6.3 GFLOPs |
| Best PyTorch model | `models/yolo11n_polyp_416_best.pt` |
| ONNX model | `models/yolo11n_polyp_416_best.onnx` |

## Validation Results

| Evaluation | Images | Precision | Recall | mAP50 | mAP50-95 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Kvasir-SEG validation | 120 | 0.746 | 0.809 | 0.846 | 0.595 |
| CVC-ClinicDB external test | 612 | 0.726 | 0.672 | 0.740 | 0.427 |

Interpretation: the model performs well on the Kvasir validation split and shows
the expected drop on CVC-ClinicDB. This is useful for the final project because it
demonstrates domain shift rather than hiding it behind a single random split.

## Runtime Results

| Runtime | Model | Device | Mean latency | p50 | p95 | Mean FPS |
| --- | --- | --- | ---: | ---: | ---: | ---: |
| ONNX Runtime | `models/yolo11n_polyp_416_best.onnx` | CPUExecutionProvider | 11.26 ms | 11.00 ms | 13.70 ms | 88.85 |
| Ultralytics demo overlay | `models/yolo11n_polyp_416_best.pt` | RTX 4070 SUPER | ~12.5 ms end-to-end | N/A | N/A | ~80 |

The ONNX benchmark is model-inference only and does not include YOLO NMS or
overlay. The Ultralytics demo includes preprocessing, inference, and
post-processing on the GPU-side demo pipeline.

## Generated Artifacts

| Artifact | Path |
| --- | --- |
| Training run | `runs/train/yolo11n_polyp_416` |
| Kvasir validation plots | `runs/val/kvasir_val_yolo11n_416` |
| CVC external validation plots | `runs/val/cvc_external_yolo11n_416` |
| CVC overlay demo | `runs/demo/cvc_yolo11n_polyp_overlay` |
| Label QA images | `runs/label_check/kvasir_train`, `runs/label_check/cvc_test` |
| STM32MP257 deployment notes | `deployment/stm32mp257/README.md` |

## Current Limitations

- STM32MP257 NPU execution has not been measured yet because `stedgeai` is not
  available in the current host PATH.
- INT8/NBG conversion remains the next hardware-specific step.
- Public datasets contain annotated frames rather than complete clinical video
  streams, so real-time clinical claims should remain prototype-level.

