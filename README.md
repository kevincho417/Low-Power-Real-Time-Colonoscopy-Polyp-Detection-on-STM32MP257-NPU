# Low-Power Real-Time Colonoscopy Polyp Detection on STM32MP257 NPU

This repository contains a biomedical AI final-project prototype for real-time
colonoscopy polyp detection on a low-power STM32MP257 NPU platform. The project
combines multi-dataset YOLO training, cross-dataset validation, INT8
quantization, STM32 `.nb` conversion, and measured embedded inference.

This is a research and education prototype. It is not a diagnostic device and
is not intended for patient care.

## Project Summary

The goal is to test whether lightweight YOLO models can provide clinically
meaningful real-time polyp highlighting while remaining deployable on an edge
NPU. Public endoscopy datasets were converted to YOLO bounding-box format, four
YOLO models were compared, and selected full-integer INT8 models were deployed
and benchmarked on STM32MP257.

Final deployment choice:

- Accuracy upper bound: YOLOv8s
- Best nano-scale accuracy: YOLO11n
- Compact but weaker deployment stability: YOLO26n
- Selected deployment model: YOLOv8n full-integer INT8 `.nb`

YOLOv8n was selected because it achieved the fastest verified STM32MP257 NPU
inference among the deployed models while preserving competitive accuracy.

## Repository Contents

```text
configs/        Training configuration files
deployment/     STM32MP257 STAI-MPU inspection, benchmark, and video scripts
docs/           Project notes and result summaries
DOC images/     Academic report figures
models/int8/    Final small INT8 TFLite and STM32 .nb deployment artifacts
runs/           Compact evaluation CSV/JSON summaries and report figures
scripts/        Dataset conversion, training, evaluation, quantization, and report builders
src/            Small local utility package
tests/          Unit tests for conversion utilities
```

Large raw datasets, converted datasets, training runs, PyTorch weights, videos,
and local board credentials are intentionally excluded from git.

## Datasets

The project used public gastrointestinal endoscopy datasets:

- Kvasir-SEG
- CVC-ClinicDB
- CVC-ColonDB
- ETIS-LaribPolypDB
- PolypGen
- Hyper-Kvasir

Segmentation masks were converted into YOLO bounding boxes. Negative lower-GI
frames were represented using empty YOLO label files to improve background
rejection.

The final multi-dataset split used:

- 6,127 training images
- 775 validation images
- 657 internal test images
- CVC-ClinicDB, CVC-ColonDB, and ETIS as external test datasets

## Main Results

Cross-dataset model summary is available in:

```text
runs/multidata_compare_eval_4models/model_summary.csv
runs/multidata_compare_eval_4models/multidata_eval.csv
```

Key accuracy results:

| Model | Params | Internal mAP50-95 | External Avg mAP50-95 |
| --- | ---: | ---: | ---: |
| YOLOv8s | 11.14M | 0.774 | 0.502 |
| YOLO11n | 2.59M | 0.748 | 0.481 |
| YOLO26n | 2.50M | 0.768 | 0.478 |
| YOLOv8n | 3.01M | 0.756 | 0.465 |

STM32MP257 NPU deployment summary:

| Model | Runtime | NPU Latency | NPU FPS | Deployment Note |
| --- | --- | ---: | ---: | --- |
| YOLOv8n | full-integer INT8 `.nb` | 35.70 ms | 28.01 | Selected final model |
| YOLO11n | full-integer INT8 `.nb` | 43.20 ms | 23.15 | Valid but slower |
| YOLO26n | full-integer INT8 `.nb` | 44.08 ms | 22.69 | Lower video stability |

## Reproducible Workflow

Dataset conversion:

```powershell
python scripts\prepare_yolo_dataset.py --help
python scripts\prepare_hyper_kvasir_bbox.py --help
python scripts\prepare_negative_yolo_dataset.py --help
```

Training:

```powershell
python scripts\train_yolo.py --help
```

Evaluation:

```powershell
python scripts\evaluate_trained_models.py `
  --output-dir runs\multidata_compare_eval_4models `
  --imgsz 416 `
  --device 0
```

Report generation:

```powershell
python scripts\build_full_paper_report.py
python scripts\update_yolo_comparison_report.py
```

STM32MP257 NPU inspection and benchmarking:

```powershell
python deployment\inspect_stai_model.py --help
python deployment\bench_stai_model.py --help
python deployment\stai_yolo_video.py --help
```

## Reports

The complete paper-style report is:

```text
Polyp Detection STM32MP257 Full Paper Report.docx
```

The model-comparison report is:

```text
Digital AI Proposed Project - YOLO Comparison.docx
```

## Notes

The uploaded repository is intentionally compact. To reproduce training, download
the public datasets separately and place them into local dataset folders before
running the conversion scripts. The exact local dataset files are not included
because of size and redistribution constraints.
