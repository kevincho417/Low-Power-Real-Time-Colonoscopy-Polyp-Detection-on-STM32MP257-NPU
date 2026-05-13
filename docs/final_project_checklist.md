# Final Project Checklist

Use this checklist to keep the implementation aligned with the clinical proposal.

## Clinical Claim Discipline

- [ ] State that this is an assistive CADe prototype, not a diagnostic device.
- [ ] Connect model output to the clinical action: inspect and potentially remove a suspected polyp.
- [ ] Mention commercial CADe systems only as evidence that the workflow is clinically meaningful.
- [ ] Avoid claiming patient outcome benefit without clinical trial data.

## Data

- [ ] Kvasir-SEG downloaded and cited.
- [ ] CVC-ClinicDB downloaded and cited.
- [ ] Masks converted to YOLO labels.
- [ ] A sample of converted labels is visually inspected.
- [ ] External validation uses CVC-ClinicDB, not only a random split.

## Model

- [ ] YOLO11n or YOLOv8n baseline trained.
- [ ] Input size 320 or 416 evaluated.
- [ ] mAP50, mAP50-95, recall, and false positives reported.
- [ ] Failure cases documented.

## Edge Deployment

- [ ] ONNX/TFLite export completed.
- [ ] INT8 quantization attempted.
- [ ] STM32MP257 conversion attempted or blockers documented.
- [ ] CPU vs NPU benchmark included where available.
- [ ] End-to-end latency is separated from model-only latency.

