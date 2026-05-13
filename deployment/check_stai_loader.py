from __future__ import annotations

from stai_mpu import stai_mpu_network


PATHS = [
    "/usr/local/x-linux-ai/object-detection/models/coco_ssd_mobilenet/ssd_mobilenet_v2_fpnlite_10_256_int8_per_tensor.nb",
    "/home/root/polyp_yolo/models/yolo26n_polyp_416.onnx",
]


for path in PATHS:
    print(f"TRY {path}")
    try:
        network = stai_mpu_network(path)
        print("OK", network.get_backend_engine(), network.get_num_inputs(), network.get_num_outputs())
        print("INPUTS", network.get_input_infos())
        print("OUTPUTS", network.get_output_infos())
    except Exception as exc:
        print("ERR", type(exc).__name__, exc)
