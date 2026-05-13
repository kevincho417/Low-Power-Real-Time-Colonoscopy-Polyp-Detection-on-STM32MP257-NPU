from __future__ import annotations

import argparse
import time

import numpy as np
from stai_mpu import stai_mpu_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    parser.add_argument("--loops", default=50, type=int)
    parser.add_argument("--warmup", default=5, type=int)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    network = stai_mpu_network(model_path=args.model, use_hw_acceleration=True)
    input_info = network.get_input_infos()[0]
    shape = tuple(input_info.get_shape())
    stai_dtype = str(input_info.get_dtype())
    dtype_map = {
        "float16": np.float16,
        "float32": np.float32,
        "int8": np.int8,
        "uint8": np.uint8,
        "int16": np.int16,
        "uint16": np.uint16,
        "int32": np.int32,
    }
    dtype = dtype_map.get(stai_dtype, np.float32)
    tensor = np.zeros(shape, dtype=dtype)

    for _ in range(args.warmup):
        network.set_input(0, tensor)
        network.run()
        _ = network.get_output(0)

    set_input_ms = []
    run_ms = []
    get_output_ms = []
    latencies = []
    for _ in range(args.loops):
        start = time.perf_counter()
        step = start
        network.set_input(0, tensor)
        set_input_ms.append((time.perf_counter() - step) * 1000.0)
        step = time.perf_counter()
        network.run()
        run_ms.append((time.perf_counter() - step) * 1000.0)
        step = time.perf_counter()
        output = network.get_output(0)
        get_output_ms.append((time.perf_counter() - step) * 1000.0)
        latencies.append((time.perf_counter() - start) * 1000.0)

    arr = np.asarray(latencies, dtype=np.float32)
    print(f"model={args.model}")
    print(f"input_shape={shape} input_dtype={dtype} stai_dtype={stai_dtype}")
    print(f"output_shape={output.shape} output_dtype={output.dtype}")
    print(f"loops={args.loops} warmup={args.warmup}")
    print(f"latency_ms mean={arr.mean():.2f} p50={np.percentile(arr, 50):.2f} p95={np.percentile(arr, 95):.2f}")
    print(f"set_input_ms mean={np.mean(set_input_ms):.2f}")
    print(f"run_ms mean={np.mean(run_ms):.2f}")
    print(f"get_output_ms mean={np.mean(get_output_ms):.2f}")
    print(f"fps={1000.0 / arr.mean():.2f}")


if __name__ == "__main__":
    main()
