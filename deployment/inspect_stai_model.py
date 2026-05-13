from __future__ import annotations

import argparse

from stai_mpu import stai_mpu_network


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("model")
    return parser.parse_args()


def tensor_summary(tensor) -> str:
    attrs = {}
    for name in (
        "get_name",
        "get_index",
        "get_rank",
        "get_shape",
        "get_dtype",
        "get_qtype",
        "get_scale",
        "get_zero_point",
        "get_fixed_point_pos",
    ):
        try:
            value = getattr(tensor, name)
            attrs[name[4:] if name.startswith("get_") else name] = value()
        except Exception:
            pass
    if attrs:
        return str(attrs)
    try:
        return str(tensor)
    except Exception:
        return repr(tensor)


def main() -> None:
    args = parse_args()
    network = stai_mpu_network(model_path=args.model, use_hw_acceleration=True)
    print("backend", network.get_backend_engine())
    print("num_inputs", network.get_num_inputs())
    print("num_outputs", network.get_num_outputs())
    print("inputs")
    for tensor in network.get_input_infos():
        print(tensor_summary(tensor))
    print("outputs")
    for tensor in network.get_output_infos():
        print(tensor_summary(tensor))


if __name__ == "__main__":
    main()
