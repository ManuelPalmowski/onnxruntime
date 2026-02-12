# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
from dataclasses import dataclass
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import onnxruntime as ort
from benchmark import Benchmark
from model_gen import ConvAttrConfig, generate_conv_model, generate_qlinearconv_model, generated_models_dir, infer_output_hw


ALL_CHANNELS = [1, 3, 4, 5, 7, 8, 9, 15, 16, 17, 31, 32, 33, 63, 64, 65, 128, 320]
ALL_SPATIAL_SIZES = [(64, 64), (160, 160), (320, 320)]


@dataclass(frozen=True)
class CaseParam:
    n: int
    cin: int
    cout: int
    h: int
    w: int
    attr_cfg: ConvAttrConfig


def _attr_matrix():
    return [
        ConvAttrConfig(tag="k3_p1_s1_d1", kernel_hw=(3, 3), pads=(1, 1, 1, 1), strides=(1, 1), dilations=(1, 1), group=1),
        ConvAttrConfig(tag="k3_p1_s2_d1", kernel_hw=(3, 3), pads=(1, 1, 1, 1), strides=(2, 2), dilations=(1, 1), group=1),
        ConvAttrConfig(tag="k3_p2_s1_d2", kernel_hw=(3, 3), pads=(2, 2, 2, 2), strides=(1, 1), dilations=(2, 2), group=1),
        ConvAttrConfig(tag="k1_p0_s1_d1", kernel_hw=(1, 1), pads=(0, 0, 0, 0), strides=(1, 1), dilations=(1, 1), group=1),
        ConvAttrConfig(tag="k3_p1_s1_d1_g4", kernel_hw=(3, 3), pads=(1, 1, 1, 1), strides=(1, 1), dilations=(1, 1), group=4),
        ConvAttrConfig(tag="k3_p1_s1_d1_dw", kernel_hw=(3, 3), pads=(1, 1, 1, 1), strides=(1, 1), dilations=(1, 1), group=-1),
    ]


def _effective_attr_cfg(cin: int, attr_cfg: ConvAttrConfig) -> ConvAttrConfig:
    if attr_cfg.group != -1:
        return attr_cfg
    return ConvAttrConfig(
        tag=attr_cfg.tag,
        kernel_hw=attr_cfg.kernel_hw,
        pads=attr_cfg.pads,
        strides=attr_cfg.strides,
        dilations=attr_cfg.dilations,
        group=cin,
        auto_pad=attr_cfg.auto_pad,
    )


def _is_valid_group_case(cin: int, cout: int, attr_cfg: ConvAttrConfig) -> bool:
    group = cin if attr_cfg.group == -1 else attr_cfg.group
    if cin % group != 0 or cout % group != 0:
        return False
    if attr_cfg.group == -1 and cout != cin:
        return False
    return True


def all_cases():
    cases = []
    attrs = _attr_matrix()

    for attr_cfg in attrs:
        for cin in ALL_CHANNELS:
            cout = cin
            if not _is_valid_group_case(cin, cout, attr_cfg):
                continue
            effective_cfg = _effective_attr_cfg(cin, attr_cfg)
            for h, w in ALL_SPATIAL_SIZES:
                cases.append(CaseParam(2, cin, cout, h, w, effective_cfg))

    return cases


def make_conv_io(case: CaseParam, dtype: type):
    rng = np.random.default_rng(0)
    x = rng.random((case.n, case.cin, case.h, case.w), dtype=np.float32).astype(dtype)
    out_h, out_w = infer_output_hw(case.h, case.w, case.attr_cfg)
    y = rng.random((case.n, case.cout, out_h, out_w), dtype=np.float32).astype(dtype)
    return {"input": x}, {"conv": y}


def make_qlinear_io(case: CaseParam):
    rng = np.random.default_rng(0)
    x = rng.integers(-64, 64, size=(case.n, case.cin, case.h, case.w), dtype=np.int32).astype(np.int8)
    out_h, out_w = infer_output_hw(case.h, case.w, case.attr_cfg)
    y = np.zeros((case.n, case.cout, out_h, out_w), dtype=np.int8)
    inputs = {
        "x": x,
        "x_scale": np.array(0.02, dtype=np.float32),
        "x_zero_point": np.array(0, dtype=np.int8),
        "y_scale": np.array(np.float32(0.02) * np.float32(0.03), dtype=np.float32),
        "y_zero_point": np.array(0, dtype=np.int8),
    }
    outputs = {"y": y}
    return inputs, outputs


def run_case(model: str, inputs: dict, outputs: dict, *, nhwc: bool, profiling: bool, cuda_algo_search: str):
    args = SimpleNamespace(provider="cuda", nhwc=nhwc, profiling=profiling, cuda_algo_search=cuda_algo_search)
    bench = Benchmark(model, inputs, outputs, args)
    try:
        return bench.benchmark(), None
    except RuntimeError as error:
        return None, str(error).splitlines()[0]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--precision", type=str, choices=["fp16", "fp32"], default="fp32", help="Conv precision")
    parser.add_argument("--profiling", action="store_true", default=False, help="Enable ORT profiling")
    parser.add_argument("--nhwc", action="store_true", default=True, help="Enable CUDA prefer_nhwc (default: true)")
    parser.add_argument("--no-nhwc", action="store_false", dest="nhwc", help="Disable CUDA prefer_nhwc")
    parser.add_argument("--max-cases", type=int, default=0, help="Run at most N cases (0 means all)")
    args = parser.parse_args()

    if "CUDAExecutionProvider" not in ort.get_available_providers():
        raise RuntimeError("CUDAExecutionProvider is not available in this environment")

    dtype = np.float16 if args.precision == "fp16" else np.float32
    base_dir = Path(__file__).resolve().parent
    out_dir = generated_models_dir(__file__)
    conv_base = base_dir / "models" / ("conv_fp16.onnx" if dtype == np.float16 else "conv_fp32.onnx")
    qconv_base = base_dir / "models" / "qlinearconv_nhwc_int8.onnx"

    cases = all_cases()
    if args.max_cases > 0:
        cases = cases[: args.max_cases]

    algo_modes = ["HEURISTIC", "EXHAUSTIVE"]

    print("algo,attr,n,cin,cout,h,w,conv_us,qlinear_us,ratio")
    ok_pairs = []

    for case in cases:
        conv_model = generate_conv_model(
            conv_base,
            out_dir,
            n=case.n,
            cout=case.cout,
            cin=case.cin,
            h=case.h,
            w=case.w,
            dtype=dtype,
            has_static_wb=True,
            attr_cfg=case.attr_cfg,
        )
        q_model = generate_qlinearconv_model(
            qconv_base,
            out_dir,
            n=case.n,
            cout=case.cout,
            cin=case.cin,
            h=case.h,
            w=case.w,
            has_static_wb=True,
            attr_cfg=case.attr_cfg,
        )

        conv_inputs, conv_outputs = make_conv_io(case, dtype)
        q_inputs, q_outputs = make_qlinear_io(case)

        for algo in algo_modes:
            conv_us, conv_err = run_case(
                conv_model,
                conv_inputs,
                conv_outputs,
                nhwc=args.nhwc,
                profiling=args.profiling,
                cuda_algo_search=algo,
            )
            q_us, q_err = run_case(
                q_model,
                q_inputs,
                q_outputs,
                nhwc=args.nhwc,
                profiling=args.profiling,
                cuda_algo_search=algo,
            )

            conv_str = f"{conv_us * 1000:.4f}" if conv_us is not None else "N/A"
            q_str = f"{q_us * 1000:.4f}" if q_us is not None else "N/A"
            if conv_us is not None and q_us is not None and conv_us > 0:
                ratio = q_us / conv_us
                ratio_str = f"{ratio:.4f}"
                ok_pairs.append(ratio)
            else:
                ratio_str = "N/A"

            print(f"{algo},{case.attr_cfg.tag},{case.n},{case.cin},{case.cout},{case.h},{case.w},{conv_str},{q_str},{ratio_str}")
            if conv_err:
                print(f"# conv_error[{algo}]: {conv_err}")
            if q_err:
                print(f"# qlinear_error[{algo}]: {q_err}")

    if ok_pairs:
        print(f"# paired_cases={len(ok_pairs)}")
        print(f"# ratio_mean={float(np.mean(ok_pairs)):.4f}")
        print(f"# ratio_median={float(np.median(ok_pairs)):.4f}")
    else:
        print("# paired_cases=0")


if __name__ == "__main__":
    main()
