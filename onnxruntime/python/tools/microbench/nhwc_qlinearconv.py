# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from benchmark import BenchmarkOp, add_arguments
from model_gen import ConvAttrConfig, generate_qlinearconv_model, generated_models_dir, infer_output_hw, make_int8_weight_bias


@dataclass
class OpParam:
    n: int
    cout: int
    cin: int
    h: int
    w: int
    has_static_wb: bool
    attr_cfg: ConvAttrConfig


BALANCED_CHANNELS = [1, 4, 8, 16, 32, 64, 128, 320, 3, 5, 7, 9, 15, 17]
BALANCED_SIZES = [(320, 320), (160, 160), (64, 64)]
FULL_COVERAGE = False


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


def _balanced_cases():
    cases = []
    attrs = _attr_matrix()
    baseline = attrs[0]

    for cin in BALANCED_CHANNELS:
        cases.append(OpParam(2, cin, cin, 64, 64, True, baseline))

    for h, w in [(160, 160), (320, 320)]:
        cases.append(OpParam(2, 32, 32, h, w, True, baseline))

    sample_channels = [8, 17, 32]
    extra_sizes = [(64, 64), (160, 160), (320, 320)]
    for attr_cfg in attrs[1:]:
        for cin in sample_channels:
            cout = cin
            if not _is_valid_group_case(cin, cout, attr_cfg):
                continue
            effective_cfg = _effective_attr_cfg(cin, attr_cfg)
            cases.append(OpParam(2, cout, cin, 64, 64, True, effective_cfg))

        for h, w in extra_sizes:
            cin = 32
            cout = 32
            if not _is_valid_group_case(cin, cout, attr_cfg):
                continue
            effective_cfg = _effective_attr_cfg(cin, attr_cfg)
            cases.append(OpParam(2, cout, cin, h, w, True, effective_cfg))

    if FULL_COVERAGE:
        full_channels = BALANCED_CHANNELS + [31, 33, 63, 65]
        for attr_cfg in attrs:
            for cin in full_channels:
                cout = cin
                if not _is_valid_group_case(cin, cout, attr_cfg):
                    continue
                effective_cfg = _effective_attr_cfg(cin, attr_cfg)
                for h, w in BALANCED_SIZES:
                    cases.append(OpParam(2, cout, cin, h, w, True, effective_cfg))

    return cases


class BenchmarkNhwcQLinearConv(BenchmarkOp):
    def __init__(self, args):
        BenchmarkOp.__init__(self, args)

    @classmethod
    def create_inputs_outputs(cls, op_param):
        rng = np.random.default_rng(0)
        input_data = rng.integers(-64, 64, size=(op_param.n, op_param.cin, op_param.h, op_param.w), dtype=np.int32).astype(np.int8)
        x_scale = np.array(0.02, dtype=np.float32)
        y_scale = np.array(x_scale * np.float32(0.03), dtype=np.float32)
        x_zero_point = np.array(0, dtype=np.int8)
        y_zero_point = np.array(0, dtype=np.int8)
        out_h, out_w = infer_output_hw(op_param.h, op_param.w, op_param.attr_cfg)
        output = np.zeros((op_param.n, op_param.cout, out_h, out_w), dtype=np.int8)
        inputs = {
            "x": input_data,
            "x_scale": x_scale,
            "x_zero_point": x_zero_point,
            "y_scale": y_scale,
            "y_zero_point": y_zero_point,
        }
        if not op_param.has_static_wb:
            weight, bias = make_int8_weight_bias(
                op_param.cout,
                op_param.cin,
                kernel_hw=op_param.attr_cfg.kernel_hw,
                group=op_param.attr_cfg.group,
            )
            inputs["w"] = weight
            inputs["w_scale"] = np.array(0.03, dtype=np.float32)
            inputs["w_zero_point"] = np.array(0, dtype=np.int8)
            inputs["b"] = bias
        outputs = {"y": output}
        return inputs, outputs

    def create_cases(self):
        base_model = Path(__file__).resolve().parent / "models" / "qlinearconv_nhwc_int8.onnx"
        output_dir = generated_models_dir(__file__)
        for shape in _balanced_cases():
            static_shape = OpParam(shape.n, shape.cout, shape.cin, shape.h, shape.w, True, shape.attr_cfg)
            static_model = generate_qlinearconv_model(base_model, output_dir,
                                                      n=static_shape.n, cout=static_shape.cout, cin=static_shape.cin,
                                                      h=static_shape.h, w=static_shape.w, has_static_wb=True,
                                                      attr_cfg=static_shape.attr_cfg)
            self.add_case(static_shape, static_model)

    @classmethod
    def case_profile(cls, op_param, time):
        wb_mode = "static_wb" if op_param.has_static_wb else "dynamic_wb"
        profile = (
            f"( mode attr n cout cin h w ) = ( {wb_mode} {op_param.attr_cfg.tag} {op_param.n} "
            f"{op_param.cout} {op_param.cin} {op_param.h} {op_param.w} ), "
            f"{time * 1000:7.4f} us"
        )
        return profile


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    bm = BenchmarkNhwcQLinearConv(args)
    bm.benchmark()


if __name__ == "__main__":
    main()
