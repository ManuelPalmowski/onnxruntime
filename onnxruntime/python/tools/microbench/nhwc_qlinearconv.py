# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from benchmark import BenchmarkOp, add_arguments
from model_gen import generate_qlinearconv_model, generated_models_dir, make_int8_weight_bias


@dataclass
class OpParam:
    n: int
    cout: int
    cin: int
    h: int
    w: int
    has_static_wb: bool


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
        output = np.zeros((op_param.n, op_param.cout, op_param.h, op_param.w), dtype=np.int8)
        inputs = {
            "x": input_data,
            "x_scale": x_scale,
            "x_zero_point": x_zero_point,
            "y_scale": y_scale,
            "y_zero_point": y_zero_point,
        }
        if not op_param.has_static_wb:
            weight, bias = make_int8_weight_bias(op_param.cout, op_param.cin)
            inputs["w"] = weight
            inputs["w_scale"] = np.array(0.03, dtype=np.float32)
            inputs["w_zero_point"] = np.array(0, dtype=np.int8)
            inputs["b"] = bias
        outputs = {"y": output}
        return inputs, outputs

    def create_cases(self):
        base_model = Path(__file__).resolve().parent / "models" / "qlinearconv_nhwc_int8.onnx"
        output_dir = generated_models_dir(__file__)
        shape_matrix = [
            OpParam(2, 64, 64, 320, 320, False),
            OpParam(2, 128, 128, 160, 160, False),
            OpParam(2, 320, 320, 64, 64, False),
        ]

        for shape in shape_matrix:
            static_shape = OpParam(shape.n, shape.cout, shape.cin, shape.h, shape.w, True)
            static_model = generate_qlinearconv_model(base_model, output_dir,
                                                      n=static_shape.n, cout=static_shape.cout, cin=static_shape.cin,
                                                      h=static_shape.h, w=static_shape.w, has_static_wb=True)
            self.add_case(static_shape, static_model)

    @classmethod
    def case_profile(cls, op_param, time):
        wb_mode = "static_wb" if op_param.has_static_wb else "dynamic_wb"
        profile = (
            f"( mode n cout cin h w ) = ( {wb_mode} {op_param.n} {op_param.cout} {op_param.cin} {op_param.h} {op_param.w} ), "
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
