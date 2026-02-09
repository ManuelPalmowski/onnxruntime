# -------------------------------------------------------------------------
# Copyright (c) Microsoft Corporation. All rights reserved.
# Licensed under the MIT License.
# --------------------------------------------------------------------------

import argparse
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from benchmark import BenchmarkOp, add_arguments
from model_gen import generate_conv_model, generated_models_dir, make_float_weight_bias


@dataclass
class OpParam:
    n: int
    cout: int
    cin: int
    h: int
    w: int
    has_static_wb: bool

    data_type: type


class BenchmarkConv(BenchmarkOp):
    def __init__(self, args):
        BenchmarkOp.__init__(self, args)

    @classmethod
    def create_inputs_outputs(cls, op_param):
        rng = np.random.default_rng(0)
        input_data = rng.random((op_param.n, op_param.cin, op_param.h, op_param.w), dtype=np.float32).astype(op_param.data_type)
        output = rng.random((op_param.n, op_param.cout, op_param.h, op_param.w), dtype=np.float32).astype(op_param.data_type)
        inputs = {"input": input_data}
        if not op_param.has_static_wb:
            weight, bias = make_float_weight_bias(op_param.cout, op_param.cin, op_param.data_type)
            inputs["weight"] = weight
            inputs["bias"] = bias
        outputs = {"conv": output}
        return inputs, outputs

    def create_cases(self):
        data_type = np.float16 if self.args.precision == "fp16" else np.float32
        base_model_name = "conv_fp16.onnx" if self.args.precision == "fp16" else "conv_fp32.onnx"
        base_model = Path(__file__).resolve().parent / "models" / base_model_name
        output_dir = generated_models_dir(__file__)

        shape_matrix = [
            OpParam(2, 64, 64, 320, 320, False, data_type),
            OpParam(2, 128, 128, 160, 160, False, data_type),
            OpParam(2, 320, 320, 64, 64, False, data_type),
        ]

        for shape in shape_matrix:
            dynamic_model = generate_conv_model(base_model, output_dir,
                                                n=shape.n, cout=shape.cout, cin=shape.cin, h=shape.h, w=shape.w,
                                                dtype=shape.data_type, has_static_wb=False)
            self.add_case(shape, dynamic_model)
            static_shape = OpParam(shape.n, shape.cout, shape.cin, shape.h, shape.w, True, data_type)
            static_model = generate_conv_model(base_model, output_dir,
                                               n=static_shape.n, cout=static_shape.cout, cin=static_shape.cin,
                                               h=static_shape.h, w=static_shape.w,
                                               dtype=static_shape.data_type, has_static_wb=True)
            self.add_case(static_shape, static_model)

    @classmethod
    def case_profile(cls, op_param, time):
        wb_mode = "static_wb" if op_param.has_static_wb else "dynamic_wb"
        profile = f"( mode n cout cin h w ) = ( {wb_mode} {op_param.n} {op_param.cout} {op_param.cin} {op_param.h} {op_param.w} ), {time * 1000:7.4f} us"
        return profile


def main():
    parser = argparse.ArgumentParser()
    add_arguments(parser)
    args = parser.parse_args()
    bm = BenchmarkConv(args)
    bm.benchmark()


if __name__ == "__main__":
    main()
