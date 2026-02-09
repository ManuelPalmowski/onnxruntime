from pathlib import Path

import numpy as np
import onnx
from onnx import TensorProto, helper, numpy_helper


def generated_models_dir(script_path: str) -> Path:
    output_dir = Path(script_path).resolve().parent / "models" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    return output_dir


def make_float_weight_bias(cout: int, cin: int, dtype: type):
    rng = np.random.default_rng(0)
    weight = rng.random((cout, cin, 3, 3), dtype=np.float32).astype(dtype)
    bias = rng.random((cout,), dtype=np.float32).astype(dtype)
    return weight, bias


def make_int8_weight_bias(cout: int, cin: int):
    rng = np.random.default_rng(0)
    weight = rng.integers(-63, 64, size=(cout, cin, 3, 3), dtype=np.int32).astype(np.int8)
    bias = rng.integers(-512, 512, size=(cout,), dtype=np.int32)
    return weight, bias


def set_symbolic_nchw_io(model: onnx.ModelProto, input_name: str, output_name: str):
    for vi in model.graph.input:
        if vi.name == input_name:
            dims = vi.type.tensor_type.shape.dim
            dims[0].ClearField("dim_value")
            dims[0].dim_param = "n"
            dims[1].ClearField("dim_value")
            dims[1].dim_param = "cin"
            dims[2].ClearField("dim_value")
            dims[2].dim_param = "h"
            dims[3].ClearField("dim_value")
            dims[3].dim_param = "w"

    for vo in model.graph.output:
        if vo.name == output_name:
            dims = vo.type.tensor_type.shape.dim
            dims[0].ClearField("dim_value")
            dims[0].dim_param = "n"
            dims[1].ClearField("dim_value")
            dims[1].dim_param = "cout"
            dims[2].ClearField("dim_value")
            dims[2].dim_param = "h"
            dims[3].ClearField("dim_value")
            dims[3].dim_param = "w"


def _replace_initializers(model: onnx.ModelProto, tensors_by_name: dict):
    retained = [t for t in model.graph.initializer if t.name not in tensors_by_name]
    retained.extend([numpy_helper.from_array(array, name) for name, array in tensors_by_name.items()])
    del model.graph.initializer[:]
    model.graph.initializer.extend(retained)


def generate_conv_model(base_model: Path, output_dir: Path, *, n: int, cout: int, cin: int, h: int, w: int,
                        dtype: type, has_static_wb: bool) -> str:
    model = onnx.load(str(base_model))

    if has_static_wb:
        weight, bias = make_float_weight_bias(cout, cin, dtype)
        _replace_initializers(model, {"weight": weight, "bias": bias})
        kept_inputs = [vi for vi in model.graph.input if vi.name not in ("weight", "bias")]
        del model.graph.input[:]
        model.graph.input.extend(kept_inputs)

    set_symbolic_nchw_io(model, "input", "conv")
    onnx.checker.check_model(model)

    dtype_tag = "fp16" if dtype == np.float16 else "fp32"
    mode_tag = "static_wb" if has_static_wb else "dynamic_wb"
    model_name = f"conv_{dtype_tag}_{mode_tag}_n{n}_cout{cout}_cin{cin}_h{h}_w{w}.onnx"
    model_path = output_dir / model_name
    onnx.save(model, str(model_path))
    return str(model_path)


def generate_qlinearconv_model(base_model: Path, output_dir: Path, *, n: int, cout: int, cin: int, h: int, w: int,
                               has_static_wb: bool) -> str:
    model = onnx.load(str(base_model))

    weight, bias = make_int8_weight_bias(cout, cin)
    w_scale = np.array(0.03, dtype=np.float32)
    w_zero_point = np.array(0, dtype=np.int8)

    if has_static_wb:
        _replace_initializers(model, {
            "w": weight,
            "w_scale": w_scale,
            "w_zero_point": w_zero_point,
            "b": bias,
        })
        kept_inputs = [vi for vi in model.graph.input if vi.name not in ("w", "w_scale", "w_zero_point", "b")]
        del model.graph.input[:]
        model.graph.input.extend(kept_inputs)
    else:
        for name in ("w", "w_scale", "w_zero_point", "b"):
            existing = [t for t in model.graph.initializer if t.name == name]
            for t in existing:
                model.graph.initializer.remove(t)

        input_names = {vi.name for vi in model.graph.input}
        expected_inputs = {
            "w": helper.make_tensor_value_info("w", TensorProto.INT8, [cout, cin, 3, 3]),
            "w_scale": helper.make_tensor_value_info("w_scale", TensorProto.FLOAT, []),
            "w_zero_point": helper.make_tensor_value_info("w_zero_point", TensorProto.INT8, []),
            "b": helper.make_tensor_value_info("b", TensorProto.INT32, [cout]),
        }
        for name, value_info in expected_inputs.items():
            if name not in input_names:
                model.graph.input.append(value_info)

    set_symbolic_nchw_io(model, "x", "y")
    onnx.checker.check_model(model)

    mode_tag = "static_wb" if has_static_wb else "dynamic_wb"
    model_name = f"qlinearconv_nhwc_int8_{mode_tag}_n{n}_cout{cout}_cin{cin}_h{h}_w{w}.onnx"
    model_path = output_dir / model_name
    onnx.save(model, str(model_path))
    return str(model_path)
