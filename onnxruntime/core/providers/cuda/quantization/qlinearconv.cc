// Licensed under the MIT License.
#include "core/providers/cuda/quantization/qlinearconv.h"

#include <algorithm>
#include <cmath>
#include <limits>

#include "core/common/common.h"
#include "core/providers/common.h"
#include "core/providers/cuda/tensor/transpose.h"

namespace onnxruntime {
namespace cuda {

using ConvPadVector = ConvAttributes::ConvPadVector;

QLinearConv::QLinearConv(const OpKernelInfo& info) : CudaKernel(info), conv_attrs_(info) {
  is_nhwc_domain_ = info.node().Domain() == kMSInternalNHWCDomain;
  channels_last_ = is_nhwc_domain_;
  std::string activation;
  if (info.GetAttr<std::string>("activation", &activation).IsOK()) {
    ORT_ENFORCE(activation == "Relu", "QLinearConv: only Relu activation is supported for fusion.");
    fuse_relu_ = true;
  }
  auto pads_size = conv_attrs_.pads.size();
  ORT_ENFORCE(pads_size % 2 == 0);
}

Status QLinearConv::PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                            bool& is_packed, PrePackedWeights* /*prepacked_weights*/) {
  is_packed = false;
  if (!is_nhwc_domain_ || input_idx != InputTensors::IN_W) {
    return Status::OK();
  }

  const auto& orig_shape = tensor.Shape();
  const auto shape_size = orig_shape.NumDimensions();
  if (shape_size <= 2) {
    return Status::OK();
  }

  std::vector<size_t> perm;
  perm.reserve(shape_size);
  perm.push_back(0);
  for (size_t i = 2; i < shape_size; ++i) {
    perm.push_back(i);
  }
  perm.push_back(1);

  TensorShapeVector nhwc_dims;
  nhwc_dims.reserve(shape_size);
  for (size_t i = 0; i < shape_size; ++i) {
    nhwc_dims.push_back(orig_shape[perm[i]]);
  }

  auto weight_sum_alloc = alloc;
  W_ = Tensor::Create(tensor.DataType(), TensorShape(nhwc_dims), std::move(alloc));
  auto status = cuda::Transpose::DoTranspose(GetDeviceProp(), DefaultCudaStream(), DefaultCublasHandle(),
                                             gsl::span<const size_t>(perm.data(), perm.size()),
                                             tensor, *W_);
  if (!status.IsOK()) {
    return status;
  }

  if (tensor.IsDataType<int8_t>()) {
    const int64_t output_channels = orig_shape[0];
    if (output_channels > 0) {
      cached_weight_sum_ = IAllocator::MakeUniquePtr<int32_t>(std::move(weight_sum_alloc),
                                                              static_cast<size_t>(output_channels));
      cached_weight_sum_size_ = output_channels;
      const int64_t weight_block_size = W_->Shape().Size() / output_channels;
      ORT_RETURN_IF_ERROR(QLinearConvComputeWeightSum(DefaultCudaStream(),
                                                      W_->Data<int8_t>(),
                                                      cached_weight_sum_.get(),
                                                      output_channels,
                                                      weight_block_size));
    }
  } else {
    cached_weight_sum_.reset();
    cached_weight_sum_size_ = 0;
  }

  CUDA_CALL_THROW(cudaStreamSynchronize(DefaultCudaStream()));
  is_packed = true;
  return Status::OK();
}

#if !defined(__CUDACC__) && !defined(USE_CUDA_MINIMAL) && CUDNN_MAJOR >= 9
Status QLinearConv::UpdateState(const TensorShapeVector& x_dims,
                                const TensorShapeVector& w_dims,
                                const TensorShapeVector& y_dims,
                                const std::vector<int64_t>& pads,
                                const std::vector<int64_t>& strides,
                                const std::vector<int64_t>& dilations,
                                cudnn_frontend::DataType_t x_type,
                                cudnn_frontend::DataType_t w_type,
                                cudnn_frontend::DataType_t y_type,
                                bool fused_quant,
                                bool fused_bias,
                                bool fused_relu,
                                bool channels_last,
                                bool w_in_nhwc,
                                cudnnHandle_t handle,
                                cudnn_frontend::HeurMode_t heur_mode) const {
  if (cudnn_fe_state_.graph && cudnn_fe_state_.x_dims == x_dims &&
      cudnn_fe_state_.w_dims == w_dims && cudnn_fe_state_.y_dims == y_dims &&
      cudnn_fe_state_.pads == pads && cudnn_fe_state_.strides == strides &&
      cudnn_fe_state_.dilations == dilations && cudnn_fe_state_.x_type == x_type &&
      cudnn_fe_state_.w_type == w_type && cudnn_fe_state_.y_type == y_type &&
      cudnn_fe_state_.fused_quant == fused_quant && cudnn_fe_state_.fused_bias == fused_bias &&
      cudnn_fe_state_.fused_relu == fused_relu &&
      cudnn_fe_state_.channels_last == channels_last &&
      cudnn_fe_state_.w_in_nhwc == w_in_nhwc) {
    return Status::OK();
  }

  cudnn_fe_state_.x_dims = x_dims;
  cudnn_fe_state_.w_dims = w_dims;
  cudnn_fe_state_.y_dims = y_dims;
  cudnn_fe_state_.pads = pads;
  cudnn_fe_state_.strides = strides;
  cudnn_fe_state_.dilations = dilations;
  cudnn_fe_state_.x_type = x_type;
  cudnn_fe_state_.w_type = w_type;
  cudnn_fe_state_.y_type = y_type;
  cudnn_fe_state_.fused_quant = fused_quant;
  cudnn_fe_state_.fused_bias = fused_bias;
  cudnn_fe_state_.fused_relu = fused_relu;
  cudnn_fe_state_.channels_last = channels_last;
  cudnn_fe_state_.w_in_nhwc = w_in_nhwc;
  cudnn_fe_state_.variant_pack.clear();
  cudnn_fe_state_.graph = std::make_unique<cudnn_frontend::graph::Graph>();

  auto intermediate_type = fused_quant ? cudnn_frontend::DataType_t::FLOAT : cudnn_frontend::DataType_t::INT32;
  auto compute_type = fused_quant ? cudnn_frontend::DataType_t::FLOAT : cudnn_frontend::DataType_t::INT32;
  cudnn_fe_state_.graph->set_io_data_type(x_type)
      .set_intermediate_data_type(intermediate_type)
      .set_compute_data_type(compute_type);

  cudnn_fe_state_.x = cudnn_fe_state_.graph->tensor(CudnnFeTensor(x_dims, "x", x_type, true).Get());
  cudnn_fe_state_.w = cudnn_fe_state_.graph->tensor(CudnnFeTensor(w_dims, "w", w_type, true).Get());

  auto conv_options = cudnn_frontend::graph::Conv_fprop_attributes()
                          .set_compute_data_type(cudnn_frontend::DataType_t::INT32)
                          .set_pre_padding(std::vector<int64_t>(pads.begin(), pads.begin() + pads.size() / 2))
                          .set_post_padding(std::vector<int64_t>(pads.begin() + pads.size() / 2, pads.end()))
                          .set_stride(strides)
                          .set_dilation(dilations);

  auto conv_output = cudnn_fe_state_.graph->conv_fprop(cudnn_fe_state_.x, cudnn_fe_state_.w, conv_options);
  std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> output_tensor = conv_output;
  cudnn_fe_state_.bias.reset();
  cudnn_fe_state_.scale.reset();
  if (fused_bias) {
    TensorShapeVector b_dims(y_dims.size(), 1);
    if (b_dims.size() > 1) {
      b_dims[1] = y_dims[1];
    }
    cudnn_fe_state_.bias = cudnn_fe_state_.graph->tensor(
        CudnnFeTensor(b_dims, "b", cudnn_frontend::DataType_t::FLOAT, true).Get());
    auto add_options = cudnn_frontend::graph::Pointwise_attributes().set_mode(cudnn_frontend::PointwiseMode_t::ADD);
    output_tensor = cudnn_fe_state_.graph->pointwise(output_tensor, cudnn_fe_state_.bias, add_options);
  }

  if (fused_quant) {
    TensorShapeVector scale_dims(y_dims.size(), 1);
    if (scale_dims.size() > 1) {
      scale_dims[1] = y_dims[1];
    }
    cudnn_fe_state_.scale = cudnn_fe_state_.graph->tensor(
        CudnnFeTensor(scale_dims, "s", cudnn_frontend::DataType_t::FLOAT, true).Get());
    auto mul_options = cudnn_frontend::graph::Pointwise_attributes().set_mode(cudnn_frontend::PointwiseMode_t::MUL);
    auto quantized_output = cudnn_fe_state_.graph->pointwise(output_tensor, cudnn_fe_state_.scale, mul_options);
    if (fused_relu) {
      auto relu_options = cudnn_frontend::graph::Pointwise_attributes().set_mode(cudnn_frontend::PointwiseMode_t::RELU_FWD);
      quantized_output = cudnn_fe_state_.graph->pointwise(quantized_output, relu_options);
    }
    cudnn_fe_state_.y = quantized_output;
  } else {
    auto identity = cudnn_frontend::graph::Pointwise_attributes().set_mode(cudnn_frontend::PointwiseMode_t::IDENTITY);
    cudnn_fe_state_.y = cudnn_fe_state_.graph->pointwise(output_tensor, output_tensor, identity);
  }

  auto y_tensor = CudnnFeTensor(y_dims, "y", y_type, true).Get();
  cudnn_fe_state_.y->set_dim(y_tensor.get_dim());
  cudnn_fe_state_.y->set_stride(y_tensor.get_stride());
  cudnn_fe_state_.y->set_output(true);
  cudnn_fe_state_.y->set_data_type(y_type);

  try {
    CUDNN_FE_CALL_THROW(cudnn_fe_state_.graph->validate());
    CUDNN_FE_CALL_THROW(cudnn_fe_state_.graph->build_operation_graph(handle));
    try {
      CUDNN_FE_CALL_THROW(cudnn_fe_state_.graph->create_execution_plans({heur_mode}));
    } catch (const std::exception& ex) {
      if (heur_mode == cudnn_frontend::HeurMode_t::FALLBACK) {
        throw;
      }
      CUDNN_FE_CALL_THROW(cudnn_fe_state_.graph->create_execution_plans({cudnn_frontend::HeurMode_t::FALLBACK}));
    }
  } catch (const std::exception& ex) {
    std::string message = MakeString("Failed to initialize CUDNN Frontend", ex.what(),
                                     "with the cudnn frontend json:\n", cudnn_fe_state_.graph->print());
    return Status(common::StatusCategory::ONNXRUNTIME, common::StatusCode::EP_FAIL, message);
  }

  try {
    CUDNN_FE_CALL_THROW(cudnn_fe_state_.graph->check_support(handle));
    CUDNN_FE_CALL_THROW(cudnn_fe_state_.graph->build_plans(handle));
  } catch (const std::exception& ex) {
    std::string message = MakeString("QLinearConv cuDNN frontend does not support this configuration: ", ex.what(),
                                     "\n", cudnn_fe_state_.graph->print());
    return Status(common::StatusCategory::ONNXRUNTIME, common::StatusCode::EP_FAIL, message);
  }

  cudnn_fe_state_.workspace_bytes = cudnn_fe_state_.graph->get_workspace_size();
  std::vector<uint8_t> serialized_graph;
  auto fe_status = cudnn_fe_state_.graph->serialize(serialized_graph);
  json graph_json = json::from_ubjson(serialized_graph);
  std::string plan_json = graph_json["cudnn_backend_data"].get<std::string>();
  auto plan = cudnn_frontend::ExecutionPlanBuilder().setHandle(handle).loadFromJson(plan_json);
  ORT_UNUSED_PARAMETER(plan);
  return Status::OK();
}
#endif

Status QLinearConv::ComputeInternal(OpKernelContext* context) const {
#if defined(USE_CUDA_MINIMAL) || CUDNN_MAJOR < 9
  ORT_UNUSED_PARAMETER(context);
  return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                         "QLinearConv requires cuDNN frontend (CUDNN_MAJOR >= 9) and full CUDA build.");
#else
  const Tensor* X = context->Input<Tensor>(InputTensors::IN_X);
  const Tensor* W = nullptr;
  if (W_) {
    W = W_.get();
  } else {
    W = context->Input<Tensor>(InputTensors::IN_W);
  }
  const Tensor* B = context->Input<Tensor>(InputTensors::IN_BIAS);
  const Tensor* X_scale = context->Input<Tensor>(InputTensors::IN_X_SCALE);
  const Tensor* X_zero_point = context->Input<Tensor>(InputTensors::IN_X_ZERO_POINT);
  const Tensor* W_scale = context->Input<Tensor>(InputTensors::IN_W_SCALE);
  const Tensor* W_zero_point = context->Input<Tensor>(InputTensors::IN_W_ZERO_POINT);
  const Tensor* Y_scale = context->Input<Tensor>(InputTensors::IN_Y_SCALE);
  const Tensor* Y_zero_point = context->Input<Tensor>(InputTensors::IN_Y_ZERO_POINT);

  const auto* cuda_ep = static_cast<const CUDAExecutionProvider*>(Info().GetExecutionProvider());
  if (!cuda_ep || !is_nhwc_domain_ || cuda_ep->GetPreferredLayout() != DataLayout::NHWC) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "QLinearConv: CUDA requires prefer_nhwc and internal NHWC domain.");
  }

  const auto input_rank = X->Shape().NumDimensions();
  const auto weight_rank = W->Shape().NumDimensions();
  if (input_rank != weight_rank || (input_rank != 3 && input_rank != 4 && input_rank != 5)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "QLinearConv: CUDA supports 1D/2D/3D NHWC only.");
  }

  if (!IsScalarOr1ElementVector(X_scale) || !IsScalarOr1ElementVector(Y_scale)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "QLinearConv: per-tensor scales are required.");
  }

  if (!IsScalarOr1ElementVector(X_zero_point) || !IsScalarOr1ElementVector(Y_zero_point)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED,
                           "QLinearConv: per-tensor x/y zero points are required.");
  }

  const bool x_is_int8 = X->IsDataType<int8_t>();
  const bool w_is_int8 = W->IsDataType<int8_t>();

  if (x_is_int8 && !w_is_int8) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "QLinearConv: int8 input requires int8 weights.");
  }
  if (!(w_is_int8 || x_is_int8)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "QLinearConv: uint8 weights require uint8 input.");
  }

  const int32_t x_zero_point_value = x_is_int8 ? static_cast<int32_t>(*X_zero_point->Data<int8_t>())
                                               : static_cast<int32_t>(*X_zero_point->Data<uint8_t>());
  constexpr int32_t kUint8ZeroPoint = 128;
  const int32_t x_zero_point_int8 = x_is_int8 ? x_zero_point_value : x_zero_point_value - kUint8ZeroPoint;

  std::unique_ptr<Tensor> runtime_weight_nhwc;
  bool weight_channels_last = channels_last_ && (W_ != nullptr || W_already_nhwc_);
  if (channels_last_ && !weight_channels_last) {
    const auto& orig_shape = W->Shape();
    const auto shape_size = orig_shape.NumDimensions();
    std::vector<size_t> perm;
    perm.reserve(shape_size);
    perm.push_back(0);
    for (size_t i = 2; i < shape_size; ++i) {
      perm.push_back(i);
    }
    perm.push_back(1);

    TensorShapeVector nhwc_dims;
    nhwc_dims.reserve(shape_size);
    for (size_t i = 0; i < shape_size; ++i) {
      nhwc_dims.push_back(orig_shape[perm[i]]);
    }

    AllocatorPtr alloc;
    ORT_RETURN_IF_ERROR(context->GetTempSpaceAllocator(&alloc));
    runtime_weight_nhwc = Tensor::Create(W->DataType(), TensorShape(nhwc_dims), alloc);
    ORT_RETURN_IF_ERROR(cuda::Transpose::DoTranspose(GetDeviceProp(),
                                                     Stream(context),
                                                     GetCublasHandle(context),
                                                     gsl::span<const size_t>(perm.data(), perm.size()),
                                                     *W,
                                                     *runtime_weight_nhwc));
    W = runtime_weight_nhwc.get();
    weight_channels_last = true;
  }

  const int64_t N = X->Shape()[0];
  const int64_t M = W->Shape()[0];
  auto is_valid_quant_param = [](const Tensor* quant_param, int64_t channels) {
    const auto& shape = quant_param->Shape();
    return (shape.NumDimensions() == 0 || (shape.NumDimensions() == 1 && (shape[0] == 1 || shape[0] == channels)));
  };

  if (!is_valid_quant_param(W_scale, M)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "QLinearConv : filter scale shape invalid");
  }

  if (!is_valid_quant_param(W_zero_point, M)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "QLinearConv : filter zero point shape invalid");
  }

  const float x_scale = *X_scale->Data<float>();
  const float y_scale = *Y_scale->Data<float>();
  const auto* w_scale_data = W_scale->Data<float>();
  const int64_t w_scale_size = W_scale->Shape().Size();
  std::vector<float> requant_scales(static_cast<size_t>(M));
  for (int64_t channel = 0; channel < M; ++channel) {
    const int64_t scale_index = (w_scale_size == 1) ? 0 : channel;
    requant_scales[static_cast<size_t>(channel)] = (x_scale * w_scale_data[scale_index]) / y_scale;
  }

  ORT_RETURN_IF_ERROR(conv_attrs_.ValidateInputShape(X->Shape(), W->Shape(), channels_last_, weight_channels_last));

  TensorShapeVector kernel_shape;
  ORT_RETURN_IF_ERROR(conv_attrs_.ComputeKernelShape(W->Shape(), kernel_shape, weight_channels_last));
  const size_t kernel_rank = kernel_shape.size();
  if (channels_last_ && (kernel_rank < 1 || kernel_rank > 3)) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT,
                           "QLinearConv: channels_last only supports 1D/2D/3D convolution.");
  }

  ConvPadVector pads(conv_attrs_.pads);
  if (pads.empty()) {
    pads.resize(kernel_rank * 2, 0);
  }

  TensorShapeVector dilations(conv_attrs_.dilations);
  if (dilations.empty()) {
    dilations.resize(kernel_rank, 1);
  }

  TensorShapeVector strides(conv_attrs_.strides);
  if (strides.empty()) {
    strides.resize(kernel_rank, 1);
  }

  TensorShapeVector y_dims;
  y_dims.reserve(kernel_rank + 2);
  if (channels_last_) {
    y_dims.push_back(N);
  } else {
    y_dims.insert(y_dims.begin(), {N, M});
  }

  const size_t spatial_dim_start = channels_last_ ? 1 : 2;
  TensorShape input_shape = X->Shape().Slice(spatial_dim_start, spatial_dim_start + kernel_rank);
  ORT_RETURN_IF_ERROR(conv_attrs_.InferPadsAndOutputShape(input_shape, kernel_shape, strides, dilations, pads, y_dims));
  if (channels_last_) {
    y_dims.push_back(M);
  }

  ConvPadVector conv_pads = pads;

  if (B != nullptr) {
    if (B->Shape().NumDimensions() != 1 || B->Shape().Size() != M) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "QLinearConv: bias must be 1-D with size equal to output channels.");
    }
  }

  Tensor* Y = context->Output(OutputTensors::OUT_Y, TensorShape(y_dims));
  if (Y->Shape().Size() == 0) {
    return Status::OK();
  }

  const bool y_is_int8 = Y->IsDataType<int8_t>();
  if (x_is_int8 != y_is_int8) {
    return ORT_MAKE_STATUS(ONNXRUNTIME, INVALID_ARGUMENT, "QLinearConv: output type must match input type.");
  }

  const int32_t y_zero_point_value = y_is_int8 ? static_cast<int32_t>(*Y_zero_point->Data<int8_t>())
                                               : static_cast<int32_t>(*Y_zero_point->Data<uint8_t>());

  bool w_zero_point_all_128 = true;
  bool w_zero_point_all_0 = true;
  if (!w_is_int8) {
    const auto* w_zero_point_data = W_zero_point->Data<uint8_t>();
    const int64_t w_zero_point_size = W_zero_point->Shape().Size();
    for (int64_t i = 0; i < w_zero_point_size; ++i) {
      if (static_cast<int32_t>(w_zero_point_data[i]) != kUint8ZeroPoint) {
        w_zero_point_all_128 = false;
      }
      if (w_zero_point_data[i] != 0) {
        w_zero_point_all_0 = false;
      }
      if (!w_zero_point_all_128 && !w_zero_point_all_0) {
        break;
      }
    }
  } else {
    const auto* w_zero_point_data = W_zero_point->Data<int8_t>();
    const int64_t w_zero_point_size = W_zero_point->Shape().Size();
    for (int64_t i = 0; i < w_zero_point_size; ++i) {
      if (w_zero_point_data[i] != 0) {
        w_zero_point_all_0 = false;
        break;
      }
    }
    w_zero_point_all_128 = false;
  }

  if (!x_is_int8) {
    if (!w_is_int8 && !w_zero_point_all_128) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "QLinearConv: uint8 weight zero point must be 128.");
    }
    if (w_is_int8 && !w_zero_point_all_0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "QLinearConv: int8 weight zero point must be 0.");
    }
  } else {
    if (!w_zero_point_all_0) {
      return ORT_MAKE_STATUS(ONNXRUNTIME, NOT_IMPLEMENTED, "QLinearConv: int8 weight zero point must be 0.");
    }
  }

  auto cuda_stream = static_cast<cudaStream_t>(context->GetComputeStream()->GetHandle());

  const int8_t* x_data = x_is_int8 ? X->Data<int8_t>() : nullptr;
  const int8_t* w_data = w_is_int8 ? W->Data<int8_t>() : nullptr;
  IAllocatorUniquePtr<int8_t> x_int8;
  IAllocatorUniquePtr<int8_t> w_int8;

  if (!x_is_int8) {
    x_int8 = GetScratchBuffer<int8_t>(static_cast<size_t>(X->Shape().Size()), context->GetComputeStream());
    ORT_RETURN_IF_ERROR(QLinearConvConvertUint8ToInt8(cuda_stream,
                                                      X->Data<uint8_t>(),
                                                      x_int8.get(),
                                                      X->Shape().Size(),
                                                      kUint8ZeroPoint));
    x_data = x_int8.get();
  }

  if (!w_is_int8) {
    w_int8 = GetScratchBuffer<int8_t>(static_cast<size_t>(W->Shape().Size()), context->GetComputeStream());
    ORT_RETURN_IF_ERROR(QLinearConvConvertUint8ToInt8(cuda_stream,
                                                      W->Data<uint8_t>(),
                                                      w_int8.get(),
                                                      W->Shape().Size(),
                                                      kUint8ZeroPoint));
    w_data = w_int8.get();
  }

  TensorShapeVector x_dims = X->Shape().AsShapeVector();
  IAllocatorUniquePtr<int8_t> padded_x;
  const bool needs_int8_padding = x_zero_point_int8 != 0 &&
                                  std::any_of(pads.begin(), pads.end(), [](int64_t pad) { return pad != 0; });
  if (needs_int8_padding) {
    ORT_ENFORCE(channels_last_ && (X->Shape().NumDimensions() == 3 || X->Shape().NumDimensions() == 4 ||
                                   X->Shape().NumDimensions() == 5),
                "QLinearConv: padded int8 NHWC requires 3D/4D/5D input.");
    const auto& x_shape = X->Shape();
    const int64_t batch = x_shape[0];
    const int64_t channels = x_shape[input_rank - 1];
    int64_t input_d = 1;
    int64_t input_h = 1;
    int64_t input_w = 0;
    int64_t pad_front = 0;
    int64_t pad_top = 0;
    int64_t pad_left = 0;
    int64_t pad_back = 0;
    int64_t pad_bottom = 0;
    int64_t pad_right = 0;
    if (kernel_rank == 1) {
      input_w = x_shape[1];
      pad_left = pads[0];
      pad_right = pads[1];
    } else if (kernel_rank == 2) {
      input_h = x_shape[1];
      input_w = x_shape[2];
      pad_top = pads[0];
      pad_left = pads[1];
      pad_bottom = pads[2];
      pad_right = pads[3];
    } else {
      input_d = x_shape[1];
      input_h = x_shape[2];
      input_w = x_shape[3];
      pad_front = pads[0];
      pad_top = pads[1];
      pad_left = pads[2];
      pad_back = pads[3];
      pad_bottom = pads[4];
      pad_right = pads[5];
    }
    const int64_t output_d = input_d + pad_front + pad_back;
    const int64_t output_h = input_h + pad_top + pad_bottom;
    const int64_t output_w = input_w + pad_left + pad_right;
    const int64_t output_size = batch * output_d * output_h * output_w * channels;
    padded_x = GetScratchBuffer<int8_t>(static_cast<size_t>(output_size), context->GetComputeStream());
    if (kernel_rank == 3) {
      ORT_RETURN_IF_ERROR(QLinearConvPadInputNDHWC(cuda_stream,
                                                   x_data,
                                                   padded_x.get(),
                                                   batch,
                                                   input_d,
                                                   input_h,
                                                   input_w,
                                                   output_d,
                                                   output_h,
                                                   output_w,
                                                   channels,
                                                   pad_front,
                                                   pad_top,
                                                   pad_left,
                                                   static_cast<int8_t>(x_zero_point_int8)));
    } else {
      ORT_RETURN_IF_ERROR(QLinearConvPadInputNHWC(cuda_stream,
                                                  x_data,
                                                  padded_x.get(),
                                                  batch,
                                                  input_h,
                                                  input_w,
                                                  output_h,
                                                  output_w,
                                                  channels,
                                                  pad_top,
                                                  pad_left,
                                                  static_cast<int8_t>(x_zero_point_int8)));
    }
    x_data = padded_x.get();
    if (kernel_rank == 1) {
      x_dims = {batch, output_w, channels};
    } else if (kernel_rank == 2) {
      x_dims = {batch, output_h, output_w, channels};
    } else {
      x_dims = {batch, output_d, output_h, output_w, channels};
    }
    conv_pads.assign(kernel_rank * 2, 0);
  }

  int64_t spatial_size = 1;
  if (channels_last_) {
    for (size_t dim = 1; dim + 1 < y_dims.size(); ++dim) {
      spatial_size *= y_dims[dim];
    }
  } else {
    for (size_t dim = 2; dim < y_dims.size(); ++dim) {
      spatial_size *= y_dims[dim];
    }
  }

  const int64_t total_elements = Y->Shape().Size();

  cudnn_frontend::HeurMode_t heur_mode;
  switch (cuda_ep->GetCudnnConvAlgo()) {
    case 0:
      heur_mode = cudnn_frontend::HeurMode_t::B;
      break;
    case 1:
      heur_mode = cudnn_frontend::HeurMode_t::A;
      break;
    case 2:
      heur_mode = cudnn_frontend::HeurMode_t::FALLBACK;
      break;
    default:
      heur_mode = cudnn_frontend::HeurMode_t::A;
      break;
  }

  std::vector<int64_t> stride_vec(strides.begin(), strides.end());
  std::vector<int64_t> dilation_vec(dilations.begin(), dilations.end());
  std::vector<int64_t> pads_vec(conv_pads.begin(), conv_pads.end());
  TensorShapeVector w_dims = W->Shape().AsShapeVector();
  TensorShapeVector y_dims_for_fe = y_dims;
  const bool w_in_nhwc = weight_channels_last;

  auto reorder_dims_for_nhwc = [](TensorShapeVector& dims) {
    if (dims.size() >= 3) {
      dims.insert(dims.begin() + 1, dims.back());
      dims.pop_back();
    }
  };

  if (channels_last_) {
    reorder_dims_for_nhwc(x_dims);
    reorder_dims_for_nhwc(y_dims_for_fe);
    if (w_in_nhwc) {
      reorder_dims_for_nhwc(w_dims);
    }
  }

  if (channels_last_ && input_rank == 3) {
    if (cuda_ep->GetCudnnConv1dPadToNc1d()) {
      x_dims.insert(x_dims.begin() + 2, 1);
      w_dims.insert(w_dims.begin() + 2, 1);
      y_dims_for_fe.insert(y_dims_for_fe.begin() + 2, 1);
      pads_vec.insert(pads_vec.begin() + kernel_rank, 0);
      pads_vec.insert(pads_vec.begin(), 0);
      stride_vec.insert(stride_vec.begin(), 1);
      dilation_vec.insert(dilation_vec.begin(), 1);
    } else {
      x_dims.push_back(1);
      w_dims.push_back(1);
      y_dims_for_fe.push_back(1);
      pads_vec.insert(pads_vec.begin() + kernel_rank, 0);
      pads_vec.insert(pads_vec.end(), 0);
      stride_vec.push_back(1);
      dilation_vec.push_back(1);
    }
  }

  if (!channels_last_ && input_rank == 3) {
    x_dims.insert(x_dims.begin() + 2, 1);
    w_dims.insert(w_dims.begin() + 2, 1);
    y_dims_for_fe.insert(y_dims_for_fe.begin() + 2, 1);
    pads_vec = {0, pads_vec[0], 0, pads_vec[1]};
    stride_vec = {1, stride_vec[0]};
    dilation_vec = {1, dilation_vec[0]};
  }

  const auto x_type = cudnn_frontend::DataType_t::INT8;
  const auto w_type = cudnn_frontend::DataType_t::INT8;
  const bool fused_quant_candidate = ((x_is_int8 && y_is_int8) || (!x_is_int8 && !y_is_int8)) &&
                                     w_is_int8 && w_zero_point_all_0;
  const bool fused_bias = fused_quant_candidate && (B != nullptr || x_zero_point_int8 != 0);
  const bool fused_output_needs_quantize = fused_quant_candidate && (!y_is_int8 || y_zero_point_value != 0);
  bool use_fused_quant = false;

  std::lock_guard<std::mutex> lock(cudnn_fe_state_.mutex);
  if (fused_quant_candidate) {
    Status fused_status = UpdateState(x_dims, w_dims, y_dims_for_fe,
                                      pads_vec, stride_vec, dilation_vec,
                                      x_type, w_type,
                                      fused_output_needs_quantize ? cudnn_frontend::DataType_t::FLOAT
                                                                  : cudnn_frontend::DataType_t::INT8,
                                      true, fused_bias, fuse_relu_, channels_last_, w_in_nhwc,
                                      GetCudnnHandle(context), heur_mode);
    if (fused_status.IsOK()) {
      use_fused_quant = true;
    } else {
      ORT_RETURN_IF_ERROR(UpdateState(x_dims, w_dims, y_dims_for_fe,
                                      pads_vec, stride_vec, dilation_vec,
                                      x_type, w_type, cudnn_frontend::DataType_t::INT32,
                                      false, false, false, channels_last_, w_in_nhwc,
                                      GetCudnnHandle(context), heur_mode));
    }
  } else {
    ORT_RETURN_IF_ERROR(UpdateState(x_dims, w_dims, y_dims_for_fe,
                                    pads_vec, stride_vec, dilation_vec,
                                    x_type, w_type, cudnn_frontend::DataType_t::INT32,
                                    false, false, false, channels_last_, w_in_nhwc,
                                    GetCudnnHandle(context), heur_mode));
  }

  if (use_fused_quant) {
    auto scale_device = GetScratchBuffer<float>(static_cast<size_t>(M), context->GetComputeStream());
    CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(scale_device.get(),
                                         requant_scales.data(),
                                         sizeof(float) * static_cast<size_t>(M),
                                         cudaMemcpyHostToDevice,
                                         Stream(context)));

    const int32_t* bias_data = B != nullptr ? B->Data<int32_t>() : nullptr;
    const int32_t* fused_bias_data_device = bias_data;
    IAllocatorUniquePtr<int32_t> weight_sum;
    IAllocatorUniquePtr<int32_t> bias_adjusted;
    if (x_zero_point_int8 != 0) {
      const int32_t* weight_sum_data = nullptr;
      if (cached_weight_sum_ && cached_weight_sum_size_ == M) {
        weight_sum_data = cached_weight_sum_.get();
      } else {
        const int64_t weight_block_size = W->Shape().Size() / M;
        weight_sum = GetScratchBuffer<int32_t>(static_cast<size_t>(M), context->GetComputeStream());
        ORT_RETURN_IF_ERROR(QLinearConvComputeWeightSum(cuda_stream,
                                                        w_data,
                                                        weight_sum.get(),
                                                        M,
                                                        weight_block_size));
        weight_sum_data = weight_sum.get();
      }

      bias_adjusted = GetScratchBuffer<int32_t>(static_cast<size_t>(M), context->GetComputeStream());
      ORT_RETURN_IF_ERROR(QLinearConvAdjustBias(cuda_stream,
                                                bias_data,
                                                weight_sum_data,
                                                bias_adjusted.get(),
                                                M,
                                                x_zero_point_int8));
      fused_bias_data_device = bias_adjusted.get();
    }

    IAllocatorUniquePtr<float> bias_float_device;
    if (fused_bias) {
      bias_float_device = GetScratchBuffer<float>(static_cast<size_t>(M), context->GetComputeStream());
      ORT_RETURN_IF_ERROR(QLinearConvConvertInt32ToFloat(cuda_stream,
                                                         fused_bias_data_device,
                                                         bias_float_device.get(),
                                                         M));
    }

    cudnn_fe_state_.variant_pack.clear();
    cudnn_fe_state_.variant_pack.insert_or_assign(cudnn_fe_state_.x, const_cast<void*>(static_cast<const void*>(x_data)));
    cudnn_fe_state_.variant_pack.insert_or_assign(cudnn_fe_state_.w, const_cast<void*>(static_cast<const void*>(w_data)));
    if (fused_bias) {
      cudnn_fe_state_.variant_pack.insert_or_assign(cudnn_fe_state_.bias, bias_float_device.get());
    }
    cudnn_fe_state_.variant_pack.insert_or_assign(cudnn_fe_state_.scale, scale_device.get());
    IAllocatorUniquePtr<float> fused_quantized_output;
    if (fused_output_needs_quantize) {
      fused_quantized_output = GetScratchBuffer<float>(static_cast<size_t>(total_elements), context->GetComputeStream());
      cudnn_fe_state_.variant_pack.insert_or_assign(cudnn_fe_state_.y, fused_quantized_output.get());
    } else {
      cudnn_fe_state_.variant_pack.insert_or_assign(cudnn_fe_state_.y, Y->MutableData<int8_t>());
    }

    auto workspace = GetScratchBuffer<void>(cudnn_fe_state_.workspace_bytes, context->GetComputeStream());
    const Status fused_execute_status = CUDNN_FE_CALL(
        cudnn_fe_state_.graph->execute(GetCudnnHandle(context), cudnn_fe_state_.variant_pack, workspace.get()));
    if (fused_execute_status.IsOK()) {
      if (fused_output_needs_quantize) {
        if (y_is_int8) {
          ORT_RETURN_IF_ERROR(QLinearConvQuantizeFloatToInt8(cuda_stream,
                                                             fused_quantized_output.get(),
                                                             Y->MutableData<int8_t>(),
                                                             total_elements,
                                                             static_cast<int8_t>(y_zero_point_value),
                                                             fuse_relu_));
        } else {
          ORT_RETURN_IF_ERROR(QLinearConvQuantizeFloatToUint8(cuda_stream,
                                                              fused_quantized_output.get(),
                                                              Y->MutableData<uint8_t>(),
                                                              total_elements,
                                                              static_cast<uint8_t>(y_zero_point_value),
                                                              fuse_relu_));
        }
      }
      return Status::OK();
    }

    ORT_RETURN_IF_ERROR(UpdateState(x_dims, w_dims, y_dims_for_fe,
                                    pads_vec, stride_vec, dilation_vec,
                                    x_type, w_type, cudnn_frontend::DataType_t::INT32,
                                    false, false, false, channels_last_, w_in_nhwc,
                                    GetCudnnHandle(context), heur_mode));
  }

  auto y_int32 = GetScratchBuffer<int32_t>(static_cast<size_t>(total_elements), context->GetComputeStream());

  cudnn_fe_state_.variant_pack.clear();
  cudnn_fe_state_.variant_pack.insert_or_assign(cudnn_fe_state_.x, const_cast<void*>(static_cast<const void*>(x_data)));
  cudnn_fe_state_.variant_pack.insert_or_assign(cudnn_fe_state_.w, const_cast<void*>(static_cast<const void*>(w_data)));
  cudnn_fe_state_.variant_pack.insert_or_assign(cudnn_fe_state_.y, y_int32.get());

  auto workspace = GetScratchBuffer<void>(cudnn_fe_state_.workspace_bytes, context->GetComputeStream());
  CUDNN_FE_RETURN_IF_ERROR(cudnn_fe_state_.graph->execute(GetCudnnHandle(context), cudnn_fe_state_.variant_pack,
                                                          workspace.get()));

  auto requant_scales_device = GetScratchBuffer<float>(static_cast<size_t>(M), context->GetComputeStream());
  CUDA_RETURN_IF_ERROR(cudaMemcpyAsync(requant_scales_device.get(),
                                       requant_scales.data(),
                                       sizeof(float) * static_cast<size_t>(M),
                                       cudaMemcpyHostToDevice,
                                       Stream(context)));

  const int32_t* bias_data = B != nullptr ? B->Data<int32_t>() : nullptr;
  const int32_t* bias_data_device = bias_data;
  IAllocatorUniquePtr<int32_t> weight_sum;
  IAllocatorUniquePtr<int32_t> bias_adjusted;
  if (x_zero_point_int8 != 0) {
    const int32_t* weight_sum_data = nullptr;
    if (cached_weight_sum_ && cached_weight_sum_size_ == M) {
      weight_sum_data = cached_weight_sum_.get();
    } else {
      const int64_t weight_block_size = W->Shape().Size() / M;
      weight_sum = GetScratchBuffer<int32_t>(static_cast<size_t>(M), context->GetComputeStream());
      ORT_RETURN_IF_ERROR(QLinearConvComputeWeightSum(cuda_stream,
                                                      w_data,
                                                      weight_sum.get(),
                                                      M,
                                                      weight_block_size));
      weight_sum_data = weight_sum.get();
    }

    bias_adjusted = GetScratchBuffer<int32_t>(static_cast<size_t>(M), context->GetComputeStream());
    ORT_RETURN_IF_ERROR(QLinearConvAdjustBias(cuda_stream,
                                              bias_data,
                                              weight_sum_data,
                                              bias_adjusted.get(),
                                              M,
                                              x_zero_point_int8));
    bias_data_device = bias_adjusted.get();
  }
  if (!y_is_int8) {
    ORT_RETURN_IF_ERROR(QLinearConvRequantizeInt32ToUint8(cuda_stream,
                                                          y_int32.get(),
                                                          Y->MutableData<uint8_t>(),
                                                          bias_data_device,
                                                          M,
                                                          spatial_size,
                                                          total_elements,
                                                          requant_scales_device.get(),
                                                          static_cast<uint8_t>(y_zero_point_value),
                                                          fuse_relu_));
  } else {
    ORT_RETURN_IF_ERROR(QLinearConvRequantizeInt32ToInt8(cuda_stream,
                                                         y_int32.get(),
                                                         Y->MutableData<int8_t>(),
                                                         bias_data_device,
                                                         M,
                                                         spatial_size,
                                                         total_elements,
                                                         requant_scales_device.get(),
                                                         static_cast<int8_t>(y_zero_point_value),
                                                         fuse_relu_));
  }

  return Status::OK();
#endif
}

#if !defined(USE_CUDA_MINIMAL) && CUDNN_MAJOR >= 9
#ifdef ENABLE_CUDA_NHWC_OPS
ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv,
    kOnnxDomain,
    10,
    int8_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 4)
        .InputMemoryType(OrtMemTypeCPUInput, 5)
        .InputMemoryType(OrtMemTypeCPUInput, 6)
        .InputMemoryType(OrtMemTypeCPUInput, 7)
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv,
    kOnnxDomain,
    10,
    uint8_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 4)
        .InputMemoryType(OrtMemTypeCPUInput, 5)
        .InputMemoryType(OrtMemTypeCPUInput, 6)
        .InputMemoryType(OrtMemTypeCPUInput, 7)
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int8_t>(),
                                                      DataTypeImpl::GetTensorType<uint8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv,
    kMSInternalNHWCDomain,
    10,
    int8_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 4)
        .InputMemoryType(OrtMemTypeCPUInput, 5)
        .InputMemoryType(OrtMemTypeCPUInput, 6)
        .InputMemoryType(OrtMemTypeCPUInput, 7)
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T2", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<int8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv);

ONNX_OPERATOR_TYPED_KERNEL_EX(
    QLinearConv,
    kMSInternalNHWCDomain,
    10,
    uint8_t,
    kCudaExecutionProvider,
    (*KernelDefBuilder::Create())
        .InputMemoryType(OrtMemTypeCPUInput, 1)
        .InputMemoryType(OrtMemTypeCPUInput, 2)
        .InputMemoryType(OrtMemTypeCPUInput, 4)
        .InputMemoryType(OrtMemTypeCPUInput, 5)
        .InputMemoryType(OrtMemTypeCPUInput, 6)
        .InputMemoryType(OrtMemTypeCPUInput, 7)
        .TypeConstraint("T1", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T2", std::vector<MLDataType>{DataTypeImpl::GetTensorType<int8_t>(),
                                                      DataTypeImpl::GetTensorType<uint8_t>()})
        .TypeConstraint("T3", DataTypeImpl::GetTensorType<uint8_t>())
        .TypeConstraint("T4", DataTypeImpl::GetTensorType<int32_t>()),
    QLinearConv);
#endif
#endif

}  // namespace cuda
}  // namespace onnxruntime
