// Licensed under the MIT License.

#pragma once

#include <memory>
#include <mutex>
#include <unordered_map>
#include <vector>

#include "core/providers/cuda/cuda_kernel.h"
#include "core/providers/cuda/cudnn_common.h"
#include "core/providers/cuda/quantization/qlinearconv_requantize.h"
#include "core/providers/common.h"
#include "core/providers/cpu/nn/conv_attributes.h"

namespace onnxruntime {
namespace cuda {

class QLinearConv final : public CudaKernel {
 public:
  explicit QLinearConv(const OpKernelInfo& info);
  Status PrePack(const Tensor& tensor, int input_idx, AllocatorPtr alloc,
                 bool& is_packed, PrePackedWeights* prepacked_weights) override;
  Status ComputeInternal(OpKernelContext* context) const override;

 private:
  enum InputTensors : int {
    IN_X = 0,
    IN_X_SCALE = 1,
    IN_X_ZERO_POINT = 2,
    IN_W = 3,
    IN_W_SCALE = 4,
    IN_W_ZERO_POINT = 5,
    IN_Y_SCALE = 6,
    IN_Y_ZERO_POINT = 7,
    IN_BIAS = 8
  };

  enum OutputTensors : int {
    OUT_Y = 0
  };

#if !defined(__CUDACC__) && !defined(USE_CUDA_MINIMAL) && CUDNN_MAJOR >= 9
  struct CudnnFeState {
    TensorShapeVector x_dims;
    TensorShapeVector w_dims;
    TensorShapeVector y_dims;
    std::vector<int64_t> pads;
    std::vector<int64_t> strides;
    std::vector<int64_t> dilations;
    cudnn_frontend::DataType_t x_type{cudnn_frontend::DataType_t::NOT_SET};
    cudnn_frontend::DataType_t w_type{cudnn_frontend::DataType_t::NOT_SET};
    cudnn_frontend::DataType_t y_type{cudnn_frontend::DataType_t::NOT_SET};
    bool fused_quant{false};
    bool fused_bias{false};
    bool fused_relu{false};
    size_t workspace_bytes{0};
    bool channels_last{false};
    bool w_in_nhwc{false};
    std::unique_ptr<cudnn_frontend::graph::Graph> graph;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> x;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> w;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> y;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> bias;
    std::shared_ptr<cudnn_frontend::graph::Tensor_attributes> scale;
    std::unordered_map<std::shared_ptr<cudnn_frontend::graph::Tensor_attributes>, void*> variant_pack;
    std::mutex mutex;
  };

  Status UpdateState(const TensorShapeVector& x_dims,
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
                     cudnn_frontend::HeurMode_t heur_mode) const;

  mutable CudnnFeState cudnn_fe_state_;
#endif

  ConvAttributes conv_attrs_;
  std::unique_ptr<Tensor> W_;
  IAllocatorUniquePtr<int32_t> cached_weight_sum_;
  int64_t cached_weight_sum_size_{0};
  bool is_nhwc_domain_{false};
  bool W_already_nhwc_{false};
  bool channels_last_{false};
  bool fuse_relu_{false};
};

}  // namespace cuda
}  // namespace onnxruntime
