// Licensed under the MIT License.

#include "core/providers/cuda/quantization/qlinearconv_requantize.h"

#include <algorithm>
#include <limits>

#include "core/providers/cuda/cu_inc/common.cuh"

namespace onnxruntime {
namespace cuda {

namespace {

constexpr int kWeightSumBlockSize = 256;

template <typename OutputType>
__global__ void QLinearConvRequantizeKernel(const int32_t* input,
                                            OutputType* output,
                                            const int32_t* bias,
                                            int64_t output_channels,
                                            int64_t spatial_size,
                                            CUDA_LONG total_elements,
                                            const float* scales,
                                            OutputType zero_point,
                                            bool fuse_relu) {
  CUDA_LONG id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < total_elements) {
    (void)spatial_size;
    int64_t channel = static_cast<int64_t>(id) % output_channels;
    int32_t value = input[id];
    if (bias != nullptr) {
      value += bias[channel];
    }
    const float min_value = static_cast<float>(std::numeric_limits<OutputType>::min()) - static_cast<float>(zero_point);
    const float max_value = static_cast<float>(std::numeric_limits<OutputType>::max()) - static_cast<float>(zero_point);
    float scaled = static_cast<float>(value) * scales[channel];
    scaled = max(min_value, min(max_value, scaled));
    int32_t quantized = __float2int_rn(scaled) + static_cast<int32_t>(zero_point);
    if (fuse_relu) {
      quantized = max(quantized, static_cast<int32_t>(zero_point));
    }
    output[id] = static_cast<OutputType>(quantized);
  }
}

__global__ void QLinearConvConvertUint8ToInt8Kernel(const uint8_t* input,
                                                    int8_t* output,
                                                    CUDA_LONG total_elements,
                                                    uint8_t zero_point) {
  CUDA_LONG id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < total_elements) {
    output[id] = static_cast<int8_t>(static_cast<int32_t>(input[id]) - static_cast<int32_t>(zero_point));
  }
}

__global__ void QLinearConvConvertInt32ToFloatKernel(const int32_t* input,
                                                     float* output,
                                                     CUDA_LONG total_elements) {
  CUDA_LONG id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < total_elements) {
    output[id] = static_cast<float>(input[id]);
  }
}

template <typename OutputType>
__global__ void QLinearConvQuantizeFloatToOutputKernel(const float* input,
                                                       OutputType* output,
                                                       CUDA_LONG total_elements,
                                                       OutputType zero_point,
                                                       bool fuse_relu) {
  CUDA_LONG id = static_cast<CUDA_LONG>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (id < total_elements) {
    int32_t quantized = __float2int_rn(input[id]) + static_cast<int32_t>(zero_point);
    quantized = max(static_cast<int32_t>(std::numeric_limits<OutputType>::min()),
                    min(static_cast<int32_t>(std::numeric_limits<OutputType>::max()), quantized));
    if (fuse_relu) {
      quantized = max(quantized, static_cast<int32_t>(zero_point));
    }
    output[id] = static_cast<OutputType>(quantized);
  }
}

__global__ void QLinearConvComputeWeightSumKernel(const int8_t* weights,
                                                  int32_t* weight_sum,
                                                  int64_t weight_block_size) {
  const int64_t channel = static_cast<int64_t>(blockIdx.x);
  const int8_t* weight_base = weights + channel * weight_block_size;
  int32_t sum = 0;
  for (int64_t idx = threadIdx.x; idx < weight_block_size; idx += blockDim.x) {
    sum += static_cast<int32_t>(weight_base[idx]);
  }

  __shared__ int32_t shared[kWeightSumBlockSize];
  shared[threadIdx.x] = sum;
  __syncthreads();

  for (int offset = blockDim.x / 2; offset > 0; offset /= 2) {
    if (threadIdx.x < offset) {
      shared[threadIdx.x] += shared[threadIdx.x + offset];
    }
    __syncthreads();
  }

  if (threadIdx.x == 0) {
    weight_sum[channel] = shared[0];
  }
}

__global__ void QLinearConvAdjustBiasKernel(const int32_t* bias,
                                            const int32_t* weight_sum,
                                            int32_t* bias_adjusted,
                                            int64_t output_channels,
                                            int32_t x_zero_point) {
  const int64_t channel = static_cast<int64_t>(blockIdx.x) * blockDim.x + threadIdx.x;
  if (channel < output_channels) {
    const int32_t bias_value = bias ? bias[channel] : 0;
    bias_adjusted[channel] = bias_value - x_zero_point * weight_sum[channel];
  }
}

__global__ void QLinearConvPadInputNHWCKernel(const int8_t* input,
                                              int8_t* output,
                                              int64_t batch,
                                              int64_t input_h,
                                              int64_t input_w,
                                              int64_t output_h,
                                              int64_t output_w,
                                              int64_t channels,
                                              int64_t pad_top,
                                              int64_t pad_left,
                                              int8_t pad_value,
                                              CUDA_LONG total_elements) {
  CUDA_LONG id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < total_elements) {
    int64_t tmp = static_cast<int64_t>(id);
    const int64_t channel = tmp % channels;
    tmp /= channels;
    const int64_t out_w = tmp % output_w;
    tmp /= output_w;
    const int64_t out_h = tmp % output_h;
    const int64_t batch_index = tmp / output_h;

    const int64_t in_h = out_h - pad_top;
    const int64_t in_w = out_w - pad_left;
    if (in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
      const int64_t input_index = ((batch_index * input_h + in_h) * input_w + in_w) * channels + channel;
      output[id] = input[input_index];
    } else {
      output[id] = pad_value;
    }
  }
}

__global__ void QLinearConvPadInputNDHWCKernel(const int8_t* input,
                                               int8_t* output,
                                               int64_t batch,
                                               int64_t input_d,
                                               int64_t input_h,
                                               int64_t input_w,
                                               int64_t output_d,
                                               int64_t output_h,
                                               int64_t output_w,
                                               int64_t channels,
                                               int64_t pad_front,
                                               int64_t pad_top,
                                               int64_t pad_left,
                                               int8_t pad_value,
                                               CUDA_LONG total_elements) {
  CUDA_LONG id = blockIdx.x * blockDim.x + threadIdx.x;
  if (id < total_elements) {
    int64_t tmp = static_cast<int64_t>(id);
    const int64_t channel = tmp % channels;
    tmp /= channels;
    const int64_t out_w = tmp % output_w;
    tmp /= output_w;
    const int64_t out_h = tmp % output_h;
    tmp /= output_h;
    const int64_t out_d = tmp % output_d;
    const int64_t batch_index = tmp / output_d;

    const int64_t in_d = out_d - pad_front;
    const int64_t in_h = out_h - pad_top;
    const int64_t in_w = out_w - pad_left;
    if (in_d >= 0 && in_d < input_d && in_h >= 0 && in_h < input_h && in_w >= 0 && in_w < input_w) {
      const int64_t input_index = (((batch_index * input_d + in_d) * input_h + in_h) * input_w + in_w) * channels +
                                  channel;
      output[id] = input[input_index];
    } else {
      output[id] = pad_value;
    }
  }
}

}  // namespace

template <typename OutputType>
Status QLinearConvRequantizeInt32ToOutput(cudaStream_t stream,
                                          const int32_t* input,
                                          OutputType* output,
                                          const int32_t* bias,
                                          int64_t output_channels,
                                          int64_t spatial_size,
                                          int64_t total_elements,
                                          const float* scales,
                                          OutputType zero_point,
                                          bool fuse_relu) {
  if (total_elements <= 0) {
    return Status::OK();
  }

  CUDA_LONG count = static_cast<CUDA_LONG>(total_elements);
  int blocks_per_grid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock));
  QLinearConvRequantizeKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      bias,
      output_channels,
      spatial_size,
      count,
      scales,
      zero_point,
      fuse_relu);
  return Status::OK();
}

Status QLinearConvRequantizeInt32ToInt8(cudaStream_t stream,
                                        const int32_t* input,
                                        int8_t* output,
                                        const int32_t* bias,
                                        int64_t output_channels,
                                        int64_t spatial_size,
                                        int64_t total_elements,
                                        const float* scales,
                                        int8_t zero_point,
                                        bool fuse_relu) {
  return QLinearConvRequantizeInt32ToOutput(stream,
                                            input,
                                            output,
                                            bias,
                                            output_channels,
                                            spatial_size,
                                            total_elements,
                                            scales,
                                            zero_point,
                                            fuse_relu);
}

Status QLinearConvRequantizeInt32ToUint8(cudaStream_t stream,
                                         const int32_t* input,
                                         uint8_t* output,
                                         const int32_t* bias,
                                         int64_t output_channels,
                                         int64_t spatial_size,
                                         int64_t total_elements,
                                         const float* scales,
                                         uint8_t zero_point,
                                         bool fuse_relu) {
  return QLinearConvRequantizeInt32ToOutput(stream,
                                            input,
                                            output,
                                            bias,
                                            output_channels,
                                            spatial_size,
                                            total_elements,
                                            scales,
                                            zero_point,
                                            fuse_relu);
}

Status QLinearConvConvertUint8ToInt8(cudaStream_t stream,
                                     const uint8_t* input,
                                     int8_t* output,
                                     int64_t total_elements,
                                     uint8_t zero_point) {
  if (total_elements <= 0) {
    return Status::OK();
  }

  CUDA_LONG count = static_cast<CUDA_LONG>(total_elements);
  int blocks_per_grid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock));
  QLinearConvConvertUint8ToInt8Kernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      count,
      zero_point);
  return Status::OK();
}

Status QLinearConvConvertInt32ToFloat(cudaStream_t stream,
                                      const int32_t* input,
                                      float* output,
                                      int64_t total_elements) {
  if (total_elements <= 0) {
    return Status::OK();
  }

  CUDA_LONG count = static_cast<CUDA_LONG>(total_elements);
  int blocks_per_grid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock));
  QLinearConvConvertInt32ToFloatKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      count);
  return Status::OK();
}

Status QLinearConvQuantizeFloatToInt8(cudaStream_t stream,
                                      const float* input,
                                      int8_t* output,
                                      int64_t total_elements,
                                      int8_t zero_point,
                                      bool fuse_relu) {
  if (total_elements <= 0) {
    return Status::OK();
  }

  CUDA_LONG count = static_cast<CUDA_LONG>(total_elements);
  int blocks_per_grid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock));
  QLinearConvQuantizeFloatToOutputKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      count,
      zero_point,
      fuse_relu);
  return Status::OK();
}

Status QLinearConvQuantizeFloatToUint8(cudaStream_t stream,
                                       const float* input,
                                       uint8_t* output,
                                       int64_t total_elements,
                                       uint8_t zero_point,
                                       bool fuse_relu) {
  if (total_elements <= 0) {
    return Status::OK();
  }

  CUDA_LONG count = static_cast<CUDA_LONG>(total_elements);
  int blocks_per_grid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock));
  QLinearConvQuantizeFloatToOutputKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      count,
      zero_point,
      fuse_relu);
  return Status::OK();
}

Status QLinearConvComputeWeightSum(cudaStream_t stream,
                                   const int8_t* weights,
                                   int32_t* weight_sum,
                                   int64_t output_channels,
                                   int64_t weight_block_size) {
  if (output_channels <= 0 || weight_block_size <= 0) {
    return Status::OK();
  }

  const dim3 blocks(static_cast<uint32_t>(output_channels));
  const dim3 threads(kWeightSumBlockSize);
  QLinearConvComputeWeightSumKernel<<<blocks, threads, 0, stream>>>(weights, weight_sum, weight_block_size);
  return Status::OK();
}

Status QLinearConvAdjustBias(cudaStream_t stream,
                             const int32_t* bias,
                             const int32_t* weight_sum,
                             int32_t* bias_adjusted,
                             int64_t output_channels,
                             int32_t x_zero_point) {
  if (output_channels <= 0) {
    return Status::OK();
  }

  CUDA_LONG count = static_cast<CUDA_LONG>(output_channels);
  int blocks_per_grid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock));
  QLinearConvAdjustBiasKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      bias,
      weight_sum,
      bias_adjusted,
      output_channels,
      x_zero_point);
  return Status::OK();
}

Status QLinearConvPadInputNHWC(cudaStream_t stream,
                               const int8_t* input,
                               int8_t* output,
                               int64_t batch,
                               int64_t input_h,
                               int64_t input_w,
                               int64_t output_h,
                               int64_t output_w,
                               int64_t channels,
                               int64_t pad_top,
                               int64_t pad_left,
                               int8_t pad_value) {
  const int64_t output_size = batch * output_h * output_w * channels;
  if (output_size <= 0) {
    return Status::OK();
  }

  CUDA_LONG count = static_cast<CUDA_LONG>(output_size);
  int blocks_per_grid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock));
  QLinearConvPadInputNHWCKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
      batch,
      input_h,
      input_w,
      output_h,
      output_w,
      channels,
      pad_top,
      pad_left,
      pad_value,
      count);
  return Status::OK();
}

Status QLinearConvPadInputNDHWC(cudaStream_t stream,
                                const int8_t* input,
                                int8_t* output,
                                int64_t batch,
                                int64_t input_d,
                                int64_t input_h,
                                int64_t input_w,
                                int64_t output_d,
                                int64_t output_h,
                                int64_t output_w,
                                int64_t channels,
                                int64_t pad_front,
                                int64_t pad_top,
                                int64_t pad_left,
                                int8_t pad_value) {
  const int64_t output_size = batch * output_d * output_h * output_w * channels;
  if (output_size <= 0) {
    return Status::OK();
  }

  CUDA_LONG count = static_cast<CUDA_LONG>(output_size);
  int blocks_per_grid = static_cast<int>(CeilDiv(count, GridDim::maxThreadsPerBlock));
  QLinearConvPadInputNDHWCKernel<<<blocks_per_grid, GridDim::maxThreadsPerBlock, 0, stream>>>(
      input,
      output,
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
      pad_value,
      count);
  return Status::OK();
}

}  // namespace cuda
}  // namespace onnxruntime
