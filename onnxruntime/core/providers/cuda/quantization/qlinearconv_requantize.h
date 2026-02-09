// Licensed under the MIT License.

#pragma once

#include <cstdint>

#include <cuda_runtime_api.h>

#include "core/common/status.h"

namespace onnxruntime {
namespace cuda {

Status QLinearConvRequantizeInt32ToInt8(cudaStream_t stream,
                                        const int32_t* input,
                                        int8_t* output,
                                        const int32_t* bias,
                                        int64_t output_channels,
                                        int64_t spatial_size,
                                        int64_t total_elements,
                                        const float* scales,
                                        int8_t zero_point,
                                        bool fuse_relu = false);

Status QLinearConvRequantizeInt32ToUint8(cudaStream_t stream,
                                         const int32_t* input,
                                         uint8_t* output,
                                         const int32_t* bias,
                                         int64_t output_channels,
                                         int64_t spatial_size,
                                         int64_t total_elements,
                                         const float* scales,
                                         uint8_t zero_point,
                                         bool fuse_relu = false);

Status QLinearConvConvertUint8ToInt8(cudaStream_t stream,
                                     const uint8_t* input,
                                     int8_t* output,
                                     int64_t total_elements,
                                     uint8_t zero_point);

Status QLinearConvConvertInt32ToFloat(cudaStream_t stream,
                                      const int32_t* input,
                                      float* output,
                                      int64_t total_elements);

Status QLinearConvQuantizeFloatToInt8(cudaStream_t stream,
                                      const float* input,
                                      int8_t* output,
                                      int64_t total_elements,
                                      int8_t zero_point,
                                      bool fuse_relu = false);

Status QLinearConvQuantizeFloatToUint8(cudaStream_t stream,
                                       const float* input,
                                       uint8_t* output,
                                       int64_t total_elements,
                                       uint8_t zero_point,
                                       bool fuse_relu = false);

Status QLinearConvComputeWeightSum(cudaStream_t stream,
                                   const int8_t* weights,
                                   int32_t* weight_sum,
                                   int64_t output_channels,
                                   int64_t weight_block_size);

Status QLinearConvAdjustBias(cudaStream_t stream,
                             const int32_t* bias,
                             const int32_t* weight_sum,
                             int32_t* bias_adjusted,
                             int64_t output_channels,
                             int32_t x_zero_point);

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
                               int8_t pad_value);

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
                                int8_t pad_value);

}  // namespace cuda
}  // namespace onnxruntime
