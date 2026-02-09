// Licensed under the MIT License.

#include "gtest/gtest.h"

#include <algorithm>
#include <limits>
#include <optional>
#include <string>
#include <type_traits>

#include <cuda_runtime.h>
#include <cudnn_version.h>

#include "core/framework/session_options.h"
#include "core/optimizer/graph_transformer_level.h"
#include "core/providers/cuda/cuda_provider_options.h"
#include "test/common/random_generator.h"
#include "test/providers/compare_provider_test_utils.h"
#include "test/util/include/default_providers.h"

namespace onnxruntime {
namespace test {

namespace {

constexpr uint32_t kRandomSeed = 1234;
constexpr int32_t kUint8RandomMin = static_cast<int32_t>(std::numeric_limits<uint8_t>::min());
constexpr int32_t kUint8RandomMax = static_cast<int32_t>(std::numeric_limits<uint8_t>::max());
constexpr int32_t kWeightRandomMin = -63;
constexpr int32_t kWeightRandomMax = 63;
constexpr int32_t kBiasRandomMin = -423;
constexpr int32_t kBiasRandomMax = 423;

using DimVector = std::vector<int64_t>;

int64_t ShapeSize(const DimVector& shape) {
  int64_t size = 1;
  for (auto dim : shape) {
    size *= dim;
  }
  return size;
}

template <typename T>
std::vector<T> GenerateRandomValues(RandomValueGenerator& random,
                                    const DimVector& shape,
                                    int32_t min_value,
                                    int32_t max_value) {
  const auto values = random.Uniform<int32_t>(shape, min_value, max_value + 1);
  std::vector<T> output(values.size());
  std::transform(values.begin(), values.end(), output.begin(),
                 [](int32_t value) { return static_cast<T>(value); });
  return output;
}

std::unique_ptr<IExecutionProvider> CreateCudaExecutionProviderWithHeuristicAlgo();

void ConfigureSessionOptionsForQLinearConvCuda(SessionOptions& session_options) {
  session_options.graph_optimization_level = TransformerLevel::Level3;
}

void CompareQLinearConvWithCpu(CompareOpTester& test) {
  std::vector<int32_t> cpu_output;
  TensorShapeVector output_shape;
  bool output_is_uint8 = false;

  SessionOptions session_options;
  ConfigureSessionOptionsForQLinearConvCuda(session_options);

  test.SetCustomOutputVerifier([&](const std::vector<OrtValue>& fetches, const std::string&) {
    ASSERT_EQ(fetches.size(), 1u);
    const auto& tensor = fetches[0].Get<Tensor>();
    output_is_uint8 = tensor.IsDataType<uint8_t>();
    const auto dims = tensor.Shape().GetDims();
    output_shape.assign(dims.begin(), dims.end());

    const size_t output_size = tensor.Shape().Size();
    cpu_output.resize(output_size);
    if (output_is_uint8) {
      const auto* data = tensor.Data<uint8_t>();
      for (size_t element = 0; element < output_size; ++element) {
        cpu_output[element] = static_cast<int32_t>(data[element]);
      }
    } else {
      const auto* data = tensor.Data<int8_t>();
      for (size_t element = 0; element < output_size; ++element) {
        cpu_output[element] = static_cast<int32_t>(data[element]);
      }
    }
  });

  std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_providers;
  cpu_execution_providers.emplace_back(DefaultCpuExecutionProvider());
  test.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cpu_execution_providers);

  test.SetCustomOutputVerifier([&](const std::vector<OrtValue>& fetches, const std::string&) {
    ASSERT_EQ(fetches.size(), 1u);
    const auto& tensor = fetches[0].Get<Tensor>();
    ASSERT_EQ(output_is_uint8, tensor.IsDataType<uint8_t>());
    const auto dims = tensor.Shape().GetDims();

    TensorShapeVector current_shape;
    current_shape.assign(dims.begin(), dims.end());
    ASSERT_EQ(output_shape, current_shape);

    const size_t output_size = tensor.Shape().Size();
    std::vector<int32_t> cuda_output(output_size);
    if (output_is_uint8) {
      const auto* data = tensor.Data<uint8_t>();
      for (size_t element = 0; element < output_size; ++element) {
        cuda_output[element] = static_cast<int32_t>(data[element]);
      }
    } else {
      const auto* data = tensor.Data<int8_t>();
      for (size_t element = 0; element < output_size; ++element) {
        cuda_output[element] = static_cast<int32_t>(data[element]);
      }
    }
    ASSERT_EQ(cpu_output.size(), cuda_output.size());
    ASSERT_GE(output_shape.size(), 3U);
    ASSERT_LE(output_shape.size(), 5U);

    for (size_t element = 0; element < cpu_output.size(); ++element) {
      EXPECT_EQ(cpu_output[element], cuda_output[element]) << "index: " << element;
    }
  });

  auto cuda_execution_provider = CreateCudaExecutionProviderWithHeuristicAlgo();
  if (!cuda_execution_provider) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }

  std::vector<std::unique_ptr<IExecutionProvider>> cuda_execution_providers;
  cuda_execution_providers.emplace_back(std::move(cuda_execution_provider));
  test.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &cuda_execution_providers);
}

std::unique_ptr<IExecutionProvider> CreateCudaExecutionProviderWithHeuristicAlgo() {
  OrtCUDAProviderOptionsV2 provider_options{};
  provider_options.do_copy_in_default_stream = true;
  provider_options.use_tf32 = false;
  provider_options.prefer_nhwc = true;
  provider_options.cudnn_conv_algo_search = OrtCudnnConvAlgoSearchHeuristic;
  return CudaExecutionProviderWithOptions(&provider_options);
}

bool IsCudaQLinearConvSupported() {
#if !defined(ENABLE_CUDA_NHWC_OPS)
  return false;
#endif
  if (CreateCudaExecutionProviderWithHeuristicAlgo() == nullptr) {
    return false;
  }

#if defined(USE_CUDA_MINIMAL) || CUDNN_MAJOR < 9
  return false;
#else
  cudaDeviceProp prop{};
  if (cudaGetDeviceProperties(&prop, 0) != cudaSuccess) {
    return false;
  }
  return prop.major > 7 || (prop.major == 7 && prop.minor >= 5);
#endif
}

template <typename ActType, typename FilterType>
class QLinearConvOpTester {
 private:
  template <typename T>
  struct QuantizedTensor {
    std::vector<T> data_;
    DimVector shape_;
    std::vector<float> scale_;
    T zero_point_{0};
  };

  RandomValueGenerator random_{kRandomSeed};
  QuantizedTensor<ActType> X_;
  QuantizedTensor<FilterType> W_;
  std::vector<int32_t> B_;
  DimVector pads_;
  DimVector strides_;
  DimVector dilations_;
  std::string auto_pad_;
  int64_t groups_{0};
  float output_scale_{1.0f};
  ActType output_zero_point_{0};
  bool channels_last_{false};
  int opset_version_{10};
  const char* domain_{onnxruntime::kOnnxDomain};
  std::optional<std::string> activation_;

  static size_t ShapeSize(const DimVector& shape) {
    size_t size = 1;
    for (auto dim : shape) {
      size *= static_cast<size_t>(dim);
    }
    return size;
  }

  template <typename T>
  void GenerateRandom(QuantizedTensor<T>& tensor,
                      const DimVector& shape,
                      float scale,
                      T zero_point,
                      int32_t min_value,
                      int32_t max_value) {
    tensor.data_ = GenerateRandomValues<T>(random_, shape, min_value, max_value);
    tensor.shape_ = shape;
    tensor.scale_ = {scale};
    tensor.zero_point_ = zero_point;
  }

  DimVector ComputeOutputShape() const {
    const size_t kernel_rank = W_.shape_.size() - 2;

    DimVector pads = pads_;
    if (pads.empty()) {
      pads.resize(kernel_rank * 2, 0);
    }

    DimVector dilations = dilations_;
    if (dilations.empty()) {
      dilations.resize(kernel_rank, 1);
    }

    DimVector strides = strides_;
    if (strides.empty()) {
      strides.resize(kernel_rank, 1);
    }

    const auto input_dim_at = [&](size_t axis) -> int64_t {
      return channels_last_ ? X_.shape_[axis + 1] : X_.shape_[axis + 2];
    };
    const auto kernel_dim_at = [&](size_t axis) -> int64_t {
      return channels_last_ ? W_.shape_[axis + 1] : W_.shape_[axis + 2];
    };

    DimVector y_shape;
    y_shape.reserve(kernel_rank + 2);
    y_shape.push_back(X_.shape_[0]);
    if (!channels_last_) {
      y_shape.push_back(W_.shape_[0]);
    }

    for (size_t axis = 0; axis < kernel_rank; ++axis) {
      const int64_t input_dim = input_dim_at(axis);
      const int64_t kernel_dim = kernel_dim_at(axis);
      const int64_t pad_before = pads[axis];
      const int64_t pad_after = pads[kernel_rank + axis];
      const int64_t dilation = dilations[axis];
      const int64_t stride = strides[axis];
      const int64_t kernel_extent = dilation * (kernel_dim - 1) + 1;

      int64_t output_dim = 0;
      if (!auto_pad_.empty()) {
        if (auto_pad_ == "SAME_UPPER" || auto_pad_ == "SAME_LOWER") {
          output_dim = (input_dim + stride - 1) / stride;
        } else if (auto_pad_ == "VALID") {
          output_dim = (input_dim - kernel_extent) / stride + 1;
        } else {
          output_dim = (input_dim + pad_before + pad_after - kernel_extent) / stride + 1;
        }
      } else {
        output_dim = (input_dim + pad_before + pad_after - kernel_extent) / stride + 1;
      }

      y_shape.push_back(output_dim);
    }

    if (channels_last_) {
      y_shape.push_back(W_.shape_[0]);
    }

    return y_shape;
  }

  std::string ExpectedFailureMessage() const {
    if (std::is_same<ActType, uint8_t>::value) {
      return "";
    }

    if (!std::is_same<FilterType, int8_t>::value) {
      return "QLinearConv: int8 input requires int8 weights.";
    }
    return "";
  }

  void ConfigureTest(CompareOpTester& test, bool all_input_initializer_except_x) const {
    const DimVector y_shape = ComputeOutputShape();
    std::vector<ActType> y_data(ShapeSize(y_shape), static_cast<ActType>(0));

    test.AddInput<ActType>("x", X_.shape_, X_.data_);
    test.AddInput<float>("x_scale", {}, X_.scale_, all_input_initializer_except_x);
    test.AddInput<ActType>("x_zero_point", {}, {X_.zero_point_}, all_input_initializer_except_x);

    const DimVector w_scale_shape{static_cast<int64_t>(W_.scale_.size())};
    test.AddInput<FilterType>("w", W_.shape_, W_.data_, all_input_initializer_except_x);
    test.AddInput<float>("w_scale", w_scale_shape, W_.scale_, all_input_initializer_except_x);
    test.AddInput<FilterType>("w_zero_point", {}, {W_.zero_point_}, true);

    test.AddInput<float>("y_scale", {}, {output_scale_}, all_input_initializer_except_x);
    test.AddInput<ActType>("y_zero_point", {}, {output_zero_point_}, all_input_initializer_except_x);

    if (!B_.empty()) {
      const DimVector b_shape{static_cast<int64_t>(B_.size())};
      test.AddInput<int32_t>("b", b_shape, B_, all_input_initializer_except_x);
    }

    test.AddOutput<ActType>("y", y_shape, y_data);

    if (!auto_pad_.empty()) {
      test.AddAttribute("auto_pad", auto_pad_);
    } else if (!pads_.empty()) {
      test.AddAttribute("pads", pads_);
    }
    if (!strides_.empty()) {
      test.AddAttribute("strides", strides_);
    }
    if (!dilations_.empty()) {
      test.AddAttribute("dilations", dilations_);
    }
    if (groups_ > 0) {
      test.AddAttribute("group", groups_);
    }
    if (activation_.has_value()) {
      test.AddAttribute("activation", *activation_);
    }
  }

 public:
  QLinearConvOpTester() = default;

  void GenerateRandomInput(const DimVector& shape, float scale, ActType zero_point) {
    GenerateRandom(X_, shape, scale, zero_point,
                   static_cast<int32_t>(std::numeric_limits<int8_t>::min()),
                   static_cast<int32_t>(std::numeric_limits<int8_t>::max()));
  }

  void GenerateRandomWeights(const DimVector& shape, float scale, FilterType zero_point) {
    if (std::is_signed<FilterType>::value) {
      GenerateRandom(W_, shape, scale, zero_point, kWeightRandomMin, kWeightRandomMax);
    } else {
      GenerateRandom(W_, shape, scale, zero_point, kUint8RandomMin, kUint8RandomMax);
    }
  }

  void SetWeightScales(const std::vector<float>& scales) {
    W_.scale_ = scales;
  }

  void GenerateRandomBias() {
    const size_t output_channels = static_cast<size_t>(W_.shape_[0]);
    const DimVector bias_shape{static_cast<int64_t>(output_channels)};
    B_ = random_.Uniform<int32_t>(bias_shape, kBiasRandomMin, kBiasRandomMax + 1);
  }

  void SetPads(const DimVector& pads) {
    pads_ = pads;
  }

  void SetStrides(const DimVector& strides) {
    strides_ = strides;
  }

  void SetDilations(const DimVector& dilations) {
    dilations_ = dilations;
  }

  void SetAutoPad(const std::string& auto_pad) {
    auto_pad_ = auto_pad;
  }

  void SetGroups(int64_t groups) {
    groups_ = groups;
  }

  void SetOutputScaleAndZeroPoint(float output_scale, ActType output_zero_point) {
    output_scale_ = output_scale;
    output_zero_point_ = output_zero_point;
  }

  void SetDomain(const char* domain, int opset_version = 10) {
    domain_ = domain;
    opset_version_ = opset_version;
    channels_last_ = std::string(domain) == kMSInternalNHWCDomain;
  }

  void SetActivation(const std::string& activation) {
    activation_ = activation;
  }

  void Run() {
    for (bool all_input_initializer_except_x : std::initializer_list<bool>{false, true}) {
      Run(all_input_initializer_except_x);
    }
  }

  void Run(bool all_input_initializer_except_x) {
    if (!IsCudaQLinearConvSupported()) {
      GTEST_SKIP() << "CUDA QLinearConv requires cuDNN frontend and SM75+.";
    }
    const auto input_rank = X_.shape_.size();
    const auto weight_rank = W_.shape_.size();
    if (input_rank != weight_rank || (input_rank != 3 && input_rank != 4 && input_rank != 5)) {
      GTEST_SKIP() << "CUDA QLinearConv NHWC currently supports 1D/2D/3D only.";
    }

    CompareOpTester test("QLinearConv", opset_version_, domain_);
    ConfigureTest(test, all_input_initializer_except_x);

    const std::string expected_failure = ExpectedFailureMessage();
    if (!expected_failure.empty()) {
      bool expect_cuda_failure = true;
      const int64_t group = groups_ > 0 ? groups_ : 1;
      if (group > 1 && W_.shape_.size() > 1) {
        const int64_t input_channels_per_group = channels_last_ ? W_.shape_.back() : W_.shape_[1];
        if (input_channels_per_group == 1) {
          expect_cuda_failure = false;
        }
      }

      if (expect_cuda_failure) {
        SessionOptions session_options;
        ConfigureSessionOptionsForQLinearConvCuda(session_options);
        auto cuda_execution_provider = CreateCudaExecutionProviderWithHeuristicAlgo();
        if (!cuda_execution_provider) {
          GTEST_SKIP() << "CUDA execution provider is not available.";
        }
        std::vector<std::unique_ptr<IExecutionProvider>> cuda_execution_providers;
        cuda_execution_providers.emplace_back(std::move(cuda_execution_provider));
        test.Run(session_options, OpTester::ExpectResult::kExpectFailure, expected_failure, {}, nullptr,
                 &cuda_execution_providers);
        return;
      }
    }

    CompareQLinearConvWithCpu(test);
  }

  void RunExpectFailure(const std::string& expected_failure) {
    for (bool all_input_initializer_except_x : std::initializer_list<bool>{false, true}) {
      RunExpectFailure(all_input_initializer_except_x, expected_failure);
    }
  }

  void RunExpectFailure(bool all_input_initializer_except_x, const std::string& expected_failure) {
    if (!IsCudaQLinearConvSupported()) {
      GTEST_SKIP() << "CUDA QLinearConv requires cuDNN frontend and SM75+.";
    }
    const auto input_rank = X_.shape_.size();
    const auto weight_rank = W_.shape_.size();
    if (input_rank != weight_rank || (input_rank != 3 && input_rank != 4 && input_rank != 5)) {
      GTEST_SKIP() << "CUDA QLinearConv NHWC currently supports 1D/2D/3D only.";
    }

    CompareOpTester test("QLinearConv", opset_version_, domain_);
    ConfigureTest(test, all_input_initializer_except_x);

    SessionOptions session_options;
    ConfigureSessionOptionsForQLinearConvCuda(session_options);

    std::vector<std::unique_ptr<IExecutionProvider>> cpu_execution_providers;
    cpu_execution_providers.emplace_back(DefaultCpuExecutionProvider());
    test.Run(session_options, OpTester::ExpectResult::kExpectFailure, expected_failure, {}, nullptr,
             &cpu_execution_providers);

    auto cuda_execution_provider = CreateCudaExecutionProviderWithHeuristicAlgo();
    if (!cuda_execution_provider) {
      GTEST_SKIP() << "CUDA execution provider is not available.";
    }

    std::vector<std::unique_ptr<IExecutionProvider>> cuda_execution_providers;
    cuda_execution_providers.emplace_back(std::move(cuda_execution_provider));
    test.Run(session_options, OpTester::ExpectResult::kExpectFailure, expected_failure, {}, nullptr,
             &cuda_execution_providers);
  }
};

void RunAutoPadParityTest(const DimVector& x_shape,
                          const DimVector& w_shape,
                          const std::string& auto_pad,
                          DimVector dilations = {},
                          DimVector strides = {}) {
  QLinearConvOpTester<uint8_t, int8_t> test;
  test.GenerateRandomInput(x_shape, 0.05f, 128);
  test.GenerateRandomWeights(w_shape, 0.125f, 0);
  test.SetAutoPad(auto_pad);
  if (!strides.empty()) {
    test.SetStrides(strides);
  }
  if (!dilations.empty()) {
    test.SetDilations(dilations);
  }
  test.SetOutputScaleAndZeroPoint(0.55f, 128);
  test.Run();
}

TEST(QLinearConvCudaTest, AutoPadSameUpper_2D) {
  RunAutoPadParityTest({1, 8, 7, 7}, {6, 8, 3, 3}, "SAME_UPPER");
}

TEST(QLinearConvCudaTest, AutoPadSameLower_2D) {
  RunAutoPadParityTest({1, 5, 5, 5}, {5, 5, 2, 2}, "SAME_LOWER", {}, {2, 2});
}

TEST(QLinearConvCudaTest, AutoPadSameUpper_1D) {
  RunAutoPadParityTest({1, 3, 9}, {6, 3, 3}, "SAME_UPPER");
}

TEST(QLinearConvCudaTest, AutoPadSameUpper_3D) {
  RunAutoPadParityTest({1, 2, 5, 5, 5}, {4, 2, 3, 3, 3}, "SAME_UPPER");
}

TEST(QLinearConvCudaTest, AutoPadValid_Dilations_2D) {
  RunAutoPadParityTest({1, 8, 9, 9}, {5, 8, 3, 3}, "VALID", {2, 2});
}

TEST(QLinearConvCudaTest, Nhwc2D_S8S8) {
  QLinearConvOpTester<int8_t, int8_t> test;
  test.GenerateRandomInput({1, 8, 32, 32}, 0.05f, 0);
  test.GenerateRandomWeights({4, 8, 3, 3}, 0.125f, 0);
  test.SetPads({1, 1, 1, 1});
  test.SetOutputScaleAndZeroPoint(0.00625f, 0);
  test.Run();
}

TEST(QLinearConvCudaTest, Nhwc2D_S8S8_CpuDims) {
  for (int64_t input_channels : {1, 2, 3, 5, 8}) {
    QLinearConvOpTester<int8_t, int8_t> test;
    test.GenerateRandomInput({1, input_channels, 7, 7}, 0.05f, 0);
    test.GenerateRandomWeights({2, input_channels, 1, 1}, 0.125f, 0);
    test.SetOutputScaleAndZeroPoint(0.55f, 0);
    test.Run();
  }
}

TEST(QLinearConvCudaTest, Nhwc2D_U8S8) {
  QLinearConvOpTester<uint8_t, int8_t> test;
  test.GenerateRandomInput({1, 8, 32, 32}, 0.05f, 128);
  test.GenerateRandomWeights({4, 8, 3, 3}, 0.125f, 0);
  test.SetPads({1, 1, 1, 1});
  test.SetOutputScaleAndZeroPoint(0.00625f, 128);
  test.Run();
}

TEST(QLinearConvCudaTest, NhwcInternalDomain2D_S8S8_FusedRelu) {
  CompareOpTester test("QLinearConv", 10, kMSInternalNHWCDomain);
  test.AddAttribute("activation", "Relu");

  test.AddInput<int8_t>("x", {1, 1, 1, 2}, {2, -3});
  test.AddInput<float>("x_scale", {}, {1.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {0});

  test.AddInput<int8_t>("w", {2, 2, 1, 1}, {1, 1, -1, -1});
  test.AddInput<float>("w_scale", {}, {1.0f});
  test.AddInput<int8_t>("w_zero_point", {}, {0}, true);

  test.AddInput<float>("y_scale", {}, {1.0f});
  test.AddInput<int8_t>("y_zero_point", {}, {0});
  test.AddOutput<int8_t>("y", {1, 1, 1, 2}, {0, 1});

  SessionOptions session_options;
  ConfigureSessionOptionsForQLinearConvCuda(session_options);

  auto cuda_execution_provider = CreateCudaExecutionProviderWithHeuristicAlgo();
  if (!cuda_execution_provider) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(std::move(cuda_execution_provider));
  test.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QLinearConvCudaTest, NhwcInternalDomain2D_U8S8_FusedRelu_AsymmetricZeroPoint) {
  CompareOpTester test("QLinearConv", 10, kMSInternalNHWCDomain);
  test.AddAttribute("activation", "Relu");

  test.AddInput<uint8_t>("x", {1, 1, 1, 2}, {127, 126});
  test.AddInput<float>("x_scale", {}, {1.0f});
  test.AddInput<uint8_t>("x_zero_point", {}, {128});

  test.AddInput<int8_t>("w", {2, 2, 1, 1}, {1, 1, -1, -1});
  test.AddInput<float>("w_scale", {}, {1.0f});
  test.AddInput<int8_t>("w_zero_point", {}, {0}, true);

  test.AddInput<float>("y_scale", {}, {1.0f});
  test.AddInput<uint8_t>("y_zero_point", {}, {120});
  test.AddOutput<uint8_t>("y", {1, 1, 1, 2}, {120, 123});

  SessionOptions session_options;
  ConfigureSessionOptionsForQLinearConvCuda(session_options);

  auto cuda_execution_provider = CreateCudaExecutionProviderWithHeuristicAlgo();
  if (!cuda_execution_provider) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(std::move(cuda_execution_provider));
  test.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QLinearConvCudaTest, NhwcInternalDomain2D_S8S8_AsymmetricXZeroPoint) {
  CompareOpTester test("QLinearConv", 10, kMSInternalNHWCDomain);

  test.AddInput<int8_t>("x", {1, 1, 1, 2}, {2, -3});
  test.AddInput<float>("x_scale", {}, {1.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {1});

  test.AddInput<int8_t>("w", {2, 2, 1, 1}, {1, 1, -1, -1});
  test.AddInput<float>("w_scale", {}, {1.0f});
  test.AddInput<int8_t>("w_zero_point", {}, {0}, true);

  test.AddInput<float>("y_scale", {}, {1.0f});
  test.AddInput<int8_t>("y_zero_point", {}, {0});
  test.AddOutput<int8_t>("y", {1, 1, 1, 2}, {-3, 3});

  SessionOptions session_options;
  ConfigureSessionOptionsForQLinearConvCuda(session_options);

  auto cuda_execution_provider = CreateCudaExecutionProviderWithHeuristicAlgo();
  if (!cuda_execution_provider) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(std::move(cuda_execution_provider));
  test.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QLinearConvCudaTest, NhwcInternalDomain2D_S8S8_AsymmetricXZeroPoint_Bias_FusedRelu) {
  CompareOpTester test("QLinearConv", 10, kMSInternalNHWCDomain);
  test.AddAttribute("activation", "Relu");

  test.AddInput<int8_t>("x", {1, 1, 1, 2}, {2, -3});
  test.AddInput<float>("x_scale", {}, {1.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {1});

  test.AddInput<int8_t>("w", {2, 2, 1, 1}, {1, 1, -1, -1});
  test.AddInput<float>("w_scale", {}, {1.0f});
  test.AddInput<int8_t>("w_zero_point", {}, {0}, true);

  test.AddInput<float>("y_scale", {}, {1.0f});
  test.AddInput<int8_t>("y_zero_point", {}, {0});
  test.AddInput<int32_t>("b", {2}, {5, -1});
  test.AddOutput<int8_t>("y", {1, 1, 1, 2}, {2, 2});

  SessionOptions session_options;
  ConfigureSessionOptionsForQLinearConvCuda(session_options);

  auto cuda_execution_provider = CreateCudaExecutionProviderWithHeuristicAlgo();
  if (!cuda_execution_provider) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(std::move(cuda_execution_provider));
  test.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QLinearConvCudaTest, NhwcInternalDomain2D_S8S8_NonZeroYZeroPoint) {
  CompareOpTester test("QLinearConv", 10, kMSInternalNHWCDomain);

  test.AddInput<int8_t>("x", {1, 1, 1, 2}, {2, -3});
  test.AddInput<float>("x_scale", {}, {1.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {0});

  test.AddInput<int8_t>("w", {2, 2, 1, 1}, {1, 1, -1, -1});
  test.AddInput<float>("w_scale", {}, {1.0f});
  test.AddInput<int8_t>("w_zero_point", {}, {0}, true);

  test.AddInput<float>("y_scale", {}, {1.0f});
  test.AddInput<int8_t>("y_zero_point", {}, {5});
  test.AddOutput<int8_t>("y", {1, 1, 1, 2}, {4, 6});

  SessionOptions session_options;
  ConfigureSessionOptionsForQLinearConvCuda(session_options);

  auto cuda_execution_provider = CreateCudaExecutionProviderWithHeuristicAlgo();
  if (!cuda_execution_provider) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(std::move(cuda_execution_provider));
  test.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QLinearConvCudaTest, NhwcInternalDomain2D_S8S8_NonZeroYZeroPoint_FusedRelu) {
  CompareOpTester test("QLinearConv", 10, kMSInternalNHWCDomain);
  test.AddAttribute("activation", "Relu");

  test.AddInput<int8_t>("x", {1, 1, 1, 2}, {2, -3});
  test.AddInput<float>("x_scale", {}, {1.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {0});

  test.AddInput<int8_t>("w", {2, 2, 1, 1}, {1, 1, -1, -1});
  test.AddInput<float>("w_scale", {}, {1.0f});
  test.AddInput<int8_t>("w_zero_point", {}, {0}, true);

  test.AddInput<float>("y_scale", {}, {1.0f});
  test.AddInput<int8_t>("y_zero_point", {}, {5});
  test.AddOutput<int8_t>("y", {1, 1, 1, 2}, {5, 6});

  SessionOptions session_options;
  ConfigureSessionOptionsForQLinearConvCuda(session_options);

  auto cuda_execution_provider = CreateCudaExecutionProviderWithHeuristicAlgo();
  if (!cuda_execution_provider) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(std::move(cuda_execution_provider));
  test.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QLinearConvCudaTest, NhwcInternalDomain2D_U8S8_YUint8) {
  CompareOpTester test("QLinearConv", 10, kMSInternalNHWCDomain);

  test.AddInput<uint8_t>("x", {1, 1, 1, 2}, {127, 126});
  test.AddInput<float>("x_scale", {}, {1.0f});
  test.AddInput<uint8_t>("x_zero_point", {}, {128});

  test.AddInput<int8_t>("w", {2, 2, 1, 1}, {1, 1, -1, -1});
  test.AddInput<float>("w_scale", {}, {1.0f});
  test.AddInput<int8_t>("w_zero_point", {}, {0}, true);

  test.AddInput<float>("y_scale", {}, {1.0f});
  test.AddInput<uint8_t>("y_zero_point", {}, {123});
  test.AddOutput<uint8_t>("y", {1, 1, 1, 2}, {120, 126});

  SessionOptions session_options;
  ConfigureSessionOptionsForQLinearConvCuda(session_options);

  auto cuda_execution_provider = CreateCudaExecutionProviderWithHeuristicAlgo();
  if (!cuda_execution_provider) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(std::move(cuda_execution_provider));
  test.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QLinearConvCudaTest, NhwcInternalDomain2D_U8S8_YUint8_FusedRelu) {
  CompareOpTester test("QLinearConv", 10, kMSInternalNHWCDomain);
  test.AddAttribute("activation", "Relu");

  test.AddInput<uint8_t>("x", {1, 1, 1, 2}, {127, 126});
  test.AddInput<float>("x_scale", {}, {1.0f});
  test.AddInput<uint8_t>("x_zero_point", {}, {128});

  test.AddInput<int8_t>("w", {2, 2, 1, 1}, {1, 1, -1, -1});
  test.AddInput<float>("w_scale", {}, {1.0f});
  test.AddInput<int8_t>("w_zero_point", {}, {0}, true);

  test.AddInput<float>("y_scale", {}, {1.0f});
  test.AddInput<uint8_t>("y_zero_point", {}, {123});
  test.AddOutput<uint8_t>("y", {1, 1, 1, 2}, {123, 126});

  SessionOptions session_options;
  ConfigureSessionOptionsForQLinearConvCuda(session_options);

  auto cuda_execution_provider = CreateCudaExecutionProviderWithHeuristicAlgo();
  if (!cuda_execution_provider) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(std::move(cuda_execution_provider));
  test.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QLinearConvCudaTest, NhwcInternalDomain2D_U8S8_YUint8_AsymmetricXZeroPoint) {
  CompareOpTester test("QLinearConv", 10, kMSInternalNHWCDomain);

  test.AddInput<uint8_t>("x", {1, 1, 1, 2}, {130, 125});
  test.AddInput<float>("x_scale", {}, {1.0f});
  test.AddInput<uint8_t>("x_zero_point", {}, {129});

  test.AddInput<int8_t>("w", {2, 2, 1, 1}, {2, 1, -1, 2});
  test.AddInput<float>("w_scale", {}, {1.0f});
  test.AddInput<int8_t>("w_zero_point", {}, {0}, true);

  test.AddInput<float>("y_scale", {}, {1.0f});
  test.AddInput<uint8_t>("y_zero_point", {}, {123});
  test.AddOutput<uint8_t>("y", {1, 1, 1, 2}, {121, 114});

  SessionOptions session_options;
  ConfigureSessionOptionsForQLinearConvCuda(session_options);

  auto cuda_execution_provider = CreateCudaExecutionProviderWithHeuristicAlgo();
  if (!cuda_execution_provider) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(std::move(cuda_execution_provider));
  test.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QLinearConvCudaTest, NhwcInternalDomain2D_U8S8_YUint8_AsymmetricXZeroPoint_Bias_FusedRelu) {
  CompareOpTester test("QLinearConv", 10, kMSInternalNHWCDomain);
  test.AddAttribute("activation", "Relu");

  test.AddInput<uint8_t>("x", {1, 1, 1, 2}, {130, 125});
  test.AddInput<float>("x_scale", {}, {1.0f});
  test.AddInput<uint8_t>("x_zero_point", {}, {129});

  test.AddInput<int8_t>("w", {2, 2, 1, 1}, {2, 1, -1, 2});
  test.AddInput<float>("w_scale", {}, {1.0f});
  test.AddInput<int8_t>("w_zero_point", {}, {0}, true);

  test.AddInput<float>("y_scale", {}, {1.0f});
  test.AddInput<uint8_t>("y_zero_point", {}, {123});
  test.AddInput<int32_t>("b", {2}, {5, -1});
  test.AddOutput<uint8_t>("y", {1, 1, 1, 2}, {126, 123});

  SessionOptions session_options;
  ConfigureSessionOptionsForQLinearConvCuda(session_options);

  auto cuda_execution_provider = CreateCudaExecutionProviderWithHeuristicAlgo();
  if (!cuda_execution_provider) {
    GTEST_SKIP() << "CUDA execution provider is not available.";
  }

  std::vector<std::unique_ptr<IExecutionProvider>> execution_providers;
  execution_providers.emplace_back(std::move(cuda_execution_provider));
  test.Run(session_options, OpTester::ExpectResult::kExpectSuccess, "", {}, nullptr, &execution_providers);
}

TEST(QLinearConvCudaTest, Nhwc2D_U8S8_AsymmetricZeroPoint) {
  constexpr uint8_t kInputZeroPoint = 123;
  constexpr uint8_t kOutputZeroPoint = 117;
  QLinearConvOpTester<uint8_t, int8_t> test;
  test.GenerateRandomInput({1, 8, 32, 32}, 0.05f, kInputZeroPoint);
  test.GenerateRandomWeights({4, 8, 3, 3}, 0.125f, 0);
  test.SetPads({1, 1, 1, 1});
  test.SetOutputScaleAndZeroPoint(0.00625f, kOutputZeroPoint);
  test.Run();
}

TEST(QLinearConvCudaTest, Nhwc3D_U8S8_AsymmetricZeroPoint) {
  constexpr uint8_t kInputZeroPoint = 123;
  constexpr uint8_t kOutputZeroPoint = 117;
  QLinearConvOpTester<uint8_t, int8_t> test;
  test.GenerateRandomInput({1, 8, 8, 32, 32}, 0.05f, kInputZeroPoint);
  test.GenerateRandomWeights({4, 8, 3, 3, 3}, 0.125f, 0);
  test.SetPads({1, 1, 1, 1, 1, 1});
  test.SetOutputScaleAndZeroPoint(0.00625f, kOutputZeroPoint);
  test.Run();
}

TEST(QLinearConvCudaTest, PerChannelScaleInvalidShape) {
  constexpr const char* kExpectedError = "QLinearConv : filter scale shape invalid";
  const DimVector w_shape{3, 2, 3, 3};
  QLinearConvOpTester<int8_t, int8_t> test;
  test.GenerateRandomInput({1, 2, 5, 5}, 0.05f, 0);
  test.GenerateRandomWeights(w_shape, 0.02f, 0);
  test.SetWeightScales(std::vector<float>(static_cast<size_t>(w_shape[0] + 1), 0.02f));
  test.SetPads({1, 1, 1, 1});
  test.SetOutputScaleAndZeroPoint(0.001f, 0);
  test.RunExpectFailure(kExpectedError);
}

TEST(QLinearConvCudaTest, Rounding) {
  CompareOpTester test("QLinearConv", 10);

  const DimVector x_shape{1, 2, 5, 5};
  const DimVector w_shape{3, 2, 3, 3};
  const int64_t kernel_size = w_shape[1] * w_shape[2] * w_shape[3];

  std::vector<int8_t> x_values(static_cast<size_t>(ShapeSize(x_shape)), 1);
  std::vector<int8_t> w_values(static_cast<size_t>(ShapeSize(w_shape)), 0);
  for (int64_t output_channel = 0; output_channel < w_shape[0]; ++output_channel) {
    w_values[static_cast<size_t>(output_channel * kernel_size)] = 1;
  }

  test.AddInput<int8_t>("x", x_shape, x_values);
  test.AddInput<float>("x_scale", {}, {1.0f});
  test.AddInput<int8_t>("x_zero_point", {}, {0});

  test.AddInput<int8_t>("w", w_shape, w_values);
  test.AddInput<float>("w_scale", {}, {1.0f});
  test.AddInput<int8_t>("w_zero_point", {}, {0}, true);

  test.AddInput<float>("y_scale", {}, {2.0f});
  test.AddInput<int8_t>("y_zero_point", {}, {0});

  std::vector<int32_t> bias_values(static_cast<size_t>(w_shape[0]), 2);
  test.AddInput<int32_t>("b", {w_shape[0]}, bias_values);

  const int64_t output_h = x_shape[2] - w_shape[2] + 1;
  const int64_t output_w = x_shape[3] - w_shape[3] + 1;
  std::vector<int8_t> y_values(static_cast<size_t>(x_shape[0] * w_shape[0] * output_h * output_w), 0);
  test.AddOutput<int8_t>("y", {x_shape[0], w_shape[0], output_h, output_w}, y_values);

  CompareQLinearConvWithCpu(test);
}

TEST(QLinearConvCudaTest, HeuristicSuccess) {
  QLinearConvOpTester<int8_t, int8_t> test;
  test.GenerateRandomInput({1, 8, 7, 7}, 0.05f, 0);
  test.GenerateRandomWeights({4, 8, 3, 3}, 0.125f, 0);
  test.SetPads({1, 1, 1, 1});
  test.SetOutputScaleAndZeroPoint(0.00625f, 0);
  test.Run();
}

TEST(QLinearConvCudaTest, Conv2D_S8S8_CpuDims) {
  for (int64_t input_channels : {1, 2, 3, 5, 8}) {
    QLinearConvOpTester<int8_t, int8_t> test;
    test.GenerateRandomInput({1, input_channels, 8, 8}, 1.0f, 0);
    test.GenerateRandomWeights({2, input_channels, 1, 1}, 1.0f, 0);
    test.SetOutputScaleAndZeroPoint(1.0f, 0);
    test.Run();
  }
}

TEST(QLinearConvCudaTest, Conv2D_S8S8_NonZeroZeroPoint) {
  QLinearConvOpTester<int8_t, int8_t> test;
  test.GenerateRandomInput({1, 2, 7, 7}, .05f, -7);
  test.GenerateRandomWeights({4, 2, 3, 3}, .125f, 0);
  test.GenerateRandomBias();
  test.SetPads({1, 1, 1, 1});
  test.SetOutputScaleAndZeroPoint(.55f, 7);
  test.Run();
}

TEST(QLinearConvCudaTest, Conv2D_S8S8_OddDims_1) {
  QLinearConvOpTester<int8_t, int8_t> test;
  test.GenerateRandomInput({1, 5, 13, 17}, .05f, 0);
  test.GenerateRandomWeights({9, 5, 3, 3}, .125f, 0);
  test.GenerateRandomBias();
  test.SetPads({1, 1, 1, 1});
  test.SetOutputScaleAndZeroPoint(.55f, 0);
  test.Run();
}

TEST(QLinearConvCudaTest, Conv2D_S8S8_OddDims_2) {
  QLinearConvOpTester<int8_t, int8_t> test;
  test.GenerateRandomInput({1, 5, 19, 21}, .05f, 0);
  test.GenerateRandomWeights({11, 5, 3, 3}, .125f, 0);
  test.GenerateRandomBias();
  test.SetPads({1, 1, 1, 1});
  test.SetOutputScaleAndZeroPoint(.55f, 0);
  test.Run();
}

}  // namespace
}  // namespace test
}  // namespace onnxruntime
