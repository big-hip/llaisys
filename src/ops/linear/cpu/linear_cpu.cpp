#include "linear_cpu.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

template <typename T>
void linear_(T *out, const T *in, const T *weight, const T *bias, size_t m, size_t k, size_t n) {
    for (size_t i = 0; i < m; i++) {
        for (size_t j = 0; j < n; j++) {
            // Y = X W^T + b；weight 未转置，第 j 行是输出特征 j 的权重
            float acc = 0.0f;
            const T *wrow = weight + j * k;
            for (size_t t = 0; t < k; t++) {
                acc += llaisys::utils::cast<float>(in[i * k + t]) * llaisys::utils::cast<float>(wrow[t]);
            }
            if (bias != nullptr) {
                acc += llaisys::utils::cast<float>(bias[j]);
            }
            out[i * n + j] = llaisys::utils::cast<T>(acc);
        }
    }
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t m, size_t k, size_t n) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return linear_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                       reinterpret_cast<const float *>(weight), reinterpret_cast<const float *>(bias), m, k, n);
    case LLAISYS_DTYPE_F16:
        return linear_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                       reinterpret_cast<const llaisys::fp16_t *>(weight), reinterpret_cast<const llaisys::fp16_t *>(bias), m, k, n);
    case LLAISYS_DTYPE_BF16:
        return linear_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                       reinterpret_cast<const llaisys::bf16_t *>(weight), reinterpret_cast<const llaisys::bf16_t *>(bias), m, k, n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
