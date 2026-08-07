#include "rms_norm_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
void rms_norm_(T *out, const T *in, const T *weight, size_t rows, size_t dim, float eps) {
    for (size_t i = 0; i < rows; i++) {
        // 逐行：mean(x^2) 后再 rsqrt
        float sum_sq = 0.0f;
        for (size_t j = 0; j < dim; j++) {
            float x = llaisys::utils::cast<float>(in[i * dim + j]);
            sum_sq += x * x;
        }
        float inv_rms = 1.0f / std::sqrt(sum_sq / static_cast<float>(dim) + eps);

        for (size_t j = 0; j < dim; j++) {
            float x = llaisys::utils::cast<float>(in[i * dim + j]);
            float w = llaisys::utils::cast<float>(weight[j]);
            out[i * dim + j] = llaisys::utils::cast<T>(x * inv_rms * w);
        }
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type,
              size_t rows, size_t dim, float eps) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rms_norm_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                         reinterpret_cast<const float *>(weight), rows, dim, eps);
    case LLAISYS_DTYPE_F16:
        return rms_norm_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                         reinterpret_cast<const llaisys::fp16_t *>(weight), rows, dim, eps);
    case LLAISYS_DTYPE_BF16:
        return rms_norm_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                         reinterpret_cast<const llaisys::bf16_t *>(weight), rows, dim, eps);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
