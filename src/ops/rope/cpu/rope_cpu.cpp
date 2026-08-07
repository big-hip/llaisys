#include "rope_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>

namespace llaisys::ops::cpu {

template <typename T>
void rope_(T *out, const T *in, const int64_t *pos_ids, float theta, size_t seq_len, size_t n_heads, size_t head_dim) {
    const size_t half = head_dim / 2;
    for (size_t i = 0; i < seq_len; i++) {
        const float p = static_cast<float>(pos_ids[i]);
        for (size_t j = 0; j < half; j++) {
            // 旋转角：p / theta^(2j/d)；只依赖位置 i 和通道 j，与 head 无关
            const float angle = p / std::pow(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
            const float cos_a = std::cos(angle);
            const float sin_a = std::sin(angle);
            for (size_t h = 0; h < n_heads; h++) {
                const size_t base = (i * n_heads + h) * head_dim;
                const float a = llaisys::utils::cast<float>(in[base + j]);
                const float b = llaisys::utils::cast<float>(in[base + j + half]);
                out[base + j] = llaisys::utils::cast<T>(a * cos_a - b * sin_a);
                out[base + j + half] = llaisys::utils::cast<T>(b * cos_a + a * sin_a);
            }
        }
    }
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, float theta,
          size_t seq_len, size_t n_heads, size_t head_dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return rope_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), theta, seq_len, n_heads, head_dim);
    case LLAISYS_DTYPE_F16:
        return rope_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), theta, seq_len, n_heads, head_dim);
    case LLAISYS_DTYPE_BF16:
        return rope_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(in),
                     reinterpret_cast<const int64_t *>(pos_ids), theta, seq_len, n_heads, head_dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
