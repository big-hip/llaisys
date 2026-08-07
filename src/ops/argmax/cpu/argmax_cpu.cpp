#include "argmax_cpu.hpp"

#include "../../../utils.hpp"

namespace llaisys::ops::cpu {

template <typename T>
void argmax_(int64_t *max_idx, T *max_val, const T *vals, size_t n) {
    // 在 float 域比较（f16/bf16 需要提升精度），记录第一个最大值的位置
    size_t best_idx = 0;
    float best = llaisys::utils::cast<float>(vals[0]);
    for (size_t i = 1; i < n; i++) {
        float v = llaisys::utils::cast<float>(vals[i]);
        if (v > best) {
            best = v;
            best_idx = i;
        }
    }
    max_idx[0] = static_cast<int64_t>(best_idx);
    max_val[0] = vals[best_idx];
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t n) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<float *>(max_val),
                       reinterpret_cast<const float *>(vals), n);
    case LLAISYS_DTYPE_F16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::fp16_t *>(max_val),
                       reinterpret_cast<const llaisys::fp16_t *>(vals), n);
    case LLAISYS_DTYPE_BF16:
        return argmax_(reinterpret_cast<int64_t *>(max_idx), reinterpret_cast<llaisys::bf16_t *>(max_val),
                       reinterpret_cast<const llaisys::bf16_t *>(vals), n);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
