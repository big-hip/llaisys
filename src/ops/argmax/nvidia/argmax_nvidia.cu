#include "argmax_nvidia.hpp"

#include "../../nvidia/nvidia_common.hpp"

namespace llaisys::ops::nvidia {

__global__ void argmax_kernel(int64_t *max_idx, std::byte *max_val, const std::byte *vals,
                              llaisysDataType_t type, size_t n) {
    constexpr int kArgBlock = 1024;
    __shared__ float s_best[kArgBlock];
    __shared__ int64_t s_idx[kArgBlock];
    const int tid = threadIdx.x;

    // 每个线程在自己的网格步长片段里找最大值；严格大于，保证并列时取第一个（与 CPU 一致）
    float best = -__int_as_float(0x7f800000);
    int64_t best_i = -1;
    for (size_t i = tid; i < n; i += kArgBlock) {
        const float v = ld_float(vals + i * esize(type), type);
        if (v > best) {
            best = v;
            best_i = static_cast<int64_t>(i);
        }
    }
    s_best[tid] = best;
    s_idx[tid] = best_i;
    __syncthreads();

    for (int s = kArgBlock / 2; s > 0; s >>= 1) {
        if (tid < s && s_best[tid + s] > s_best[tid]) {
            s_best[tid] = s_best[tid + s];
            s_idx[tid] = s_idx[tid + s];
        }
        __syncthreads();
    }

    if (tid == 0) {
        *max_idx = s_idx[0];
        st_float(max_val, type, s_best[0]);
    }
}

void argmax(std::byte *max_idx, std::byte *max_val, const std::byte *vals, llaisysDataType_t type, size_t n) {
    argmax_kernel<<<1, 1024>>>(reinterpret_cast<int64_t *>(max_idx), max_val, vals, type, n);
    sync();
}

} // namespace llaisys::ops::nvidia
