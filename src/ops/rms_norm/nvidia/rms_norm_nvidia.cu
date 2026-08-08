#include "rms_norm_nvidia.hpp"

#include "../../nvidia/nvidia_common.hpp"

namespace llaisys::ops::nvidia {

__global__ void rms_norm_kernel(std::byte *out, const std::byte *in, const std::byte *weight,
                                llaisysDataType_t type, size_t rows, size_t dim, float eps) {
    const size_t row = blockIdx.x;
    if (row >= rows) {
        return;
    }
    extern __shared__ float s_sum[];
    const int tid = threadIdx.x;
    const size_t es = esize(type);

    // 每行一个 block：先归约 mean(x^2)，再逐元素缩放
    float local = 0.0f;
    for (size_t j = tid; j < dim; j += blockDim.x) {
        const float x = ld_float(in + (row * dim + j) * es, type);
        local += x * x;
    }
    s_sum[tid] = local;
    __syncthreads();
    for (int s = blockDim.x / 2; s > 0; s >>= 1) {
        if (tid < s) {
            s_sum[tid] += s_sum[tid + s];
        }
        __syncthreads();
    }
    if (tid == 0) {
        s_sum[0] = 1.0f / sqrtf(s_sum[0] / static_cast<float>(dim) + eps);
    }
    __syncthreads();
    const float inv = s_sum[0];

    for (size_t j = tid; j < dim; j += blockDim.x) {
        const size_t o = row * dim + j;
        const float x = ld_float(in + o * es, type);
        const float w = ld_float(weight + j * es, type);
        st_float(out + o * es, type, x * inv * w);
    }
}

void rms_norm(std::byte *out, const std::byte *in, const std::byte *weight, llaisysDataType_t type,
              size_t rows, size_t dim, float eps) {
    rms_norm_kernel<<<rows, kBlock, kBlock * sizeof(float)>>>(out, in, weight, type, rows, dim, eps);
    sync();
}

} // namespace llaisys::ops::nvidia
