#include "linear_nvidia.hpp"

#include "../../nvidia/nvidia_common.hpp"

namespace llaisys::ops::nvidia {

__global__ void linear_kernel(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
                              llaisysDataType_t type, size_t m, size_t k, size_t n) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = m * n;
    if (idx >= total) {
        return;
    }
    const size_t i = idx / n, j = idx % n;
    const size_t es = esize(type);

    // Y = X W^T + b；weight 未转置，第 j 行是输出特征 j 的权重
    float acc = 0.0f;
    for (size_t t = 0; t < k; t++) {
        acc += ld_float(in + (i * k + t) * es, type) * ld_float(weight + (j * k + t) * es, type);
    }
    if (bias != nullptr) {
        acc += ld_float(bias + j * es, type);
    }
    st_float(out + idx * es, type, acc);
}

void linear(std::byte *out, const std::byte *in, const std::byte *weight, const std::byte *bias,
            llaisysDataType_t type, size_t m, size_t k, size_t n) {
    const size_t total = m * n;
    const size_t blocks = (total + kBlock - 1) / kBlock;
    linear_kernel<<<blocks, kBlock>>>(out, in, weight, bias, type, m, k, n);
    sync();
}

} // namespace llaisys::ops::nvidia
