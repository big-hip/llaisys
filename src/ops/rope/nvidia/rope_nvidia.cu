#include "rope_nvidia.hpp"

#include "../../nvidia/nvidia_common.hpp"

namespace llaisys::ops::nvidia {

__global__ void rope_kernel(std::byte *out, const std::byte *in, const int64_t *pos_ids,
                            llaisysDataType_t type, float theta, size_t seq_len, size_t n_heads, size_t head_dim) {
    const size_t half = head_dim / 2;
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = seq_len * half;
    if (idx >= total) {
        return;
    }
    const size_t i = idx / half, j = idx % half;
    const size_t es = esize(type);

    // 旋转角只依赖位置 i 和通道 j，与 head 无关，每个线程算一次供所有 head 复用
    const float p = static_cast<float>(pos_ids[i]);
    const float angle = p / powf(theta, 2.0f * static_cast<float>(j) / static_cast<float>(head_dim));
    const float cos_a = cosf(angle);
    const float sin_a = sinf(angle);

    for (size_t h = 0; h < n_heads; h++) {
        const size_t base = (i * n_heads + h) * head_dim;
        const float a = ld_float(in + (base + j) * es, type);
        const float b = ld_float(in + (base + j + half) * es, type);
        st_float(out + (base + j) * es, type, a * cos_a - b * sin_a);
        st_float(out + (base + j + half) * es, type, b * cos_a + a * sin_a);
    }
}

void rope(std::byte *out, const std::byte *in, const std::byte *pos_ids, llaisysDataType_t type, float theta,
          size_t seq_len, size_t n_heads, size_t head_dim) {
    const size_t total = seq_len * (head_dim / 2);
    const size_t blocks = (total + kBlock - 1) / kBlock;
    rope_kernel<<<blocks, kBlock>>>(out, in, reinterpret_cast<const int64_t *>(pos_ids), type, theta,
                                    seq_len, n_heads, head_dim);
    sync();
}

} // namespace llaisys::ops::nvidia
