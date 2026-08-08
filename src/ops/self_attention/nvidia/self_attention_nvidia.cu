#include "self_attention_nvidia.hpp"

#include "../../nvidia/nvidia_common.hpp"

#include <stdexcept>

namespace llaisys::ops::nvidia {

// 每个线程处理一个 (i, head) 的注意力输出。
// 用 online softmax（flash attention 的思路）单遍扫描，维护 running max/sum，
// 避免为整个 total_len 的 scores 分配显存/大数组。
// Qwen2 的 dh=128，测试里 dv 更小；这里给足 256 的上限。
constexpr int kMaxDv = 256;

__global__ void self_attention_kernel(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                                      llaisysDataType_t type, float scale,
                                      size_t seq_len, size_t n_head, size_t n_kv_head, size_t total_len,
                                      size_t d, size_t dv) {
    const size_t id = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = seq_len * n_head;
    if (id >= total) {
        return;
    }
    const size_t i = id / n_head, h = id % n_head;
    const size_t kv_h = h / (n_head / n_kv_head); // GQA：kv 头被 repeat 个查询头共享
    const int64_t kv_offset = static_cast<int64_t>(total_len) - static_cast<int64_t>(seq_len);
    const size_t es = esize(type);

    const std::byte *qi = q + (i * n_head + h) * d * es;

    float m = -__int_as_float(0x7f800000);
    float l = 0.0f;
    float acc[kMaxDv];

    // 因果掩码：s 递增，一旦超过 i + kv_offset 之后全部被掩，直接 break
    for (size_t s = 0; s < total_len; s++) {
        if (static_cast<int64_t>(s) > static_cast<int64_t>(i) + kv_offset) {
            break;
        }
        const std::byte *ks = k + (s * n_kv_head + kv_h) * d * es;
        float dot = 0.0f;
        for (size_t j = 0; j < d; j++) {
            dot += ld_float(qi + j * es, type) * ld_float(ks + j * es, type);
        }
        dot *= scale;

        const float m_new = fmaxf(m, dot);
        const float alpha = expf(m - m_new);
        const float beta = expf(dot - m_new);
        const std::byte *vs = v + (s * n_kv_head + kv_h) * dv * es;
        for (size_t j = 0; j < dv; j++) {
            acc[j] = acc[j] * alpha + beta * ld_float(vs + j * es, type);
        }
        l = l * alpha + beta;
        m = m_new;
    }

    // 正常情况 l > 0；兜底防除零
    const float inv_l = (l > 0.0f) ? (1.0f / l) : 0.0f;
    std::byte *oi = out + (i * n_head + h) * dv * es;
    for (size_t j = 0; j < dv; j++) {
        st_float(oi + j * es, type, acc[j] * inv_l);
    }
}

void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, float scale, size_t seq_len, size_t n_head, size_t n_kv_head,
                    size_t total_len, size_t d, size_t dv) {
    if (dv > kMaxDv) {
        throw std::invalid_argument("self_attention: dv exceeds kernel capacity");
    }
    const size_t total = seq_len * n_head;
    const size_t blocks = (total + kBlock - 1) / kBlock;
    self_attention_kernel<<<blocks, kBlock>>>(out, q, k, v, type, scale, seq_len, n_head, n_kv_head,
                                              total_len, d, dv);
    sync();
}

} // namespace llaisys::ops::nvidia
