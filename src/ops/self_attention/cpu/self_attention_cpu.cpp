#include "self_attention_cpu.hpp"

#include "../../../utils.hpp"

#include <cmath>
#include <limits>
#include <vector>

namespace llaisys::ops::cpu {

template <typename T>
void self_attention_(T *out, const T *q, const T *k, const T *v, float scale,
                     size_t seq_len, size_t n_head, size_t n_kv_head, size_t total_len, size_t d, size_t dv) {
    const size_t repeat = n_head / n_kv_head; // GQA：每个 kv 头被 repeat 个查询头共享
    const int64_t kv_offset = static_cast<int64_t>(total_len) - static_cast<int64_t>(seq_len);

    std::vector<float> scores(total_len);
    for (size_t i = 0; i < seq_len; i++) {
        for (size_t h = 0; h < n_head; h++) {
            const size_t kv_h = h / repeat;
            const T *qi = q + (i * n_head + h) * d;

            // 1) 打分 QK^T * scale，带因果掩码：只允许 s <= i + kv_offset
            float max_s = -std::numeric_limits<float>::infinity();
            for (size_t s = 0; s < total_len; s++) {
                if (static_cast<int64_t>(s) > static_cast<int64_t>(i) + kv_offset) {
                    scores[s] = -std::numeric_limits<float>::infinity();
                    continue;
                }
                const T *ks = k + (s * n_kv_head + kv_h) * d;
                float dot = 0.0f;
                for (size_t j = 0; j < d; j++) {
                    dot += llaisys::utils::cast<float>(qi[j]) * llaisys::utils::cast<float>(ks[j]);
                }
                scores[s] = dot * scale;
                if (scores[s] > max_s) {
                    max_s = scores[s];
                }
            }
            if (std::isinf(max_s)) {
                max_s = 0.0f; // 整行被掩码（正常情况不会发生），兜底
            }

            // 2) softmax：减 max 防溢出；掩码处 exp(-inf) = 0
            float sum = 0.0f;
            for (size_t s = 0; s < total_len; s++) {
                scores[s] = std::exp(scores[s] - max_s);
                sum += scores[s];
            }

            // 3) 加权求和 V
            T *oi = out + (i * n_head + h) * dv;
            for (size_t j = 0; j < dv; j++) {
                float acc = 0.0f;
                for (size_t s = 0; s < total_len; s++) {
                    const T *vs = v + (s * n_kv_head + kv_h) * dv;
                    acc += (scores[s] / sum) * llaisys::utils::cast<float>(vs[j]);
                }
                oi[j] = llaisys::utils::cast<T>(acc);
            }
        }
    }
}

void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, float scale, size_t seq_len, size_t n_head, size_t n_kv_head,
                    size_t total_len, size_t d, size_t dv) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return self_attention_(reinterpret_cast<float *>(out), reinterpret_cast<const float *>(q),
                               reinterpret_cast<const float *>(k), reinterpret_cast<const float *>(v),
                               scale, seq_len, n_head, n_kv_head, total_len, d, dv);
    case LLAISYS_DTYPE_F16:
        return self_attention_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const llaisys::fp16_t *>(q),
                               reinterpret_cast<const llaisys::fp16_t *>(k), reinterpret_cast<const llaisys::fp16_t *>(v),
                               scale, seq_len, n_head, n_kv_head, total_len, d, dv);
    case LLAISYS_DTYPE_BF16:
        return self_attention_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const llaisys::bf16_t *>(q),
                               reinterpret_cast<const llaisys::bf16_t *>(k), reinterpret_cast<const llaisys::bf16_t *>(v),
                               scale, seq_len, n_head, n_kv_head, total_len, d, dv);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
