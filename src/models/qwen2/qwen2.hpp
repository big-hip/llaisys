#pragma once

#include "../../tensor/tensor.hpp"

#include <cstdint>
#include <vector>

namespace llaisys::model {

struct Qwen2Meta {
    llaisysDataType_t dtype = LLAISYS_DTYPE_BF16;
    size_t nlayer = 0, hs = 0, nh = 0, nkvh = 0, dh = 0, di = 0, maxseq = 0, voc = 0;
    float epsilon = 1e-6f, theta = 10000.0f;
    int64_t end_token = 0;
};

class Qwen2 {
public:
    Qwen2(const Qwen2Meta &meta, llaisysDeviceType_t device, int device_id);
    ~Qwen2() = default;

    // 权重张量：构造函数里按 meta 分配好内存，数据由 python 侧 load 进来
    tensor_t in_embed;   // [voc, hs]
    tensor_t out_embed;  // [voc, hs]  (lm_head)
    tensor_t out_norm_w; // [hs]       (model.norm.weight)
    std::vector<tensor_t> attn_norm_w, attn_q_w, attn_q_b, attn_k_w, attn_k_b;
    std::vector<tensor_t> attn_v_w, attn_v_b, attn_o_w;
    std::vector<tensor_t> mlp_norm_w, mlp_gate_w, mlp_up_w, mlp_down_w;

    const Qwen2Meta &meta() const { return _meta; }

    // 首次调用（prefill）：传完整 prompt；之后（decode）：每次传一个 token。
    // 内部维护 KV cache，返回 argmax 采样的下一个 token id。
    int64_t infer(const int64_t *token_ids, size_t ntoken);

private:
    Qwen2Meta _meta;
    llaisysDeviceType_t _device;
    int _device_id;
    size_t _cur_len = 0; // 已经处理过的 token 数（KV cache 里有多少行）

    std::vector<tensor_t> _k_cache, _v_cache; // 每层一个 [cap, nkvh, dh]
    std::vector<size_t> _capacity;

    void _ensure_capacity(size_t needed);
    tensor_t _forward(const tensor_t &x, size_t ntoken); // x: [ntoken, hs]，返回 logits [ntoken, voc]
};

} // namespace llaisys::model
