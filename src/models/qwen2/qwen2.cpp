#include "qwen2.hpp"

#include "../../core/llaisys_core.hpp"

#include "../../ops/add/op.hpp"
#include "../../ops/argmax/op.hpp"
#include "../../ops/embedding/op.hpp"
#include "../../ops/linear/op.hpp"
#include "../../ops/rms_norm/op.hpp"
#include "../../ops/rope/op.hpp"
#include "../../ops/self_attention/op.hpp"
#include "../../ops/swiglu/op.hpp"
#include "../../utils.hpp"

#include <cmath>
#include <cstring>

namespace llaisys::model {

static tensor_t make_tensor(const std::vector<size_t> &shape, llaisysDataType_t dtype,
                            llaisysDeviceType_t device, int device_id) {
    return Tensor::create(shape, dtype, device, device_id);
}

// 用当前设备的 runtime API 拷贝内存：CPU 就是 memcpy，GPU 是 cudaMemcpy。
// 模型里有几处直接对 data() 指针做 std::memcpy，在设备内存上会崩，必须走这里。
static void device_memcpy(void *dst, const void *src, size_t bytes, llaisysMemcpyKind_t kind,
                          llaisysDeviceType_t device, int device_id) {
    core::context().setDevice(device, device_id);
    core::context().runtime().api()->memcpy_sync(dst, src, bytes, kind);
}

Qwen2::Qwen2(const Qwen2Meta &meta, llaisysDeviceType_t device, int device_id)
    : _meta(meta), _device(device), _device_id(device_id) {
    const size_t n = meta.nlayer;
    const size_t hs = meta.hs, dh = meta.dh, nkvh = meta.nkvh, di = meta.di, voc = meta.voc;
    const auto dtype = meta.dtype;

    in_embed = make_tensor({voc, hs}, dtype, device, device_id);
    out_embed = make_tensor({voc, hs}, dtype, device, device_id);
    out_norm_w = make_tensor({hs}, dtype, device, device_id);

    attn_norm_w.resize(n);
    attn_q_w.resize(n);
    attn_q_b.resize(n);
    attn_k_w.resize(n);
    attn_k_b.resize(n);
    attn_v_w.resize(n);
    attn_v_b.resize(n);
    attn_o_w.resize(n);
    mlp_norm_w.resize(n);
    mlp_gate_w.resize(n);
    mlp_up_w.resize(n);
    mlp_down_w.resize(n);

    for (size_t l = 0; l < n; l++) {
        attn_norm_w[l] = make_tensor({hs}, dtype, device, device_id);
        attn_q_w[l] = make_tensor({hs, hs}, dtype, device, device_id);
        attn_q_b[l] = make_tensor({hs}, dtype, device, device_id);
        attn_k_w[l] = make_tensor({nkvh * dh, hs}, dtype, device, device_id);
        attn_k_b[l] = make_tensor({nkvh * dh}, dtype, device, device_id);
        attn_v_w[l] = make_tensor({nkvh * dh, hs}, dtype, device, device_id);
        attn_v_b[l] = make_tensor({nkvh * dh}, dtype, device, device_id);
        attn_o_w[l] = make_tensor({hs, hs}, dtype, device, device_id);
        mlp_norm_w[l] = make_tensor({hs}, dtype, device, device_id);
        mlp_gate_w[l] = make_tensor({di, hs}, dtype, device, device_id);
        mlp_up_w[l] = make_tensor({di, hs}, dtype, device, device_id);
        mlp_down_w[l] = make_tensor({hs, di}, dtype, device, device_id);
    }
}

void Qwen2::_ensure_capacity(size_t needed) {
    if (needed > _meta.maxseq) {
        CHECK_ARGUMENT(false, "qwen2: sequence exceeds max_position_embeddings");
    }
    // 首次调用 _capacity 为空时不能提前返回，必须先把 cache 分配出来
    if (!_k_cache.empty() && _capacity[0] >= needed) {
        return;
    }
    size_t cap = _capacity.empty() ? 256 : _capacity[0];
    while (cap < needed) {
        cap *= 2;
    }
    cap = std::min(cap, _meta.maxseq);

    const size_t per = _meta.nkvh * _meta.dh;
    const size_t es = utils::dsize(_meta.dtype);
    std::vector<tensor_t> new_k(_meta.nlayer), new_v(_meta.nlayer);
    for (size_t l = 0; l < _meta.nlayer; l++) {
        new_k[l] = make_tensor({cap, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_id);
        new_v[l] = make_tensor({cap, _meta.nkvh, _meta.dh}, _meta.dtype, _device, _device_id);
        if (l < _k_cache.size()) {
            // 把旧 cache 的有效前缀搬进新 buffer
            const size_t rows = std::min(_cur_len, _capacity[l]);
            device_memcpy(new_k[l]->data(), _k_cache[l]->data(), rows * per * es,
                          LLAISYS_MEMCPY_D2D, _device, _device_id);
            device_memcpy(new_v[l]->data(), _v_cache[l]->data(), rows * per * es,
                          LLAISYS_MEMCPY_D2D, _device, _device_id);
        }
    }
    _k_cache = std::move(new_k);
    _v_cache = std::move(new_v);
    _capacity.assign(_meta.nlayer, cap);
}

tensor_t Qwen2::_forward(const tensor_t &x_in, size_t ntoken) {
    const auto &m = _meta;
    const size_t hs = m.hs, nh = m.nh, nkvh = m.nkvh, dh = m.dh, di = m.di, voc = m.voc;
    const auto dtype = m.dtype;
    const float scale = 1.0f / std::sqrt(static_cast<float>(dh));

    // 位置 id：当前 token 在全局序列里的位置是 [_cur_len, _cur_len + ntoken)。
    // 先在宿主机填好再拷到设备（设备张量的 data() 不能从 host 直接写）。
    auto pos = make_tensor({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    {
        std::vector<int64_t> pos_h(ntoken);
        for (size_t i = 0; i < ntoken; i++) {
            pos_h[i] = static_cast<int64_t>(_cur_len + i);
        }
        device_memcpy(pos->data(), pos_h.data(), ntoken * sizeof(int64_t),
                      LLAISYS_MEMCPY_H2D, _device, _device_id);
    }

    tensor_t x = x_in;
    for (size_t l = 0; l < m.nlayer; l++) {
        // ---- Attention 块 ----
        tensor_t residual = x;
        auto normed = make_tensor({ntoken, hs}, dtype, _device, _device_id);
        ops::rms_norm(normed, x, attn_norm_w[l], m.epsilon);

        auto q = make_tensor({ntoken, hs}, dtype, _device, _device_id);
        ops::linear(q, normed, attn_q_w[l], attn_q_b[l]);
        auto k = make_tensor({ntoken, nkvh * dh}, dtype, _device, _device_id);
        ops::linear(k, normed, attn_k_w[l], attn_k_b[l]);
        auto v = make_tensor({ntoken, nkvh * dh}, dtype, _device, _device_id);
        ops::linear(v, normed, attn_v_w[l], attn_v_b[l]);

        auto q3 = q->view({ntoken, nh, dh});
        auto k3 = k->view({ntoken, nkvh, dh});
        auto v3 = v->view({ntoken, nkvh, dh});

        auto qr = make_tensor({ntoken, nh, dh}, dtype, _device, _device_id);
        ops::rope(qr, q3, pos, m.theta);
        auto kr = make_tensor({ntoken, nkvh, dh}, dtype, _device, _device_id);
        ops::rope(kr, k3, pos, m.theta);

        // 把新的 k/v 写进 cache 的 [_cur_len, _cur_len+ntoken) 行
        {
            const size_t es = utils::dsize(dtype);
            const size_t per = nkvh * dh;
            device_memcpy(_k_cache[l]->data() + _cur_len * per * es, kr->data(), ntoken * per * es,
                          LLAISYS_MEMCPY_D2D, _device, _device_id);
            device_memcpy(_v_cache[l]->data() + _cur_len * per * es, v3->data(), ntoken * per * es,
                          LLAISYS_MEMCPY_D2D, _device, _device_id);
        }
        auto k_view = _k_cache[l]->slice(0, 0, _cur_len + ntoken);
        auto v_view = _v_cache[l]->slice(0, 0, _cur_len + ntoken);

        auto attn = make_tensor({ntoken, nh, dh}, dtype, _device, _device_id);
        ops::self_attention(attn, qr, k_view, v_view, scale);

        auto o = make_tensor({ntoken, hs}, dtype, _device, _device_id);
        ops::linear(o, attn->view({ntoken, hs}), attn_o_w[l], tensor_t()); // o_proj 无 bias
        auto x_attn = make_tensor({ntoken, hs}, dtype, _device, _device_id);
        ops::add(x_attn, residual, o);
        x = x_attn;

        // ---- MLP 块 ----
        tensor_t residual2 = x;
        auto normed2 = make_tensor({ntoken, hs}, dtype, _device, _device_id);
        ops::rms_norm(normed2, x, mlp_norm_w[l], m.epsilon);

        auto gate = make_tensor({ntoken, di}, dtype, _device, _device_id);
        ops::linear(gate, normed2, mlp_gate_w[l], tensor_t());
        auto up = make_tensor({ntoken, di}, dtype, _device, _device_id);
        ops::linear(up, normed2, mlp_up_w[l], tensor_t());
        auto sw = make_tensor({ntoken, di}, dtype, _device, _device_id);
        ops::swiglu(sw, gate, up);
        auto down = make_tensor({ntoken, hs}, dtype, _device, _device_id);
        ops::linear(down, sw, mlp_down_w[l], tensor_t());

        auto x_mlp = make_tensor({ntoken, hs}, dtype, _device, _device_id);
        ops::add(x_mlp, residual2, down);
        x = x_mlp;
    }

    // 最终归一化 + LM head
    auto normed_final = make_tensor({ntoken, hs}, dtype, _device, _device_id);
    ops::rms_norm(normed_final, x, out_norm_w, m.epsilon);
    auto logits = make_tensor({ntoken, voc}, dtype, _device, _device_id);
    ops::linear(logits, normed_final, out_embed, tensor_t());
    return logits;
}

int64_t Qwen2::infer(const int64_t *token_ids, size_t ntoken) {
    const auto &m = _meta;
    _ensure_capacity(_cur_len + ntoken);

    // embedding：token -> [ntoken, hs]；token_ids 是宿主机数组
    auto idx = make_tensor({ntoken}, LLAISYS_DTYPE_I64, _device, _device_id);
    device_memcpy(idx->data(), token_ids, ntoken * sizeof(int64_t),
                  LLAISYS_MEMCPY_H2D, _device, _device_id);
    auto x = make_tensor({ntoken, m.hs}, m.dtype, _device, _device_id);
    ops::embedding(x, idx, in_embed);

    auto logits = _forward(x, ntoken);

    // 取最后一行 logits 做 argmax
    auto last = logits->slice(0, ntoken - 1, ntoken)->view({m.voc});
    auto max_idx = make_tensor({1}, LLAISYS_DTYPE_I64, _device, _device_id);
    auto max_val = make_tensor({1}, m.dtype, _device, _device_id);
    ops::argmax(max_idx, max_val, last);

    // argmax 结果在设备上，先拷回宿主机再返回
    int64_t next = 0;
    device_memcpy(&next, max_idx->data(), sizeof(int64_t),
                  LLAISYS_MEMCPY_D2H, _device, _device_id);
    _cur_len += ntoken;
    return next;
}

} // namespace llaisys::model
