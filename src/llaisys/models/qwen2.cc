#include "llaisys/models/qwen2.h"

#include "../llaisys_tensor.hpp"
#include "../../models/qwen2/qwen2.hpp"

#include <cstddef>

namespace {

// 把一列 model 权重张量包装成 C 层的 LlaisysTensor* 数组
template <typename Vec>
llaisysTensor_t *wrap_array(const Vec &vec) {
    const size_t n = vec.size();
    llaisysTensor_t *arr = new llaisysTensor_t[n];
    for (size_t i = 0; i < n; i++) {
        arr[i] = new LlaisysTensor{vec[i]};
    }
    return arr;
}

// 删除 C 层数组里的每个包装对象和数组本身
void delete_array(llaisysTensor_t *arr, size_t n) {
    for (size_t i = 0; i < n; i++) {
        delete arr[i];
    }
    delete[] arr;
}

} // namespace

__C {
    struct LlaisysQwen2Model {
        llaisys::model::Qwen2 *model;
        LlaisysQwen2Weights *weights;
        size_t nlayer;
    };

    LlaisysQwen2Model *llaisysQwen2ModelCreate(const LlaisysQwen2Meta *meta, llaisysDeviceType_t device, int *device_ids, int ndevice) {
        llaisys::model::Qwen2Meta m;
        m.dtype = meta->dtype;
        m.nlayer = meta->nlayer;
        m.hs = meta->hs;
        m.nh = meta->nh;
        m.nkvh = meta->nkvh;
        m.dh = meta->dh;
        m.di = meta->di;
        m.maxseq = meta->maxseq;
        m.voc = meta->voc;
        m.epsilon = meta->epsilon;
        m.theta = meta->theta;
        m.end_token = meta->end_token;

        const int dev = (ndevice > 0 && device_ids != nullptr) ? device_ids[0] : 0;
        llaisys::model::Qwen2 *q = new llaisys::model::Qwen2(m, device, dev);

        LlaisysQwen2Weights *w = new LlaisysQwen2Weights{};
        w->in_embed = new LlaisysTensor{q->in_embed};
        w->out_embed = new LlaisysTensor{q->out_embed};
        w->out_norm_w = new LlaisysTensor{q->out_norm_w};
        w->attn_norm_w = wrap_array(q->attn_norm_w);
        w->attn_q_w = wrap_array(q->attn_q_w);
        w->attn_q_b = wrap_array(q->attn_q_b);
        w->attn_k_w = wrap_array(q->attn_k_w);
        w->attn_k_b = wrap_array(q->attn_k_b);
        w->attn_v_w = wrap_array(q->attn_v_w);
        w->attn_v_b = wrap_array(q->attn_v_b);
        w->attn_o_w = wrap_array(q->attn_o_w);
        w->mlp_norm_w = wrap_array(q->mlp_norm_w);
        w->mlp_gate_w = wrap_array(q->mlp_gate_w);
        w->mlp_up_w = wrap_array(q->mlp_up_w);
        w->mlp_down_w = wrap_array(q->mlp_down_w);

        return new LlaisysQwen2Model{q, w, meta->nlayer};
    }

    void llaisysQwen2ModelDestroy(LlaisysQwen2Model *model) {
        if (model == nullptr) {
            return;
        }
        LlaisysQwen2Weights *w = model->weights;
        const size_t n = model->nlayer;
        delete w->in_embed;
        delete w->out_embed;
        delete w->out_norm_w;
        delete_array(w->attn_norm_w, n);
        delete_array(w->attn_q_w, n);
        delete_array(w->attn_q_b, n);
        delete_array(w->attn_k_w, n);
        delete_array(w->attn_k_b, n);
        delete_array(w->attn_v_w, n);
        delete_array(w->attn_v_b, n);
        delete_array(w->attn_o_w, n);
        delete_array(w->mlp_norm_w, n);
        delete_array(w->mlp_gate_w, n);
        delete_array(w->mlp_up_w, n);
        delete_array(w->mlp_down_w, n);
        delete w;
        delete model->model;
        delete model;
    }

    LlaisysQwen2Weights *llaisysQwen2ModelWeights(LlaisysQwen2Model *model) {
        return model->weights;
    }

    int64_t llaisysQwen2ModelInfer(LlaisysQwen2Model *model, int64_t *token_ids, size_t ntoken) {
        return model->model->infer(token_ids, ntoken);
    }
}
