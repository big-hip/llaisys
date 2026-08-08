#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/self_attention_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/self_attention_nvidia.hpp"
#endif

namespace llaisys::ops {
void self_attention(tensor_t attn_val, tensor_t q, tensor_t k, tensor_t v, float scale) {
    CHECK_SAME_DEVICE(attn_val, q, k, v);
    CHECK_SAME_DTYPE(attn_val->dtype(), q->dtype(), k->dtype(), v->dtype());
    CHECK_ARGUMENT(attn_val->ndim() == 3 && q->ndim() == 3 && k->ndim() == 3 && v->ndim() == 3,
                   "self_attention: all tensors must be 3D");
    CHECK_ARGUMENT(attn_val->shape()[0] == q->shape()[0], "self_attention: seq len mismatch");
    CHECK_ARGUMENT(attn_val->shape()[1] == q->shape()[1], "self_attention: n_head mismatch");
    CHECK_ARGUMENT(q->shape()[2] == k->shape()[2], "self_attention: q dim != k dim");
    CHECK_ARGUMENT(attn_val->shape()[2] == v->shape()[2], "self_attention: out dim != v dim");
    CHECK_ARGUMENT(k->shape()[0] == v->shape()[0] && k->shape()[1] == v->shape()[1],
                   "self_attention: k/v shape mismatch");
    CHECK_ARGUMENT(q->shape()[1] % k->shape()[1] == 0, "self_attention: n_head must be a multiple of n_kv_head");
    ASSERT(attn_val->isContiguous() && q->isContiguous() && k->isContiguous() && v->isContiguous(),
           "self_attention: all tensors must be contiguous");

    core::context().setDevice(attn_val->deviceType(), attn_val->deviceId());
    switch (attn_val->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), scale,
                                   q->shape()[0], q->shape()[1], k->shape()[1], k->shape()[0],
                                   q->shape()[2], v->shape()[2]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::self_attention(attn_val->data(), q->data(), k->data(), v->data(), attn_val->dtype(), scale,
                                      q->shape()[0], q->shape()[1], k->shape()[1], k->shape()[0],
                                      q->shape()[2], v->shape()[2]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
