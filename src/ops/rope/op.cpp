#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rope_cpu.hpp"

namespace llaisys::ops {
void rope(tensor_t out, tensor_t in, tensor_t pos_ids, float theta) {
    CHECK_SAME_DEVICE(out, in, pos_ids);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype());
    CHECK_ARGUMENT(in->ndim() == 3 && out->ndim() == 3, "rope: in/out must be 3D");
    CHECK_ARGUMENT(out->shape() == in->shape(), "rope: out shape != in shape");
    CHECK_ARGUMENT(in->shape()[2] % 2 == 0, "rope: head dim must be even");
    CHECK_ARGUMENT(pos_ids->dtype() == LLAISYS_DTYPE_I64, "rope: pos_ids must be int64");
    CHECK_ARGUMENT(pos_ids->ndim() == 1 && pos_ids->numel() == in->shape()[0], "rope: invalid pos_ids");
    ASSERT(in->isContiguous() && out->isContiguous() && pos_ids->isContiguous(),
           "rope: all tensors must be contiguous");

    core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rope(out->data(), in->data(), pos_ids->data(), out->dtype(), theta,
                         in->shape()[0], in->shape()[1], in->shape()[2]);
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
