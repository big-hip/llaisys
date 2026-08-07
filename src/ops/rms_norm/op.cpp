#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/rms_norm_cpu.hpp"

namespace llaisys::ops {
void rms_norm(tensor_t out, tensor_t in, tensor_t weight, float eps) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(in->ndim() == 2 && out->ndim() == 2, "rms_norm: in/out must be 2D");
    CHECK_ARGUMENT(out->shape() == in->shape(), "rms_norm: out shape != in shape");
    CHECK_ARGUMENT(weight->ndim() == 1 && weight->numel() == in->shape()[1], "rms_norm: invalid weight");
    ASSERT(in->isContiguous() && out->isContiguous() && weight->isContiguous(),
           "rms_norm: all tensors must be contiguous");

    core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::rms_norm(out->data(), in->data(), weight->data(), out->dtype(),
                             in->shape()[0], in->shape()[1], eps);
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
