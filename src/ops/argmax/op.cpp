#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/argmax_cpu.hpp"

namespace llaisys::ops {
void argmax(tensor_t max_idx, tensor_t max_val, tensor_t vals) {
    CHECK_SAME_DEVICE(max_idx, max_val, vals);
    CHECK_SAME_DTYPE(max_val->dtype(), vals->dtype());
    CHECK_ARGUMENT(max_idx->dtype() == LLAISYS_DTYPE_I64, "argmax: max_idx must be int64");
    CHECK_ARGUMENT(vals->numel() > 0, "argmax: input must not be empty");
    ASSERT(max_idx->isContiguous() && max_val->isContiguous() && vals->isContiguous(),
           "argmax: all tensors must be contiguous");

    core::context().setDevice(vals->deviceType(), vals->deviceId());
    switch (vals->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::argmax(max_idx->data(), max_val->data(), vals->data(), vals->dtype(), vals->numel());
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
