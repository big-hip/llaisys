#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/linear_cpu.hpp"
#ifdef ENABLE_NVIDIA_API
#include "nvidia/linear_nvidia.hpp"
#endif

namespace llaisys::ops {
void linear(tensor_t out, tensor_t in, tensor_t weight, tensor_t bias) {
    CHECK_SAME_DEVICE(out, in, weight);
    CHECK_SAME_DTYPE(out->dtype(), in->dtype(), weight->dtype());
    CHECK_ARGUMENT(in->ndim() == 2 && weight->ndim() == 2 && out->ndim() == 2, "linear: in/weight/out must be 2D");
    CHECK_ARGUMENT(out->shape()[0] == in->shape()[0], "linear: out rows != in rows");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[0], "linear: out cols != weight rows");
    CHECK_ARGUMENT(in->shape()[1] == weight->shape()[1], "linear: in cols != weight cols");
    if (bias) {
        CHECK_SAME_DEVICE(bias, out);
        CHECK_SAME_DTYPE(bias->dtype(), out->dtype());
        CHECK_ARGUMENT(bias->ndim() == 1 && bias->numel() == out->shape()[1], "linear: invalid bias");
    }
    ASSERT(in->isContiguous() && weight->isContiguous() && out->isContiguous(),
           "linear: in/weight/out must be contiguous");

    core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::linear(out->data(), in->data(), weight->data(),
                           bias ? bias->data() : nullptr, out->dtype(),
                           out->shape()[0], in->shape()[1], out->shape()[1]);
#ifdef ENABLE_NVIDIA_API
    case LLAISYS_DEVICE_NVIDIA:
        return nvidia::linear(out->data(), in->data(), weight->data(),
                              bias ? bias->data() : nullptr, out->dtype(),
                              out->shape()[0], in->shape()[1], out->shape()[1]);
#endif
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
