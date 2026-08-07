#include "op.hpp"

#include "../../core/llaisys_core.hpp"
#include "../../utils.hpp"

#include "cpu/embedding_cpu.hpp"

namespace llaisys::ops {
void embedding(tensor_t out, tensor_t index, tensor_t weight) {
    CHECK_SAME_DEVICE(out, index, weight);
    CHECK_SAME_DTYPE(out->dtype(), weight->dtype());
    CHECK_ARGUMENT(index->dtype() == LLAISYS_DTYPE_I64, "embedding: index must be int64");
    CHECK_ARGUMENT(weight->ndim() == 2 && out->ndim() == 2 && index->ndim() == 1, "embedding: unexpected shapes");
    CHECK_ARGUMENT(out->shape()[0] == index->numel(), "embedding: out rows != index length");
    CHECK_ARGUMENT(out->shape()[1] == weight->shape()[1], "embedding: out dim != weight dim");
    ASSERT(out->isContiguous() && index->isContiguous() && weight->isContiguous(),
           "embedding: all tensors must be contiguous");

    core::context().setDevice(out->deviceType(), out->deviceId());
    switch (out->deviceType()) {
    case LLAISYS_DEVICE_CPU:
        return cpu::embedding(out->data(), index->data(), weight->data(), out->dtype(),
                              out->shape()[0], weight->shape()[0], out->shape()[1]);
    default:
        EXCEPTION_UNSUPPORTED_DEVICE;
    }
}
} // namespace llaisys::ops
