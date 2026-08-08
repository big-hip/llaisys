#include "swiglu_nvidia.hpp"

#include "../../nvidia/nvidia_common.hpp"

namespace llaisys::ops::nvidia {

__global__ void swiglu_kernel(std::byte *out, const std::byte *gate, const std::byte *up,
                              llaisysDataType_t type, size_t numel) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= numel) {
        return;
    }
    const size_t o = i * esize(type);
    const float g = ld_float(gate + o, type);
    const float u = ld_float(up + o, type);
    const float sig = 1.0f / (1.0f + expf(-g)); // sigmoid(gate)
    st_float(out + o, type, u * g * sig);       // SwiGLU: up * gate * sigmoid(gate)
}

void swiglu(std::byte *out, const std::byte *gate, const std::byte *up, llaisysDataType_t type, size_t numel) {
    const size_t blocks = (numel + kBlock - 1) / kBlock;
    swiglu_kernel<<<blocks, kBlock>>>(out, gate, up, type, numel);
    sync();
}

} // namespace llaisys::ops::nvidia
