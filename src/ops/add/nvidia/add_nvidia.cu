#include "add_nvidia.hpp"

#include "../../nvidia/nvidia_common.hpp"

namespace llaisys::ops::nvidia {

__global__ void add_kernel(const std::byte *a, const std::byte *b, std::byte *c,
                           llaisysDataType_t type, size_t numel) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < numel) {
        const size_t o = i * esize(type);
        st_float(c + o, type, ld_float(a + o, type) + ld_float(b + o, type));
    }
}

void add(std::byte *c, const std::byte *a, const std::byte *b, llaisysDataType_t type, size_t numel) {
    const size_t blocks = (numel + kBlock - 1) / kBlock;
    add_kernel<<<blocks, kBlock>>>(a, b, c, type, numel);
    sync();
}

} // namespace llaisys::ops::nvidia
