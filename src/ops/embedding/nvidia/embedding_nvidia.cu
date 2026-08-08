#include "embedding_nvidia.hpp"

#include "../../nvidia/nvidia_common.hpp"

namespace llaisys::ops::nvidia {

__global__ void embedding_kernel(std::byte *out, const int64_t *index, const std::byte *weight,
                                 llaisysDataType_t type, size_t rows, size_t vocab, size_t dim) {
    const size_t idx = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const size_t total = rows * dim;
    if (idx >= total) {
        return;
    }
    const size_t i = idx / dim, j = idx % dim;
    const int64_t r = index[i];
    if (r < 0 || r >= static_cast<int64_t>(vocab)) {
        return; // 越界保护（测试数据都合法）
    }
    const size_t es = esize(type);
    st_float(out + idx * es, type, ld_float(weight + (static_cast<size_t>(r) * dim + j) * es, type));
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type,
               size_t rows, size_t vocab, size_t dim) {
    const size_t total = rows * dim;
    const size_t blocks = (total + kBlock - 1) / kBlock;
    embedding_kernel<<<blocks, kBlock>>>(out, reinterpret_cast<const int64_t *>(index), weight, type, rows, vocab, dim);
    sync();
}

} // namespace llaisys::ops::nvidia
