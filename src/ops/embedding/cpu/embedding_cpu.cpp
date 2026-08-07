#include "embedding_cpu.hpp"

#include "../../../utils.hpp"

#include <cstring>

namespace llaisys::ops::cpu {

template <typename T>
void embedding_(T *out, const int64_t *index, const T *weight, size_t rows, size_t vocab, size_t dim) {
    for (size_t i = 0; i < rows; i++) {
        const int64_t idx = index[i];
        CHECK_ARGUMENT(idx >= 0 && idx < static_cast<int64_t>(vocab), "embedding: index out of range");
        // 整行搬运：weight 的第 idx 行 → out 的第 i 行
        std::memcpy(out + i * dim, weight + idx * dim, dim * sizeof(T));
    }
}

void embedding(std::byte *out, const std::byte *index, const std::byte *weight, llaisysDataType_t type,
               size_t rows, size_t vocab, size_t dim) {
    switch (type) {
    case LLAISYS_DTYPE_F32:
        return embedding_(reinterpret_cast<float *>(out), reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const float *>(weight), rows, vocab, dim);
    case LLAISYS_DTYPE_F16:
        return embedding_(reinterpret_cast<llaisys::fp16_t *>(out), reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::fp16_t *>(weight), rows, vocab, dim);
    case LLAISYS_DTYPE_BF16:
        return embedding_(reinterpret_cast<llaisys::bf16_t *>(out), reinterpret_cast<const int64_t *>(index),
                          reinterpret_cast<const llaisys::bf16_t *>(weight), rows, vocab, dim);
    default:
        EXCEPTION_UNSUPPORTED_DATATYPE(type);
    }
}
} // namespace llaisys::ops::cpu
