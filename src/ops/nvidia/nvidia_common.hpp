#pragma once

#include "llaisys.h"

#include <cuda_bf16.h>
#include <cuda_fp16.h>
#include <cuda_runtime.h>

namespace llaisys::ops::nvidia {

constexpr int kBlock = 256;

// 元素字节数：作业里算子只涉及 f32 / bf16 / f16
__device__ __forceinline__ size_t esize(llaisysDataType_t type) {
    return (type == LLAISYS_DTYPE_F32) ? 4 : 2;
}

// 从原始字节读出一个 float。
// f32 直接读；bf16/f16 用 CUDA 硬件指令转到 float，保证与 torch 在 GPU 上的精度一致。
__device__ __forceinline__ float ld_float(const std::byte *p, llaisysDataType_t type) {
    if (type == LLAISYS_DTYPE_F32) {
        return *reinterpret_cast<const float *>(p);
    }
    const uint16_t bits = *reinterpret_cast<const uint16_t *>(p);
    if (type == LLAISYS_DTYPE_BF16) {
        return __bfloat162float(__ushort_as_bfloat16(bits));
    }
    return __half2float(__ushort_as_half(bits));
}

__device__ __forceinline__ void st_float(std::byte *p, llaisysDataType_t type, float v) {
    if (type == LLAISYS_DTYPE_F32) {
        *reinterpret_cast<float *>(p) = v;
        return;
    }
    uint16_t bits;
    if (type == LLAISYS_DTYPE_BF16) {
        bits = __bfloat16_as_ushort(__float2bfloat16(v));
    } else {
        bits = __half_as_ushort(__float2half(v));
    }
    *reinterpret_cast<uint16_t *>(p) = bits;
}

// 每个算子返回前同步一次，保证结果已就绪（模型里后续用的是 device memcpy）
inline void sync() { cudaDeviceSynchronize(); }

} // namespace llaisys::ops::nvidia
