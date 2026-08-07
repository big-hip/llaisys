#pragma once

#include "llaisys.h"

#include <cstddef>

namespace llaisys::ops::cpu {
void self_attention(std::byte *out, const std::byte *q, const std::byte *k, const std::byte *v,
                    llaisysDataType_t type, float scale, size_t seq_len, size_t n_head, size_t n_kv_head,
                    size_t total_len, size_t d, size_t dv);
}
