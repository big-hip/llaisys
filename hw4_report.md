# 作业 4 报告：Integrate CUDA into LLAISYS

## 概述

本作业为 LLAISYS 框架接入 CUDA 后端，包含三部分：CUDA Runtime API、8 个算子的 CUDA 内核、Qwen2 模型 CUDA 推理。环境为 NVIDIA RTX A6000 ×2、CUDA 12.4。

## 实现内容

### 1. CUDA Runtime APIs

在 `src/device/nvidia/nvidia_runtime_api.cu` 实现了框架预置的 12 个 `TO_BE_IMPLEMENTED()` 桩：

- 设备：`getDeviceCount` / `setDevice` / `deviceSynchronize`
- 流：`createStream` / `destroyStream` / `streamSynchronize`
- 内存：`mallocDevice` / `freeDevice` / `mallocHost` / `freeHost`
- 拷贝：`memcpySync` / `memcpyAsync`（`llaisysMemcpyKind_t` 映射为 `cudaMemcpyKind`）

构建配置写在 `xmake/nvidia.lua`（仿照 `xmake/cpu.lua`），定义 `llaisys-device-nvidia` 与 `llaisys-ops-nvidia` 两个 target。关键点：用 `set_values("cuda.rdc", false)` 关闭 xmake cuda rule 默认强制加的 `-rdc=true`，否则 `g++` 链接 `.so` 时出现未定义的 `__cudaRegisterLinkedBinary_<hash>` 符号。

### 2. CUDA 算子

按 README 要求在 `src/ops/<op>/nvidia/` 子目录实现 8 个算子的 CUDA 内核（与 CPU 的 `cpu/` 子目录对应），并在各 `op.cpp` 的分发 `switch` 中增加 `case LLAISYS_DEVICE_NVIDIA`：

| 算子 | 内核思路 |
|---|---|
| add / swiglu | 逐元素，线程间网格步长 |
| embedding | 按 token 索引 gather 行，逐元素拷贝 |
| argmax | 单 block 归约，严格大于保证并列取第一个 |
| linear | 每个输出元素一个线程，内层归约 k，float 累加 |
| rms_norm | 每行一个 block，共享内存归约 mean(x²) |
| rope | 每个 (seq, 半通道) 一个线程，算一次 cos/sin 供所有 head 复用 |
| self_attention | 每个 (seq, head) 一个线程，online softmax 单遍扫描（类 flash attention），避免大 scores 数组 |

共享工具头 `src/ops/nvidia/nvidia_common.hpp` 提供 f32/bf16/f16 与 float 的互转（用 CUDA 硬件指令，保证与 torch 在 GPU 上精度一致）。

### 3. 模型 CUDA 支持

`src/models/qwen2/qwen2.cpp` 中有 5 处直接对设备指针做 `std::memcpy` 或 host 直写/直读的地方（KV cache 扩容、新 k/v 写入 cache、pos 位置填充、token 输入、argmax 结果读取），在 GPU 上会崩溃。全部改为通过 `core::context().runtime().api()->memcpy_sync` 按 `H2D/D2D/D2H` 拷贝。

## 复现步骤

### 构建

```bash
xmake f --nv-gpu=y
xmake
xmake install
pip install ./python/
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
```

### 测试

```bash
python test/test_runtime.py --device nvidia

python test/ops/add.py --device nvidia
python test/ops/argmax.py --device nvidia
python test/ops/embedding.py --device nvidia
python test/ops/linear.py --device nvidia
python test/ops/rms_norm.py --device nvidia
python test/ops/rope.py --device nvidia
python test/ops/self_attention.py --device nvidia
python test/ops/swiglu.py --device nvidia

python test/test_infer.py --model <model_dir> --test --device nvidia
```

（CPU 侧默认构建 `xmake` 即可，对应 CI 的 Assignment 0–3。）

## 结果

| 测试 | 结果 |
|---|---|
| `test_runtime.py --device nvidia` | ✅ 发现 2 块 GPU，H2D/D2D/D2H memcpy 全部通过 |
| 8 个算子 `--device nvidia` | ✅ 8/8 通过（f32/f16/bf16 全 dtype） |
| 8 个算子 `--device cpu`（回归） | ✅ 8/8 通过 |
| `test_infer.py --test --device nvidia` | ✅ 通过，约 8s 生成 |
| `test_infer.py --test --device cpu`（回归） | ✅ 通过 |
| CI（默认 CPU 构建）Assignment 0/1/2/3 | ✅ 通过 |

## 支持平台与状态

| 平台 | 状态 |
|---|---|
| **Nvidia**（RTX A6000 ×2，CUDA 12.4） | ✅ 已完成，Runtime API / 算子 / 模型推理全部通过 |
| **第二平台**（Metax 沐曦 / Iluvatar 天数智芯 / Moore Threads 摩尔线程 之一） | ⏳ 留白：本机无对应硬件与 SDK，待申请到算力后实现 |

## 说明

- `test/ops/self_attention.py` 的因果掩码 `temp_mask` 原先生成在 CPU 上，`--device nvidia` 时会因 device 不一致报错（上游代码同样存在），已修复为 `device=query.device`。
