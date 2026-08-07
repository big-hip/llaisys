from typing import Sequence

import json
from ctypes import c_int64, byref
from pathlib import Path

import safetensors.torch

from ..libllaisys import LIB_LLAISYS
from ..libllaisys import DeviceType, DataType
from ..libllaisys.llaisys_types import llaisysDataType_t, llaisysDeviceType_t
from ..libllaisys.qwen2 import LlaisysQwen2Meta


class Qwen2:

    def __init__(self, model_path, device: DeviceType = DeviceType.CPU):
        model_path = Path(model_path)

        with open(model_path / "config.json") as f:
            cfg = json.load(f)

        self.device = device
        self.dtype = DataType.BF16
        self.eos_token = int(cfg["eos_token_id"])

        nlayer = cfg["num_hidden_layers"]
        hs = cfg["hidden_size"]
        nh = cfg["num_attention_heads"]
        dh = cfg.get("head_dim") or (hs // nh)

        meta = LlaisysQwen2Meta()
        meta.dtype = llaisysDataType_t(int(self.dtype))
        meta.nlayer = nlayer
        meta.hs = hs
        meta.nh = nh
        meta.nkvh = cfg["num_key_value_heads"]
        meta.dh = dh
        meta.di = cfg["intermediate_size"]
        meta.maxseq = cfg.get("max_position_embeddings", 2048)
        meta.voc = cfg["vocab_size"]
        meta.epsilon = float(cfg["rms_norm_eps"])
        meta.theta = float(cfg.get("rope_theta", 10000.0))
        meta.end_token = self.eos_token

        self._model = LIB_LLAISYS.llaisysQwen2ModelCreate(
            byref(meta), llaisysDeviceType_t(int(device)), None, 0
        )

        w = LIB_LLAISYS.llaisysQwen2ModelWeights(self._model).contents

        # safetensors 权重名 → llaisys 权重张量
        mapping = {
            "model.embed_tokens.weight": w.in_embed,
            "lm_head.weight": w.out_embed,
            "model.norm.weight": w.out_norm_w,
        }
        for i in range(nlayer):
            p = f"model.layers.{i}."
            mapping[p + "input_layernorm.weight"] = w.attn_norm_w[i]
            mapping[p + "self_attn.q_proj.weight"] = w.attn_q_w[i]
            mapping[p + "self_attn.q_proj.bias"] = w.attn_q_b[i]
            mapping[p + "self_attn.k_proj.weight"] = w.attn_k_w[i]
            mapping[p + "self_attn.k_proj.bias"] = w.attn_k_b[i]
            mapping[p + "self_attn.v_proj.weight"] = w.attn_v_w[i]
            mapping[p + "self_attn.v_proj.bias"] = w.attn_v_b[i]
            mapping[p + "self_attn.o_proj.weight"] = w.attn_o_w[i]
            mapping[p + "post_attention_layernorm.weight"] = w.mlp_norm_w[i]
            mapping[p + "mlp.gate_proj.weight"] = w.mlp_gate_w[i]
            mapping[p + "mlp.up_proj.weight"] = w.mlp_up_w[i]
            mapping[p + "mlp.down_proj.weight"] = w.mlp_down_w[i]

        # 逐权重加载：bf16 字节原样拷进 llaisys 的 BF16 张量
        for file in sorted(model_path.glob("*.safetensors")):
            with safetensors.torch.safe_open(file, framework="pt", device="cpu") as sf:
                for name in sf.keys():
                    if name in mapping:
                        t = sf.get_tensor(name).contiguous()
                        LIB_LLAISYS.tensorLoad(mapping[name], t.data_ptr())
                        del t

    def generate(
        self,
        inputs: Sequence[int],
        max_new_tokens: int = None,
        top_k: int = 1,
        top_p: float = 0.8,
        temperature: float = 0.8,
    ):
        # 测试用 top_k=1 / top_p=1 / temp=1，等价于贪心 argmax。
        # 首次调用把完整输入传进去（prefill 填 KV cache），之后每次传一个 token。
        tokens = [int(t) for t in inputs]
        max_new = max_new_tokens if max_new_tokens else 128
        out = list(tokens)

        for _ in range(max_new):
            n = len(tokens)
            arr = (c_int64 * n)(*tokens)
            nxt = int(LIB_LLAISYS.llaisysQwen2ModelInfer(self._model, arr, n))
            tokens = [nxt]
            out.append(nxt)
            if nxt == self.eos_token:
                break

        return out
