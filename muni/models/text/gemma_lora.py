"""Gemma-3 causal LM with a PEFT LoRA adapter and paired tokenizer."""

from __future__ import annotations

from pathlib import Path
from typing import Iterable

import torch
import torch.nn as nn

from muni.util.misc import print_on_main as print


def _resolve_dtype(name: str) -> torch.dtype:
    mapping = {
        "float32": torch.float32,
        "fp32": torch.float32,
        "bfloat16": torch.bfloat16,
        "bf16": torch.bfloat16,
        "float16": torch.float16,
        "fp16": torch.float16,
    }
    return mapping[name]


class GemmaLoraLM(nn.Module):
    """Keep foundation weights frozen and train only the configured LoRA delta."""

    def __init__(
        self,
        pretrained_path: str,
        tokenizer_path: str | None = None,
        init_from_gemmae_ckpt: str | None = None,
        use_lora: bool = True,
        lora_r: int = 16,
        lora_alpha: int = 32,
        lora_dropout: float = 0.05,
        lora_target_modules: Iterable[str] = ("q_proj", "k_proj", "v_proj", "o_proj"),
        dtype: str = "bfloat16",
        attn_implementation: str = "eager",
        gradient_checkpointing: bool = True,
    ) -> None:
        super().__init__()
        torch_dtype = _resolve_dtype(dtype)

        from transformers import AutoModelForCausalLM, AutoTokenizer

        lm = AutoModelForCausalLM.from_pretrained(
            pretrained_path,
            dtype=torch_dtype,
            attn_implementation=attn_implementation,
        )

        # Match the foundation used by the frozen GemmaE evaluation decoder.
        if init_from_gemmae_ckpt:
            ckpt_path = Path(init_from_gemmae_ckpt)
            payload = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            state = payload.get("model", payload)
            missing, unexpected = lm.load_state_dict(state, strict=False)
            if unexpected:
                raise RuntimeError(
                    f"Unexpected keys while loading {ckpt_path.name}: {len(unexpected)} (first 3: {unexpected[:3]})"
                )
            # HF releases may add non-checkpointed buffers such as rotary caches.
            if missing:
                print(
                    f"[GemmaLoraLM] {len(missing)} missing key(s) vs pretrained "
                    f"({', '.join(missing[:3])}...); fallback to HF weights."
                )

        self.input_embed_dim = int(lm.get_input_embeddings().embedding_dim)

        # Keep this invariant independent of PEFT's default freezing behavior.
        for p in lm.parameters():
            p.requires_grad_(False)

        if use_lora:
            from peft import LoraConfig, TaskType, get_peft_model

            lora_cfg = LoraConfig(
                r=int(lora_r),
                lora_alpha=int(lora_alpha),
                lora_dropout=float(lora_dropout),
                target_modules=list(lora_target_modules),
                bias="none",
                task_type=TaskType.CAUSAL_LM,
            )
            lm = get_peft_model(lm, lora_cfg)

        if gradient_checkpointing:
            try:
                lm.enable_input_require_grads()
            except AttributeError:
                lm.get_input_embeddings().register_forward_hook(lambda _m, _in, out: out.requires_grad_(True))
            lm.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})

        self.lm = lm

        tok_path = tokenizer_path or pretrained_path
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id or 0

    def get_input_embeddings(self) -> nn.Module:
        return self.lm.get_input_embeddings()

    @property
    def pad_token_id(self) -> int:
        return int(self.tokenizer.pad_token_id)

    @property
    def bos_token_id(self) -> int:
        bos = self.tokenizer.bos_token_id
        if bos is None:
            bos = getattr(self.lm.config, "bos_token_id", None) or 0
        return int(bos)

    @property
    def eos_token_id(self) -> int:
        eos = self.tokenizer.eos_token_id
        if eos is None:
            eos = getattr(self.lm.config, "eos_token_id", None) or 0
        return int(eos)

    def forward(self, **kwargs):
        return self.lm(**kwargs)

    def generate(self, **kwargs):
        return self.lm.generate(**kwargs)

    def trainable_named_parameters(self):
        for name, param in self.named_parameters():
            if param.requires_grad:
                yield name, param
