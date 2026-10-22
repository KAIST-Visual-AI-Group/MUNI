"""HuggingFace causal-LM decoder conditioned by a learned prefix."""

from __future__ import annotations

import random
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from .core import BaseDecoder
from .ops import scale_grad


_GENERATE_KWARGS = {
    "bad_words_ids",
    "diversity_penalty",
    "early_stopping",
    "encoder_repetition_penalty",
    "epsilon_cutoff",
    "eta_cutoff",
    "forced_bos_token_id",
    "forced_eos_token_id",
    "length_penalty",
    "logits_processor",
    "min_new_tokens",
    "no_repeat_ngram_size",
    "num_beam_groups",
    "num_beams",
    "penalty_alpha",
    "repetition_penalty",
    "stopping_criteria",
}


class GemmaARTextDecoderAdapter(BaseDecoder):
    """Decode raw text with a LoRA LM and a latent-to-prefix projector."""

    def __init__(
        self,
        lm: nn.Module,
        *,
        projector: nn.Module,
        encoder_grad_scale: float = 1.0,
        max_new_tokens: int = 64,
        temperature: float = 0.0,
        top_k: int = 0,
        top_p: float = 1.0,
        label_smoothing: float = 0.0,
        align_weight: float = 0.0,
        align_loss_type: str = "mse",
        ce_loss_prob: float = 1.0,
        ce_loss_requires_modality: str | None = None,
        ce_loss_batch_ratio: float = 1.0,
        ce_loss_weight: float = 1.0,
        num_beams: int | None = None,
        length_penalty: float | None = None,
        early_stopping: bool | None = None,
        no_repeat_ngram_size: int | None = None,
        repetition_penalty: float | None = None,
        min_new_tokens: int | None = None,
    ) -> None:
        super().__init__()
        # build_model passes the configured decoder backbone positionally and
        # resolves projector_module as this keyword argument.
        self.projector = projector
        self.lm = lm
        self.encoder_grad_scale = float(encoder_grad_scale)
        self.max_new_tokens = int(max_new_tokens)
        self.temperature = float(temperature)
        self.top_k = int(top_k)
        self.top_p = float(top_p)
        self.label_smoothing = float(label_smoothing)
        self.align_weight = float(align_weight)
        if align_loss_type not in ("mse", "cosine"):
            raise ValueError(
                f"align_loss_type must be 'mse' or 'cosine'; got {align_loss_type!r}."
            )
        self.align_loss_type = align_loss_type
        if (
            self.align_weight > 0.0
            and getattr(self.projector, "align_head", None) is None
        ):
            raise ValueError(
                "align_weight > 0 requires an alignment-capable projector."
            )
        if not 0.0 <= ce_loss_prob <= 1.0:
            raise ValueError(f"ce_loss_prob must be in [0, 1]; got {ce_loss_prob!r}.")
        self.ce_loss_prob = float(ce_loss_prob)
        self.ce_loss_requires_modality = (
            None
            if ce_loss_requires_modality in (None, "")
            else str(ce_loss_requires_modality)
        )
        if not 0.0 < ce_loss_batch_ratio <= 1.0:
            raise ValueError(
                f"ce_loss_batch_ratio must be in (0, 1]; got {ce_loss_batch_ratio!r}."
            )
        self.ce_loss_batch_ratio = float(ce_loss_batch_ratio)
        if ce_loss_weight < 0.0:
            raise ValueError(f"ce_loss_weight must be >= 0; got {ce_loss_weight!r}.")
        self.ce_loss_weight = float(ce_loss_weight)
        self._ce_step = 0
        self._default_gen_kwargs = {
            key: cast(value)
            for key, value, cast in (
                ("num_beams", num_beams, int),
                ("length_penalty", length_penalty, float),
                ("early_stopping", early_stopping, bool),
                ("no_repeat_ngram_size", no_repeat_ngram_size, int),
                ("repetition_penalty", repetition_penalty, float),
                ("min_new_tokens", min_new_tokens, int),
            )
            if value is not None
        }

    @property
    def pad_token_id(self) -> int:
        return self.lm.pad_token_id

    def _build_prefix(
        self, z_cond: torch.Tensor, *, train: bool
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        z = scale_grad(z_cond, self.encoder_grad_scale) if train else z_cond
        prefix, align_feat = self.projector(z)
        target_dtype = self.lm.get_input_embeddings().weight.dtype
        return prefix.to(dtype=target_dtype), align_feat

    def _should_compute_ce(self, available_modalities) -> bool:
        """Apply the deterministic, rank-synchronous stochastic CE gate."""
        if self.ce_loss_prob <= 0.0:
            return False
        required = self.ce_loss_requires_modality
        if required is not None and (
            available_modalities is None or required not in available_modalities
        ):
            return False
        if self.ce_loss_prob >= 1.0:
            return True
        rng = random.Random(self._ce_step)
        self._ce_step += 1
        return rng.random() < self.ce_loss_prob

    def compute_loss(
        self,
        target: Any,
        z_cond: torch.Tensor,
        **kwargs,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        tokens = target.get("tokens") if isinstance(target, Mapping) else target
        if tokens is not None:
            if tokens.dtype != torch.long:
                tokens = tokens.long()

        prefix, align_feat = self._build_prefix(z_cond, train=True)
        compute_ce = tokens is not None and self._should_compute_ce(
            kwargs.get("available_modalities")
        )
        loss = prefix.new_zeros(())

        if compute_ce:
            batch_size = max(1, int(tokens.shape[0] * self.ce_loss_batch_ratio))
            prefix = prefix[:batch_size]
            tokens = tokens[:batch_size]
            prefix_len = prefix.shape[1]

            token_embeddings = self.lm.get_input_embeddings()(tokens).to(
                dtype=prefix.dtype
            )
            inputs_embeds = torch.cat((prefix, token_embeddings), dim=1)
            prefix_attention = tokens.new_ones((batch_size, prefix_len))
            token_attention = tokens.ne(self.pad_token_id).long()
            attention_mask = torch.cat((prefix_attention, token_attention), dim=1)
            prefix_labels = tokens.new_full((batch_size, prefix_len), -100)
            labels = torch.cat(
                (prefix_labels, tokens.masked_fill(tokens.eq(self.pad_token_id), -100)),
                dim=1,
            )

            outputs = self.lm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=None if self.label_smoothing > 0.0 else labels,
                use_cache=False,
            )
            if self.label_smoothing > 0.0:
                shift_logits = rearrange(outputs.logits[:, :-1], "b t v -> (b t) v")
                shift_labels = rearrange(labels[:, 1:], "b t -> (b t)")
                ce_loss = F.cross_entropy(
                    shift_logits,
                    shift_labels,
                    ignore_index=-100,
                    label_smoothing=self.label_smoothing,
                )
            else:
                ce_loss = outputs.loss
            loss = self.ce_loss_weight * ce_loss

        if (
            self.align_weight > 0.0
            and isinstance(target, Mapping)
            and "feature" in target
        ):
            target_feat = target["feature"].to(
                dtype=align_feat.dtype, device=align_feat.device
            )
            if self.align_loss_type == "mse":
                align_loss = F.mse_loss(align_feat, target_feat)
            else:
                align_loss = (
                    1.0 - F.cosine_similarity(align_feat, target_feat, dim=-1).mean()
                )
            loss = loss + self.align_weight * align_loss

        return loss, {}

    @torch.no_grad()
    def sample(
        self,
        z_cond: torch.Tensor,
        temperature: float | None = None,
        top_k: int | None = None,
        top_p: float | None = None,
        max_new_tokens: int | None = None,
        **kwargs,
    ) -> list[str]:
        temperature = self.temperature if temperature is None else float(temperature)
        top_k = self.top_k if top_k is None else int(top_k)
        top_p = self.top_p if top_p is None else float(top_p)
        max_new_tokens = (
            self.max_new_tokens if max_new_tokens is None else int(max_new_tokens)
        )

        prefix, _ = self._build_prefix(z_cond, train=False)
        bos_ids = torch.full(
            (z_cond.shape[0], 1), self.lm.bos_token_id,
            device=z_cond.device, dtype=torch.long,
        )
        bos_emb = self.lm.get_input_embeddings()(bos_ids).to(prefix.dtype)
        inputs_embeds = torch.cat((prefix, bos_emb), dim=1)
        attention_mask = bos_ids.new_ones(inputs_embeds.shape[:2])

        extra = dict(self._default_gen_kwargs)
        extra.update(
            (key, value) for key, value in kwargs.items()
            if key in _GENERATE_KWARGS and value is not None
        )
        penalty_alpha = extra.get("penalty_alpha")
        do_sample = (
            int(extra.get("num_beams", 1)) <= 1
            and penalty_alpha is None
            and (temperature > 0.0 or top_k > 0 or 0.0 < top_p < 1.0)
        )

        gen_ids = self.lm.generate(
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            do_sample=do_sample,
            temperature=temperature if do_sample else None,
            top_k=(top_k if top_k > 0 else 4) if penalty_alpha is not None
            else (top_k if do_sample and top_k > 0 else None),
            top_p=top_p if do_sample and 0.0 < top_p < 1.0 else None,
            eos_token_id=self.lm.eos_token_id,
            pad_token_id=self.pad_token_id,
            use_cache=True,
            **extra,
        )
        return self.lm.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
