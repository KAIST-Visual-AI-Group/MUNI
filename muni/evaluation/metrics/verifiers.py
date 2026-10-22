"""Online metrics backed by cached, frozen task verifiers."""

from __future__ import annotations

import torch

from muni.evaluation.metrics.base import BaseMetric
from muni.evaluation.metrics.utils import per_sample_class_correct

from muni.util.misc import preserve_rng_state
from verifiers.verify import FrozenVerifier


_VERIFIER_CACHE: dict[tuple[str, str], FrozenVerifier] = {}


def _load_verifier(ckpt: str, device: str = "cuda") -> FrozenVerifier:
    key = (str(ckpt), str(device))
    if key not in _VERIFIER_CACHE:
        with preserve_rng_state(torch.device(device)):
            _VERIFIER_CACHE[key] = FrozenVerifier.from_checkpoint(ckpt, device=device)
    return _VERIFIER_CACHE[key]


def _pick_source(sources, key: str):
    """Select one modality from multi-source input; preserve tensor input."""
    if isinstance(sources, dict):
        return sources[key]
    return sources


class VerifierClassificationMetric(BaseMetric):
    """Score an image translation with one frozen classifier head."""

    def __init__(self, ckpt: str, head: str, device: str = "cuda", **kwargs):
        self.verifier = _load_verifier(ckpt, device=device)
        self.head = str(head)
        self.correct = 0
        self.total = 0

    def update(self, preds: torch.Tensor, targets: torch.Tensor | None = None, **kwargs) -> None:
        raw_sources = kwargs.get("sources")
        if raw_sources is None:
            return  # nothing to score against
        source = _pick_source(raw_sources, self.head)
        pred_cls = self.verifier.predict_classes(preds, heads=(self.head,))[self.head]
        tgt_cls = source.flatten().long().to(pred_cls.device)
        self.correct += per_sample_class_correct(pred_cls, tgt_cls).sum().item()
        self.total += tgt_cls.numel()

    def compute(self) -> dict[str, float]:
        acc = self.correct / max(self.total, 1)
        self.correct = 0
        self.total = 0
        # Head-qualified keys avoid collisions in multi-head suites.
        return {f"VerifierAcc/{self.head}": acc}


class VerifierAccuracyAcrossModalities(BaseMetric):
    """Average one verifier head across co-generated image modalities."""

    def __init__(
        self,
        ckpt: str,
        head: str,
        modality_keys,
        device: str = "cuda",
        **kwargs,
    ):
        self.verifier = _load_verifier(ckpt, device=device)
        self.head = str(head)
        self.modality_keys: list[str] = [str(m) for m in modality_keys]
        self._correct: dict[str, int] = {m: 0 for m in self.modality_keys}
        self._total: dict[str, int] = {m: 0 for m in self.modality_keys}

    @torch.no_grad()
    def update(self, preds: torch.Tensor, targets: torch.Tensor | None = None, **kwargs) -> None:
        raw_sources = kwargs.get("sources")
        if not isinstance(raw_sources, dict):
            return
        gt = raw_sources.get(self.head)
        if gt is None:
            return
        gt_long = gt.flatten().long()
        for mk in self.modality_keys:
            mod_img = raw_sources.get(mk)
            if mod_img is None:
                continue
            pred = self.verifier.predict_classes(mod_img, heads=(self.head,))[self.head]
            self._correct[mk] += per_sample_class_correct(
                pred, gt_long.to(pred.device)
            ).sum().item()
            self._total[mk] += pred.numel()

    def compute(self) -> dict[str, float]:
        per_mod: dict[str, float] = {}
        contributing: list[float] = []
        for mk in self.modality_keys:
            total = self._total[mk]
            accuracy = self._correct[mk] / max(total, 1)
            per_mod[mk] = accuracy
            if total:
                contributing.append(accuracy)
            self._correct[mk] = 0
            self._total[mk] = 0
        avg = (sum(contributing) / len(contributing)) if contributing else 0.0
        out: dict[str, float] = {f"VerifierAcc/{self.head}/avg": avg}
        for mk, v in per_mod.items():
            out[f"VerifierAcc/{self.head}/{mk}"] = v
        return out
