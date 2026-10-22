"""Online adapters for metrics that require decoding model features to media."""

from __future__ import annotations

from collections.abc import Mapping

import torch

from muni.evaluation.metrics.image_text_audio import (
    compute_alignment,
    select_alignment_kind,
)
from muni.evaluation.image_text_audio.decoding import decode_features
from muni.util.misc import unwrap_modality_feature


class DecodedAlignmentMetric:
    """Decode online outputs, then invoke canonical multimodal score kernels."""

    def __init__(
        self,
        num_samples: int = 32,
        checkpoint_paths: dict | None = None,
        decode_batch_size: int = 8,
        decoder_names: dict | None = None,
        renderer_settings: dict | None = None,
    ):
        self.num_samples = int(num_samples)
        self.checkpoint_paths = checkpoint_paths
        self.decode_batch_size = int(decode_batch_size)
        self.decoder_names = dict(decoder_names or {})
        self.renderer_settings = renderer_settings
        self._scores: dict[tuple[str, str, str], list[torch.Tensor]] = {}
        self._route: tuple[tuple[str, ...], str] | None = None
        self._num_collected = 0

    def update(
        self,
        preds,
        targets=None,
        *,
        sources=None,
        src_modality: str | None = None,
        src_modalities: list[str] | None = None,
        tgt_modality: str | None = None,
        raw_source_gt=None,
        **kwargs,
    ) -> None:
        del targets, kwargs
        if tgt_modality is None:
            return

        canonical_sources = tuple(
            sorted(src_modalities or ([src_modality] if src_modality else []))
        )
        if not canonical_sources:
            return

        route = (canonical_sources, tgt_modality)

        predictions_are_strings = isinstance(preds, list)

        remaining = self.num_samples - self._num_collected
        if remaining <= 0:
            return
        count = min(len(preds), remaining)
        if count == 0:
            return

        source_container = raw_source_gt if raw_source_gt is not None else sources
        if source_container is None:
            return
        if len(canonical_sources) > 1:
            source_payloads = {
                source: source_container[source] for source in canonical_sources
            }
        else:
            source = canonical_sources[0]
            if isinstance(source_container, Mapping) and source in source_container:
                source_container = source_container[source]
            source_payloads = {source: source_container}

        device = "cuda" if torch.cuda.is_available() else "cpu"
        raw_target = (
            preds[:count]
            if predictions_are_strings
            else decode_features(
                preds[:count],
                tgt_modality,
                device=device,
                batch_size=self.decode_batch_size,
                checkpoint_paths=self.checkpoint_paths,
                decoder_name=self.decoder_names.get(tgt_modality),
                renderer_settings=self.renderer_settings,
            )
        )

        batch_scores: dict[tuple[str, str, str], torch.Tensor] = {}
        for source_modality_name in canonical_sources:
            source_payload = source_payloads[source_modality_name]
            if raw_source_gt is not None:
                raw_source = source_payload[:count]
            else:
                source_features = unwrap_modality_feature(source_payload)
                raw_source = decode_features(
                    source_features[:count],
                    source_modality_name,
                    device=device,
                    batch_size=self.decode_batch_size,
                    checkpoint_paths=self.checkpoint_paths,
                    decoder_name=self.decoder_names.get(source_modality_name),
                    renderer_settings=self.renderer_settings,
                )

            kind = select_alignment_kind(source_modality_name, tgt_modality)
            _, scores = compute_alignment(
                kind,
                source_modality_name,
                raw_source,
                raw_target,
                device,
            )
            batch_scores[(source_modality_name, tgt_modality, kind)] = scores

        self._route = route
        self._num_collected += count
        for score_key, scores in batch_scores.items():
            self._scores.setdefault(score_key, []).append(scores)

    def compute(self) -> dict[str, float]:
        if not self._scores or self._route is None:
            return {}
        source_modalities, _ = self._route
        multi_source = len(source_modalities) > 1
        results: dict[str, float] = {}
        for (source, target, kind), parts in sorted(self._scores.items()):
            name = f"{kind}_score"
            if multi_source:
                name = f"{source}_to_{target}/{name}"
            results[name] = float(torch.cat(parts).mean().item())
        return results
