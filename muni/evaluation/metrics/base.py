"""Metric protocol shared by pluggable evaluation implementations."""

import torch


class BaseMetric:
    def update(self, preds: torch.Tensor | None = None, targets: torch.Tensor | None = None, **kwargs) -> None:
        raise NotImplementedError

    def compute(self):
        return {}

