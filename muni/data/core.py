"""Registry and base contract for config-driven subdatasets.

This module is dependency-light so concrete sources and dataset composition
can share one contract without importing each other.
"""

from abc import ABC, abstractmethod
import torch


SUBDATASET_REGISTRY: dict[str, type] = {}


def register_subdataset(cls):
    """Register a :class:`BaseSubdataset` subclass by ``registry_name``."""
    SUBDATASET_REGISTRY[cls.registry_name] = cls
    return cls


class BaseSubdataset(ABC):
    """Key-addressable data source for one modality."""

    registry_name: str

    @abstractmethod
    def keys(self) -> list[str]:
        """Return sorted sample keys."""
        ...

    @abstractmethod
    def get(
        self,
        key: str,
        augment_state: dict | None = None,
    ) -> torch.Tensor | dict[str, torch.Tensor]:
        """Return one sample, optionally using synchronized augmentation."""
        ...
