"""Spatial tokenization and gradient utilities."""

from collections.abc import Sequence as SequenceABC

import torch
import torch.nn as nn
from einops import rearrange


# Gradient helpers


def scale_grad(x: torch.Tensor, scale: float | torch.Tensor = 1.0) -> torch.Tensor:
    """Scale gradients of *x* by *scale* (scalar or per-sample) using the detach trick."""
    if not x.requires_grad:
        return x
    if isinstance(scale, (int, float)) and scale == 1.0:
        return x
    if isinstance(scale, torch.Tensor):
        scale = scale.view(-1, *([1] * (x.ndim - 1)))
    return x.detach() + scale * (x - x.detach())


# Grid positions


def make_grid_pos(h: int, w: int, device: torch.device | None = None) -> torch.Tensor:
    """2D grid positions ``(h*w, 2)`` with row-major ordering."""
    ys = torch.arange(h, device=device, dtype=torch.float32)
    xs = torch.arange(w, device=device, dtype=torch.float32)
    grid = torch.stack(torch.meshgrid(ys, xs, indexing="ij"), dim=-1)
    return rearrange(grid, "h w d -> (h w) d")


def make_sequence_grid_pos(length: int, device: torch.device) -> torch.Tensor:
    """1D sequence positions as ``(length, 2)`` with the second column zeroed."""
    positions = torch.arange(length, device=device, dtype=torch.float32)
    return torch.stack([positions, torch.zeros(length, device=device)], dim=-1)


# Patch conversion


def patchify(x: torch.Tensor, patch_size: int) -> torch.Tensor:
    """``(B, C, H, W) -> (B, num_patches, patch_size² * C)``."""
    return rearrange(
        x,
        "B C (h p1) (w p2) -> B (h w) (p1 p2 C)",
        p1=patch_size,
        p2=patch_size,
    )


def unpatchify(
    x: torch.Tensor,
    patch_size: int,
    in_channels: int,
    grid_h: int,
    grid_w: int,
) -> torch.Tensor:
    """``(B, num_patches, patch_size² * C) -> (B, C, H, W)``."""
    return rearrange(
        x,
        "B (h w) (p1 p2 C) -> B C (h p1) (w p2)",
        h=grid_h,
        w=grid_w,
        p1=patch_size,
        p2=patch_size,
        C=in_channels,
    )


# Image size


def resolve_img_hw(img_size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    """Normalize *img_size* to an ``(H, W)`` tuple."""
    if isinstance(img_size, SequenceABC) and not isinstance(img_size, (str, bytes)):
        height, width = img_size
        return int(height), int(width)
    side = int(img_size)
    return side, side


# Spatial tokenizer


class JiTTokenizer(nn.Module):
    """Convert images to patch sequences and back without learned parameters."""

    def __init__(
        self,
        img_size: int | tuple[int, int] | list[int] = 256,
        patch_size: int = 16,
        in_channels: int = 3,
    ):
        super().__init__()
        self.img_h, self.img_w = resolve_img_hw(img_size)
        self.patch_size = patch_size
        self.in_channels = in_channels

        if self.img_h <= 0 or self.img_w <= 0:
            raise ValueError("Image height and width must be positive")
        if self.img_h % patch_size != 0 or self.img_w % patch_size != 0:
            raise ValueError(
                f"Image size ({self.img_h}, {self.img_w}) must be divisible by patch_size={patch_size}"
            )

        self.grid_h = self.img_h // patch_size
        self.grid_w = self.img_w // patch_size
        self.seq_len = self.grid_h * self.grid_w

        self.register_buffer("_grid_pos", make_grid_pos(self.grid_h, self.grid_w))

    @property
    def grid_pos(self):
        return self._grid_pos

    def tokenize(self, x: torch.Tensor):
        return patchify(x, self.patch_size), self._grid_pos

    def decode(self, x: torch.Tensor) -> torch.Tensor:
        return unpatchify(
            x, self.patch_size, self.in_channels, self.grid_h, self.grid_w
        )
