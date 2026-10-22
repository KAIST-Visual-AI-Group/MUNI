from functools import partial

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange


class ResnetBlock(nn.Module):
    """Residual block with optional zero-init modality FiLM after each norm."""

    def __init__(
        self,
        in_c: int,
        out_c: int,
        down: bool = False,
        *,
        num_modalities: int | None = None,
    ):
        super().__init__()
        self.conv1 = nn.Conv2d(
            in_c, out_c, 3, stride=2 if down else 1, padding=1, bias=False
        )
        self.norm1 = nn.GroupNorm(32, out_c)
        self.film1 = (
            ModalityFiLM2d(num_modalities, out_c) if num_modalities is not None else None
        )
        self.conv2 = nn.Conv2d(out_c, out_c, 3, stride=1, padding=1, bias=False)
        self.norm2 = nn.GroupNorm(32, out_c)
        self.film2 = (
            ModalityFiLM2d(num_modalities, out_c) if num_modalities is not None else None
        )

        self.down = down
        if in_c != out_c or down:
            self.shortcut = nn.Conv2d(
                in_c, out_c, 1, stride=2 if down else 1, bias=False
            )
        else:
            self.shortcut = nn.Identity()

    def forward(
        self,
        x: torch.Tensor,
        modality_idx: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        h = self.norm1(self.conv1(x))
        if self.film1 is not None:
            h = self.film1(h, modality_idx)
        h = F.silu(h)
        h = self.norm2(self.conv2(h))
        if self.film2 is not None:
            h = self.film2(h, modality_idx)
        return F.silu(h + self.shortcut(x))


def _modality_indices(
    modality_idx: int | torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> torch.Tensor:
    if isinstance(modality_idx, int):
        return torch.full((batch_size,), modality_idx, device=device, dtype=torch.long)
    idx = modality_idx.to(device=device, dtype=torch.long)
    if idx.ndim == 0 or idx.numel() == 1:
        return idx.reshape(1).expand(batch_size)
    if idx.ndim != 1 or idx.shape[0] != batch_size:
        raise ValueError(
            f"modality_idx tensor must be scalar or shape (B,), got {tuple(idx.shape)} for batch_size={batch_size}"
        )
    return idx


def _scalar_modality_idx(
    modality_idx: int | torch.Tensor,
    batch_size: int,
    device: torch.device,
) -> int:
    if isinstance(modality_idx, int):
        return modality_idx
    idx = _modality_indices(modality_idx, batch_size, device)
    first = int(idx[0].item())
    if bool((idx != first).any().item()):
        raise ValueError(
            "SDStyleResNetEncoder shared mode expects one modality per "
            "forward call because input/output heads are modality-specific. "
            "Call it separately per modality."
        )
    return first


class ModalityFiLM2d(nn.Module):
    """Zero-initialised modality FiLM for a 2D feature map."""

    def __init__(self, num_modalities: int, channels: int):
        super().__init__()
        self.num_modalities = int(num_modalities)
        self.channels = int(channels)
        self.embed = nn.Embedding(self.num_modalities, 2 * self.channels)
        nn.init.zeros_(self.embed.weight)

    def forward(
        self,
        x: torch.Tensor,
        modality_idx: int | torch.Tensor,
    ) -> torch.Tensor:
        idx = _modality_indices(modality_idx, x.shape[0], x.device)
        gamma, beta = self.embed(idx).to(dtype=x.dtype).chunk(2, dim=-1)
        gamma = gamma[..., None, None]
        beta = beta[..., None, None]
        return x * (1.0 + gamma) + beta


# SD VAE-style encoder components


class SelfAttention2d(nn.Module):
    """Spatial self-attention for 2D feature maps."""

    def __init__(self, channels: int):
        super().__init__()
        self.norm = nn.GroupNorm(32, channels)
        self.qkv = nn.Conv2d(channels, channels * 3, 1)
        self.proj = nn.Conv2d(channels, channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, H, W = x.shape
        h = self.norm(x)
        # Single-head SDPA: (B, 3C, H, W) → 3 × (B, 1, HW, C)
        qkv = rearrange(self.qkv(h), "B (three C) H W -> three B 1 (H W) C", three=3)
        q, k, v = qkv.unbind(0)
        out = F.scaled_dot_product_attention(q, k, v)
        out = rearrange(out, "B 1 (H W) C -> B C H W", H=H, W=W)
        return x + self.proj(out)


class Downsample2d(nn.Module):
    """Strided 3×3 convolution for spatial downsampling."""

    def __init__(self, channels: int):
        super().__init__()
        self.conv = nn.Conv2d(channels, channels, 3, stride=2, padding=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)


def _make_coord_proj(pos_channels: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Conv2d(2, pos_channels, 1, bias=False),
        nn.SiLU(),
        nn.Conv2d(pos_channels, pos_channels, 1, bias=False),
    )


def _append_position_channels(x: torch.Tensor, coord_proj: nn.Module) -> torch.Tensor:
    batch_size, _, height, width = x.shape
    ys = torch.linspace(-1.0, 1.0, height, device=x.device, dtype=x.dtype)
    xs = torch.linspace(-1.0, 1.0, width, device=x.device, dtype=x.dtype)
    yy, xx = torch.meshgrid(ys, xs, indexing="ij")
    coords = torch.stack([yy, xx], dim=0).unsqueeze(0).expand(batch_size, -1, -1, -1)
    return torch.cat([x, coord_proj(coords)], dim=1)


class SDStyleResNetEncoder(nn.Module):
    """SD-style trunk with scalar or modality-specific input/output boundaries.

    ``num_modalities=None`` selects scalar boundaries; an integer, including
    one, selects shared-trunk boundaries and requires a modality ID at forward.
    """

    def __init__(
        self,
        c_in: int = 3,
        c_out: int = 512,
        ch: int = 128,
        ch_mult: tuple[int, ...] | list[int] = (1, 2, 4, 4),
        num_res_blocks: int = 2,
        pos_channels: int = 32,
        num_modalities: int | None = None,
        modality_film: bool = True,
    ):
        super().__init__()
        self.num_modalities = None if num_modalities is None else int(num_modalities)
        self.modality_film = bool(modality_film)

        self.coord_proj = _make_coord_proj(pos_channels)
        self.conv_in = (
            nn.Conv2d(c_in + pos_channels, ch, 3, padding=1)
            if self.num_modalities is None
            else nn.ModuleList(
                [
                    nn.Conv2d(c_in + pos_channels, ch, 3, padding=1)
                    for _ in range(self.num_modalities)
                ]
            )
        )
        block_factory = partial(
            ResnetBlock,
            num_modalities=self.num_modalities if self.modality_film else None,
        )
        self.down_blocks = nn.ModuleList()
        self.down_samples = nn.ModuleList()
        in_ch = int(ch)
        for mult in ch_mult:
            out_ch = int(ch) * int(mult)
            blocks = nn.ModuleList()
            for j in range(int(num_res_blocks)):
                blocks.append(block_factory(in_ch if j == 0 else out_ch, out_ch))
                in_ch = out_ch
            self.down_blocks.append(blocks)
            self.down_samples.append(Downsample2d(out_ch))
        self.mid_block1 = block_factory(in_ch, in_ch)
        self.mid_attn = SelfAttention2d(in_ch)
        self.mid_block2 = block_factory(in_ch, in_ch)

        self.norm_out = (
            nn.GroupNorm(32, in_ch)
            if self.num_modalities is None
            else nn.ModuleList(
                [nn.GroupNorm(32, in_ch) for _ in range(self.num_modalities)]
            )
        )
        self.conv_out = (
            nn.Conv2d(in_ch, c_out, 1)
            if self.num_modalities is None
            else nn.ModuleList(
                [nn.Conv2d(in_ch, c_out, 1) for _ in range(self.num_modalities)]
            )
        )

    def forward(
        self,
        x: torch.Tensor,
        modality_idx: int | torch.Tensor | None = None,
    ) -> torch.Tensor:
        conv_in, norm_out, conv_out = self.conv_in, self.norm_out, self.conv_out
        if self.num_modalities is not None:
            m = _scalar_modality_idx(modality_idx, x.shape[0], x.device)
            if m < 0 or m >= self.num_modalities:
                raise ValueError(
                    f"modality_idx must be in [0, {self.num_modalities}), got {m}"
                )
            conv_in, norm_out, conv_out = conv_in[m], norm_out[m], conv_out[m]

        x = conv_in(_append_position_channels(x, self.coord_proj))
        for blocks, downsample in zip(self.down_blocks, self.down_samples):
            for block in blocks:
                x = block(x, modality_idx)
            x = downsample(x)
        x = self.mid_block1(x, modality_idx)
        x = self.mid_attn(x)
        x = self.mid_block2(x, modality_idx)
        x = F.silu(norm_out(x))
        return conv_out(x)
