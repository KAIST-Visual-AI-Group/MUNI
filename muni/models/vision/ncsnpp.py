from collections.abc import Sequence as SequenceABC
from functools import partial
import math

import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange

from muni.models.components.transformer import zero_module




def _resolve_grid_hw(grid_size: int | tuple[int, int] | list[int]) -> tuple[int, int]:
    if isinstance(grid_size, SequenceABC) and not isinstance(grid_size, (str, bytes)):
        if len(grid_size) != 2:
            raise ValueError("grid_size sequence must have exactly two values: (H, W)")
        return int(grid_size[0]), int(grid_size[1])

    side = int(grid_size)
    return side, side


def _make_group_norm(channels: int) -> nn.GroupNorm:
    groups = min(max(channels // 4, 1), 32)
    while channels % groups != 0 and groups > 1:
        groups -= 1
    return nn.GroupNorm(num_groups=groups, num_channels=channels, eps=1e-6)


def _normalized_fir_kernel(kernel: SequenceABC | torch.Tensor) -> torch.Tensor:
    if isinstance(kernel, torch.Tensor):
        fir = kernel.float()
    else:
        fir = torch.tensor(kernel, dtype=torch.float32)
    if fir.ndim == 1:
        fir = torch.outer(fir, fir)
    fir = fir / fir.sum().clamp_min(1e-12)
    return fir


def _blur_same(x: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    kh, kw = kernel.shape
    pad = ((kw - 1) // 2, kw // 2, (kh - 1) // 2, kh // 2)
    x = F.pad(x, pad, mode="reflect")
    weight = kernel.to(device=x.device, dtype=x.dtype).expand(x.shape[1], 1, kh, kw)
    return F.conv2d(x, weight, groups=x.shape[1])


class GaussianFourierProjection(nn.Module):
    """Fixed random Fourier features for continuous timesteps.

    The frequencies are persistent buffers so checkpoints preserve their basis.
    """

    def __init__(self, embedding_size: int, scale: float = 16.0):
        super().__init__()
        self.register_buffer(
            "weight",
            torch.randn(embedding_size) * scale,
            persistent=True,
        )

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        angles = t[:, None] * self.weight[None, :] * (2.0 * math.pi)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=-1)


class SelfAttention2d(nn.Module):
    """2D self-attention."""

    def __init__(self, channels: int, num_heads: int, *, skip_scale: float = 1.0):
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = channels // self.num_heads
        self.norm = _make_group_norm(channels)
        self.to_qkv = nn.Linear(channels, channels * 3, bias=False)
        self.out_proj = zero_module(nn.Linear(channels, channels))
        # ``skip_scale`` matches EDM's UNetBlock attention rescale:
        # after residual add the whole tensor is multiplied by √0.5.  With
        # the default 1.0 this is a no-op and falls back to classic NCSN++.
        self.skip_scale = float(skip_scale)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        _, _, h, w = x.shape
        tokens = rearrange(self.norm(x), "b c h w -> b (h w) c")
        qkv = self.to_qkv(tokens)  # (B, L, 3C)
        q, k, v = rearrange(
            qkv,
            "b l (three n d) -> three b n l d",
            three=3,
            n=self.num_heads,
        )
        attn_out = F.scaled_dot_product_attention(q, k, v)  # (B, n, L, d)
        attn_out = rearrange(attn_out, "b n l d -> b l (n d)")
        attn_out = self.out_proj(attn_out)
        attn_out = rearrange(attn_out, "b (h w) c -> b c h w", h=h, w=w)
        out = x + attn_out
        if self.skip_scale != 1.0:
            out = out * self.skip_scale
        return out


class FIRUpsample(nn.Module):
    def __init__(
        self, fir: bool = True, fir_kernel: SequenceABC | torch.Tensor = (1, 3, 3, 1)
    ):
        super().__init__()
        self.fir = bool(fir)
        self.register_buffer(
            "fir_kernel", _normalized_fir_kernel(fir_kernel), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, scale_factor=2.0, mode="nearest")
        if self.fir:
            x = _blur_same(x, self.fir_kernel)
        return x


class FIRDownsample(nn.Module):
    def __init__(
        self, fir: bool = True, fir_kernel: SequenceABC | torch.Tensor = (1, 3, 3, 1)
    ):
        super().__init__()
        self.fir = bool(fir)
        self.register_buffer(
            "fir_kernel", _normalized_fir_kernel(fir_kernel), persistent=False
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.fir:
            x = _blur_same(x, self.fir_kernel)
            return x[:, :, ::2, ::2]
        return F.avg_pool2d(x, kernel_size=2, stride=2)


class ResnetBlockBigGANpp(nn.Module):
    """BigGAN-style residual block with optional FIR-filtered resampling."""

    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        temb_dim: int,
        *,
        dropout: float = 0.0,
        up: bool = False,
        down: bool = False,
        fir: bool = True,
        fir_kernel: SequenceABC | torch.Tensor = (1, 3, 3, 1),
    ):
        super().__init__()
        self.in_ch = int(in_ch)
        self.out_ch = int(out_ch)
        self.up = bool(up)
        self.down = bool(down)
        self.blur_up = FIRUpsample(fir=fir, fir_kernel=fir_kernel)
        self.blur_down = FIRDownsample(fir=fir, fir_kernel=fir_kernel)

        self.norm1 = _make_group_norm(self.in_ch)
        self.conv1 = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=3, padding=1)
        self.temb_proj = nn.Linear(temb_dim, self.out_ch)

        self.norm2 = _make_group_norm(self.out_ch)
        self.dropout = nn.Dropout(dropout)
        self.conv2 = zero_module(
            nn.Conv2d(self.out_ch, self.out_ch, kernel_size=3, padding=1)
        )

        if self.in_ch != self.out_ch or self.up or self.down:
            self.skip = nn.Conv2d(self.in_ch, self.out_ch, kernel_size=1)
        else:
            self.skip = nn.Identity()

    def _resample(self, x: torch.Tensor) -> torch.Tensor:
        if self.up:
            return self.blur_up(x)
        if self.down:
            return self.blur_down(x)
        return x

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        residual = self._resample(x)

        h = self.norm1(x)
        h = F.silu(h)
        h = self._resample(h)
        h = self.conv1(h)

        h = h + rearrange(self.temb_proj(F.silu(temb)), "B C -> B C 1 1")
        h = self.norm2(h)
        h = F.silu(h)
        h = self.dropout(h)
        h = self.conv2(h)

        residual = self.skip(residual)
        return (residual + h) * (2.0**-0.5)


class NCSNppBlock(nn.Module):
    def __init__(
        self,
        in_ch: int,
        out_ch: int,
        temb_dim: int,
        *,
        dropout: float,
        num_heads: int,
        use_self_attn: bool,
        up: bool = False,
        down: bool = False,
        attn_skip_scale: float = 1.0,
        fir: bool = True,
        fir_kernel: SequenceABC | torch.Tensor = (1, 3, 3, 1),
    ):
        super().__init__()
        self.resblock = ResnetBlockBigGANpp(
            in_ch=in_ch,
            out_ch=out_ch,
            temb_dim=temb_dim,
            dropout=dropout,
            up=up,
            down=down,
            fir=fir,
            fir_kernel=fir_kernel,
        )
        self.self_attn = (
            SelfAttention2d(out_ch, num_heads, skip_scale=attn_skip_scale)
            if use_self_attn
            else None
        )

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        x = self.resblock(x, temb)
        if self.self_attn is not None:
            x = self.self_attn(x)
        return x




class NCSNpp(nn.Module):
    """Token-grid NCSN++ with latent-context conditioning."""

    def _tokens_to_grid(self, raw: torch.Tensor) -> torch.Tensor:
        return rearrange(raw, "b (h w) d -> b d h w", h=self.grid_h, w=self.grid_w)

    def _grid_to_tokens(self, x: torch.Tensor) -> torch.Tensor:
        return rearrange(x, "b d h w -> b (h w) d")

    def _set_resolution_schedule(self, ch_mult, min_resolution):
        if self.num_res_blocks <= 0:
            raise ValueError("num_res_blocks must be positive")
        if self.progressive_input not in {"none", "input_skip"}:
            raise ValueError(f"Unsupported progressive_input mode: {self.progressive_input}")
        active_mults = []
        cur_h, cur_w = self.grid_h, self.grid_w
        for mult in ch_mult:
            active_mults.append(int(mult))
            if cur_h % 2 != 0 or cur_w % 2 != 0 or min(cur_h, cur_w) <= min_resolution:
                break
            cur_h //= 2
            cur_w //= 2
        self.ch_mult = tuple(active_mults)
        self.num_resolutions = len(self.ch_mult)
        self.all_resolutions = tuple(
            min(self.grid_h, self.grid_w) // (2**i) for i in range(self.num_resolutions)
        )

    def _build_time_label_modules(self, fourier_scale):
        temb_dim = self.nf * 4
        self.time_proj = GaussianFourierProjection(self.nf, scale=fourier_scale)
        self.time_mlp = nn.Sequential(
            nn.Linear(self.nf * 2, temb_dim),
            nn.SiLU(),
            nn.Linear(temb_dim, temb_dim),
        )
        self.label_emb = nn.Embedding(self.num_classes + 1, temb_dim)
        return temb_dim

    def _build_context_projection(self, temb_dim):
        if self.context_dim is not None:
            self.context_proj = nn.Linear(self.context_dim, temb_dim)
            self.context_to_temb = nn.Sequential(
                nn.LayerNorm(temb_dim),
                nn.Linear(temb_dim, temb_dim),
                nn.SiLU(),
                nn.Linear(temb_dim, temb_dim),
            )
        else:
            self.context_proj = None
            self.context_to_temb = None

    def _init_weights(self) -> None:
        nn.init.normal_(self.label_emb.weight, std=0.02)


    def _build_unet(
        self, *, temb_dim, dropout, num_heads, attn_skip_scale,
        fir, fir_kernel,
    ):
        """Build one U-Net topology with stable module registration order."""
        block = partial(
            NCSNppBlock, temb_dim=temb_dim, dropout=dropout, num_heads=num_heads,
            attn_skip_scale=attn_skip_scale,
            fir=fir, fir_kernel=fir_kernel,
        )
        self.input_conv = nn.Conv2d(self.token_dim, self.nf, kernel_size=3, padding=1)
        self.input_pyramid_downsamples = nn.ModuleList()
        self.input_pyramid_projs = nn.ModuleList()
        hs_c = [self.nf]
        in_ch = self.nf
        self.down_blocks = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for level, mult in enumerate(self.ch_mult):
            out_ch = self.nf * mult
            resolution = self.all_resolutions[level]
            blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks):
                blocks.append(block(
                    in_ch=in_ch, out_ch=out_ch,
                    use_self_attn=resolution in self.attn_resolutions,
                ))
                in_ch = out_ch
                hs_c.append(in_ch)
            self.down_blocks.append(blocks)
            if level != self.num_resolutions - 1:
                self.downsamples.append(block(
                    in_ch=in_ch, out_ch=in_ch, down=True,
                    use_self_attn=False,
                ))
                hs_c.append(in_ch)
                if self.progressive_input == "input_skip":
                    self.input_pyramid_downsamples.append(FIRDownsample(fir=fir, fir_kernel=fir_kernel))
                    self.input_pyramid_projs.append(nn.Conv2d(self.token_dim, in_ch, kernel_size=1))
            else:
                self.downsamples.append(nn.Identity())
        self.mid_block1 = block(
            in_ch=in_ch, out_ch=in_ch, use_self_attn=True,
        )
        self.mid_block2 = block(
            in_ch=in_ch, out_ch=in_ch, use_self_attn=False,
        )
        self.up_blocks = nn.ModuleList()
        self.upsamples = nn.ModuleList()
        for up_idx, level in enumerate(reversed(range(self.num_resolutions))):
            resolution = self.all_resolutions[level]
            out_ch = self.nf * self.ch_mult[level]
            blocks = nn.ModuleList()
            for _ in range(self.num_res_blocks + 1):
                blocks.append(block(
                    in_ch=in_ch + hs_c.pop(), out_ch=out_ch,
                    use_self_attn=resolution in self.attn_resolutions,
                ))
                in_ch = out_ch
            self.up_blocks.append(blocks)
            if level != 0:
                self.upsamples.append(block(
                    in_ch=in_ch, out_ch=in_ch, up=True,
                    use_self_attn=False,
                ))
            else:
                self.upsamples.append(nn.Identity())
        self.out_norm = _make_group_norm(in_ch)
        self.out_conv = zero_module(nn.Conv2d(in_ch, self.token_dim, kernel_size=3, padding=1))

    def _time_embedding(self, t: torch.Tensor) -> torch.Tensor:
        return self.time_mlp(self.time_proj(t))


    def _run_unet(self, x, h, temb):
        hs = [h]
        input_pyramid = x if self.progressive_input == "input_skip" else None
        input_skip_idx = 0
        for level, blocks in enumerate(self.down_blocks):
            for block in blocks:
                h = block(h, temb)
                hs.append(h)
            downsample = self.downsamples[level]
            if not isinstance(downsample, nn.Identity):
                h = downsample(h, temb)
                if self.progressive_input == "input_skip":
                    input_pyramid = self.input_pyramid_downsamples[input_skip_idx](
                        input_pyramid
                    )
                    h = h + self.input_pyramid_projs[input_skip_idx](input_pyramid)
                    input_skip_idx += 1
                hs.append(h)
        h = self.mid_block1(h, temb)
        h = self.mid_block2(h, temb)
        for up_idx, blocks in enumerate(self.up_blocks):
            for block in blocks:
                h = torch.cat([h, hs.pop()], dim=1)
                h = block(h, temb)
            upsample = self.upsamples[up_idx]
            if not isinstance(upsample, nn.Identity):
                h = upsample(h, temb)
        return self.out_conv(F.silu(self.out_norm(h)))


    def __init__(
        self,
        token_dim: int,
        num_classes: int,
        grid_size: int | tuple[int, int] | list[int],
        *,
        context_dim: int | None = None,
        label_drop_prob: float = 0.0,
        nf: int = 64,
        ch_mult: tuple[int, ...] = (1, 1, 2, 2, 2, 2),
        num_res_blocks: int = 2,
        attn_resolutions: tuple[int, ...] = (16,),
        dropout: float = 0.0,
        attn_skip_scale: float = 1.0,
        num_heads: int = 4,
        fourier_scale: float = 16.0,
        min_resolution: int = 4,
        fir: bool = True,
        fir_kernel: SequenceABC | torch.Tensor = (1, 3, 3, 1),
        progressive_input: str = "input_skip",
        context_seq_len: int | None = None,
        context_drop_prob: float = 0.0,
    ):
        super().__init__()

        self.token_dim = int(token_dim)
        self.num_classes = int(num_classes)
        self.context_dim = int(context_dim) if context_dim is not None else None
        self.label_drop_prob = float(label_drop_prob)
        self.grid_h, self.grid_w = _resolve_grid_hw(grid_size)
        self.seq_len = self.grid_h * self.grid_w
        self.nf = int(nf)
        self.num_res_blocks = int(num_res_blocks)
        self.attn_resolutions = set(int(r) for r in attn_resolutions)
        self.progressive_input = progressive_input
        self.context_drop_prob = float(context_drop_prob)

        self._set_resolution_schedule(ch_mult, min_resolution)

        temb_dim = self._build_time_label_modules(fourier_scale)
        self._build_context_projection(temb_dim)

        if self.context_dim is not None and self.context_drop_prob > 0.0:
            self.null_context = torch.nn.Parameter(
                torch.zeros(1, int(context_seq_len), temb_dim)
            )
        else:
            self.null_context = None

        self._build_unet(
            temb_dim=temb_dim, dropout=dropout, num_heads=num_heads,
            attn_skip_scale=attn_skip_scale,
            fir=fir, fir_kernel=fir_kernel,
        )

        self._init_weights()

    def _drop_context(self, memory: torch.Tensor, batch_size: int) -> torch.Tensor:
        if (
            self.training
            and self.context_drop_prob > 0.0
            and self.null_context is not None
        ):
            drop = (
                torch.rand(batch_size, 1, 1, device=memory.device)
                < self.context_drop_prob
            )
            null_ctx = self.null_context.to(memory.dtype).expand(batch_size, -1, -1)
            memory = torch.where(drop, null_ctx, memory)
        return memory

    def _condition(self, temb, context, batch_size):
        memory = None
        if self.context_dim is not None:
            if context is not None:
                memory = self._drop_context(self.context_proj(context), batch_size)
            elif self.null_context is not None:
                memory = self.null_context.to(temb.dtype).expand(batch_size, -1, -1)
        if memory is not None:
            temb = temb + self.context_to_temb(memory.mean(dim=1))
        return temb

    def forward(
        self,
        raw: torch.Tensor,
        grid_pos: torch.Tensor,
        t: torch.Tensor,
        y: torch.Tensor,
        context: torch.Tensor | None = None,
    ) -> torch.Tensor:
        x = self._tokens_to_grid(raw)
        b = x.shape[0]
        temb = self._time_embedding(t)
        if self.training and self.label_drop_prob > 0.0:
            drop = torch.rand(b, device=y.device) < self.label_drop_prob
            y = torch.where(drop, torch.full_like(y, self.num_classes), y)
        temb = temb + self.label_emb(y)
        temb = self._condition(temb, context, b)
        h = self.input_conv(x)
        return self._grid_to_tokens(self._run_unet(x, h, temb))


