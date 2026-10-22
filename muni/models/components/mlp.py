import torch
import torch.nn as nn

from muni.models.components.resampler import PerceiverResampler


class ResBlock(nn.Module):
    """Pre-normalized residual MLP block."""

    def __init__(self, dim: int):
        super().__init__()
        self.norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.fc1 = nn.Linear(dim, dim)
        self.fc2 = nn.Linear(dim, dim)
        self.act = nn.SiLU()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.norm(x)
        h = self.fc2(self.act(self.fc1(self.act(h))))
        return x + h


class MLP(nn.Module):
    """Residual multilayer perceptron."""

    def __init__(self, in_dim: int, hidden_dim: int, out_dim: int, depth: int = 2):
        super().__init__()
        self.in_proj = nn.Sequential(nn.Linear(in_dim, hidden_dim), nn.SiLU())
        self.blocks = nn.Sequential(*[ResBlock(hidden_dim) for _ in range(depth)])
        self.out_proj = nn.Linear(hidden_dim, out_dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.in_proj(x)
        x = self.blocks(x)
        return self.out_proj(x)


class ClassEncoderMLP(nn.Module):
    """Embeds a class label into a learned fixed-length latent sequence `(B, L, D)`."""

    def __init__(
        self,
        num_classes: int,
        latent_dim: int,
        hidden_dim: int,
        depth: int = 2,
        num_latents: int = 1,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.num_latents = int(num_latents)
        self.embed = nn.Embedding(self.num_classes, hidden_dim)
        self.cond_proj = MLP(hidden_dim, hidden_dim, hidden_dim, depth=depth)
        self.latent_tokens = nn.Parameter(
            torch.randn(1, self.num_latents, hidden_dim) * 0.02
        )
        self.modulation = nn.Linear(hidden_dim, hidden_dim * 2)
        self.proj = MLP(hidden_dim, hidden_dim, latent_dim, depth=depth)

    def _normalize_labels(self, x: torch.Tensor) -> torch.Tensor:
        if x.ndim == 2 and x.shape[1] == 1:
            x = x[:, 0]
        return x.long()

    def _project_condition(self, cond: torch.Tensor) -> torch.Tensor:
        cond = self.cond_proj(cond)
        scale, shift = self.modulation(cond).chunk(2, dim=-1)
        tokens = self.latent_tokens.expand(cond.shape[0], -1, -1)
        tokens = tokens * (1.0 + scale[:, None, :]) + shift[:, None, :]
        return self.proj(tokens)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self._project_condition(self.embed(self._normalize_labels(x)))


class LatentClassifierMLP(nn.Module):
    """Categorical classifier with learned attention-style pooling over latent tokens."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int,
        num_classes: int,
        depth: int = 2,
        pool_depth: int = 2,
        num_heads: int = 8,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.pool = PerceiverResampler(
            input_dim=latent_dim,
            hidden_size=hidden_dim,
            output_dim=hidden_dim,
            num_queries=1,
            depth=pool_depth,
            num_heads=num_heads,
        )
        self.backbone = MLP(hidden_dim, hidden_dim, self.num_classes, depth=depth)

    def forward(self, z_cond: torch.Tensor) -> torch.Tensor:
        pooled = self.pool(z_cond)
        return self.backbone(pooled[:, 0])


class MeanPooledLatentClassifierMLP(nn.Module):
    """Categorical classifier over the mean latent token."""

    def __init__(
        self,
        latent_dim: int,
        hidden_dim: int = 128,
        num_classes: int = 1,
        depth: int = 1,
    ):
        super().__init__()
        self.num_classes = int(num_classes)
        self.latent_dim = int(latent_dim)
        self.hidden_dim = int(hidden_dim)
        self.depth = int(depth)
        self.backbone = MLP(
            self.latent_dim, self.hidden_dim, self.num_classes, depth=self.depth
        )

    def forward(self, z_cond: torch.Tensor) -> torch.Tensor:
        if z_cond.shape[1] == 0:
            raise ValueError(
                "MeanPooledLatentClassifierMLP received an empty latent sequence."
            )
        return self.backbone(z_cond.mean(dim=1))
