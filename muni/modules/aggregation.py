"""Posterior fusion implementations used by MUNI and its baselines."""

from __future__ import annotations

import torch

from .core import (
    BaseAggregator,
    DiagonalGaussianPosterior,
)


def _poe_combine(
    components: list[DiagonalGaussianPosterior],
    prior_expert: bool = False,
) -> DiagonalGaussianPosterior:
    """Closed-form Gaussian product-of-experts fusion.

    Precision sum: ``Lambda = sum_m exp(-logvar_m)`` (optionally ``+ I`` when
    ``prior_expert``).  Precision-weighted mean: ``mu_PoE = Lambda^{-1} *
    sum_m exp(-logvar_m) * mu_m``.  Logvar: ``-log(Lambda)``.
    """
    precisions = [(-c.latent_logvar).exp() for c in components]
    means = [c.latent_mean for c in components]
    precision_sum = sum(precisions)
    numerator = sum(p * m for p, m in zip(precisions, means))
    if prior_expert:
        # Adds an N(0, I) expert: precision +1, mean contribution 0.
        precision_sum = precision_sum + 1.0
    mean_poe = numerator / precision_sum
    logvar_poe = -precision_sum.log()
    return DiagonalGaussianPosterior(latent_mean=mean_poe, latent_logvar=logvar_poe)


class PoEAggregator(BaseAggregator):
    """Gaussian product-of-experts fusion.

    ``prior_expert=True`` adds the Gaussian-prior expert from MVAE. Learned
    flow priors use ``False`` because their pressure enters through the KL.
    """

    def __init__(
        self,
        modalities: list[str],
        latent_dim: int,
        prior_expert: bool = False,
    ):
        super().__init__()
        self.modalities = list(modalities)
        self.latent_dim = int(latent_dim)
        self.prior_expert = bool(prior_expert)

    def forward(
        self,
        posteriors: dict[str, DiagonalGaussianPosterior],
        modality_order: list[str],
    ) -> DiagonalGaussianPosterior:
        components = [posteriors[m] for m in modality_order if m in posteriors]
        return _poe_combine(components, prior_expert=self.prior_expert)


class MoEAggregator(BaseAggregator):
    """Uniform mixture of per-modality Gaussians (MMVAE, Shi et al. 2019).

    A singleton passes through. Otherwise each batch element selects one
    observed component, exactly sampling the uniform posterior mixture.
    """

    def __init__(self, modalities: list[str], latent_dim: int):
        super().__init__()
        self.modalities = list(modalities)
        self.latent_dim = int(latent_dim)

    def forward(
        self,
        posteriors: dict[str, DiagonalGaussianPosterior],
        modality_order: list[str],
    ) -> DiagonalGaussianPosterior:
        observed = [m for m in modality_order if m in posteriors]
        if len(observed) == 1:
            return posteriors[observed[0]]
        first = posteriors[observed[0]].latent_mean
        B = int(first.shape[0])
        K = len(observed)
        device = first.device
        idx = torch.randint(K, (B,), device=device)
        batch_range = torch.arange(B, device=device)
        means = torch.stack(
            [posteriors[m].latent_mean for m in observed], dim=0
        )  # (K, B, *latent_shape)
        logvars = torch.stack([posteriors[m].latent_logvar for m in observed], dim=0)
        return DiagonalGaussianPosterior(
            latent_mean=means[idx, batch_range],
            latent_logvar=logvars[idx, batch_range],
        )


class HellingerAggregator(BaseAggregator):
    """HELVAE all-pairs Hellinger moment matching."""

    def __init__(
        self,
        modalities: list[str],
        latent_dim: int,
        ver_type: str = "f32",
        eps: float = 1e-8,
    ):
        super().__init__()
        if ver_type not in {"f32", "f64"}:
            raise ValueError(f"ver_type must be 'f32' or 'f64', got {ver_type!r}.")

        self.modalities = list(modalities)
        self.latent_dim = int(latent_dim)
        self.ver_type = ver_type
        self.eps = float(eps)

    def forward(
        self,
        posteriors: dict[str, DiagonalGaussianPosterior],
        modality_order: list[str],
    ) -> DiagonalGaussianPosterior:
        observed = [m for m in modality_order if m in posteriors]
        mus = torch.stack([posteriors[m].latent_mean for m in observed], dim=0)
        logvars = torch.stack([posteriors[m].latent_logvar for m in observed], dim=0)

        joint_mu, joint_logvar = self.helvae(mus, logvars, self.ver_type, self.eps)

        return DiagonalGaussianPosterior(
            latent_mean=joint_mu,
            latent_logvar=joint_logvar,
        )

    @staticmethod
    def helvae(
        mus: torch.Tensor,
        logvars: torch.Tensor,
        ver_type: str = "f32",
        eps: float = 1e-8,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Apply the official all-pairs formula to ``(K, ..., D)`` tensors."""
        if ver_type == "f64":
            mus = mus.double()
            logvars = logvars.double()

        ln_inv_vars = -logvars

        mu_i = mus.unsqueeze(1)
        mu_j = mus.unsqueeze(0)

        ln_inv_var_i = ln_inv_vars.unsqueeze(1)
        ln_inv_var_j = ln_inv_vars.unsqueeze(0)

        ln2 = torch.log(torch.tensor(2.0, device=mus.device, dtype=mus.dtype))

        muij = (mu_i * torch.exp(ln_inv_var_i) + mu_j * torch.exp(ln_inv_var_j)) / (
            torch.exp(ln_inv_var_i) + torch.exp(ln_inv_var_j)
        )

        ln_vij = ln2 - torch.log(torch.exp(ln_inv_var_i) + torch.exp(ln_inv_var_j))

        ln_A = 0.5 * (
            ln2
            + 0.5 * ln_inv_var_i
            + 0.5 * ln_inv_var_j
            - torch.log(torch.exp(ln_inv_var_i) + torch.exp(ln_inv_var_j))
        )

        ln_B = (
            -0.25
            * ((mu_i - mu_j) ** 2)
            / (torch.exp(-ln_inv_var_i) + torch.exp(-ln_inv_var_j) + eps)
        )

        ln_Sij = ln_A + ln_B
        ln_Sij = torch.sum(ln_Sij, dim=-1).unsqueeze(-1)

        Sij = torch.exp(ln_Sij)

        Mij = muij * Sij

        numerator_mu = (
            torch.sum(mus, dim=0)
            + torch.sum(Mij, dim=(0, 1))
            - Mij.diagonal(offset=0, dim1=0, dim2=1).sum(dim=-1)
        )

        denominator = (
            mus.shape[0]
            + torch.sum(Sij, dim=(0, 1))
            - Sij.diagonal(offset=0, dim1=0, dim2=1).sum(dim=-1)
        )

        joint_mu = numerator_mu / denominator

        Kij = (muij**2 + torch.exp(ln_vij)) * Sij

        numerator_v = (
            torch.sum(mus**2 + torch.exp(logvars), dim=0)
            + torch.sum(Kij, dim=(0, 1))
            - Kij.diagonal(offset=0, dim1=0, dim2=1).sum(dim=-1)
        )

        joint_v = torch.clamp(
            numerator_v / denominator - joint_mu**2,
            min=eps,
        )

        joint_logvar = torch.log(joint_v)

        if ver_type == "f64":
            joint_mu = joint_mu.float()
            joint_logvar = joint_logvar.float()

        return joint_mu, joint_logvar
