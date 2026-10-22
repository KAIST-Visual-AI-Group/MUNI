"""Flow-matching decoders and priors for continuous modalities."""

import math
from typing import Any

import torch
import torch.nn as nn
from einops import rearrange

from .core import BaseDecoder, BasePrior, DiagonalGaussianPosterior
from .ops import JiTTokenizer, make_sequence_grid_pos, scale_grad


# Flow matching helpers
def _logit_normal_log_prob(u: torch.Tensor, mu: float, s: float) -> torch.Tensor:
    """Log-density of the logit-normal distribution at u ∈ (0, 1)."""
    logit_u = torch.logit(u)
    log_p = (
        -0.5 * math.log(2.0 * math.pi)
        - math.log(s)
        - torch.log(u)
        - torch.log(1.0 - u)
        - 0.5 * ((logit_u - mu) / s).pow(2)
    )
    return log_p


def _sample_logit_normal(
    batch_size: int,
    device: torch.device,
    p_mean: float = 0.0,
    p_std: float = 1.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    z = torch.randn(batch_size, device=device) * p_std + p_mean
    u = torch.sigmoid(z).clamp(1e-6, 1.0 - 1e-6)
    log_p = _logit_normal_log_prob(u, p_mean, p_std)
    return u, log_p.exp()


def _sample_uniform(
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    u = torch.rand(batch_size, device=device).clamp(1e-6, 1.0 - 1e-6)
    p_u = torch.ones_like(u)
    return u, p_u


def _sample_cubic(
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``u=t³`` with density ``p(u)=u^(-2/3)/3``."""
    t = torch.rand(batch_size, device=device).clamp(1e-6, 1.0 - 1e-6)
    u = t.pow(3.0).clamp(1e-6, 1.0 - 1e-6)
    # Inverse-transform density: p(u) = |dt/du| · p(t) = (1/3) · u^(−2/3).
    p_u = 1.0 / (3.0 * u.pow(2.0 / 3.0))
    return u, p_u


def _sample_cosine(
    batch_size: int,
    device: torch.device,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Sample ``u=2 arcsin(t)/π`` with density ``π cos(πu/2)/2``."""
    t = torch.rand(batch_size, device=device).clamp(1e-6, 1.0 - 1e-6)
    u = ((2.0 / math.pi) * torch.arcsin(t)).clamp(1e-6, 1.0 - 1e-6)
    p_u = (math.pi / 2.0) * torch.cos(math.pi * u / 2.0)
    return u, p_u


class JiTFlowScheduler(nn.Module):
    def __init__(
        self,
        noise_scale: float = 1.0,
        denom_eps: float = 5e-2,
        p_mean: float = 0.0,
        p_std: float = 1.0,
        pred_type: str = "x0",
        timestep_dist: str = "logit_normal",
    ):
        super().__init__()
        self.noise_scale = float(noise_scale)
        if self.noise_scale <= 0.0:
            raise ValueError(f"noise_scale must be positive, got {self.noise_scale}.")
        self.denom_eps = float(denom_eps)
        if self.denom_eps <= 0.0:
            raise ValueError(f"denom_eps must be positive, got {self.denom_eps}.")
        self.p_mean = float(p_mean)
        self.p_std = float(p_std)
        if self.p_std <= 0.0:
            raise ValueError(f"p_std must be positive, got {self.p_std}.")
        if pred_type not in ("x0", "v"):
            raise ValueError(f"pred_type must be 'x0' or 'v', got {pred_type!r}.")
        if timestep_dist not in ("logit_normal", "uniform", "cubic", "cosine"):
            raise ValueError(
                f"timestep_dist must be 'logit_normal', 'uniform', 'cubic', or 'cosine', got {timestep_dist!r}."
            )
        self.pred_type = pred_type
        self.timestep_dist = timestep_dist

    def sample_timestep(self, batch_size: int, device: torch.device):
        if self.timestep_dist == "uniform":
            return _sample_uniform(batch_size, device)
        if self.timestep_dist == "cubic":
            return _sample_cubic(batch_size, device)
        if self.timestep_dist == "cosine":
            return _sample_cosine(batch_size, device)
        return _sample_logit_normal(batch_size, device, self.p_mean, self.p_std)

    def alpha_sigma(self, u: torch.Tensor):
        """Return (alpha, sigma) each of shape (B,)."""
        alpha = u
        sigma = (1.0 - u) * self.noise_scale
        return alpha, sigma

    def noise(self, u: torch.Tensor, clean: torch.Tensor):
        alpha, sigma = self.alpha_sigma(u)
        a = rearrange(alpha, "B -> B 1 1")
        s = rearrange(sigma, "B -> B 1 1")
        eps = torch.randn_like(clean)
        return a * clean + s * eps, eps

    def denoising_state(self, u: torch.Tensor) -> torch.Tensor:
        """Map uniform timestep u (B,) to denoising progress scalar (B,)."""
        return u / (u + (1.0 - u) * self.noise_scale)

    def pred_to_x0(
        self, pred: torch.Tensor, noised: torch.Tensor, u: torch.Tensor
    ) -> torch.Tensor:
        """Convert model output to x0 prediction based on ``pred_type``.

        For ``"x0"`` the model directly predicts x0.
        For ``"v"``  the model predicts velocity v = (x0 - x_u) / (1 - u),
        so x0 = x_u + (1 - u) * v.
        """
        if self.pred_type == "x0":
            return pred
        # v-prediction: x_0 = x_u + (1 - u) * v
        one_minus_u = rearrange((1.0 - u).clamp_min(self.denom_eps), "B -> B 1 1")
        return noised + one_minus_u * pred

    def loss_weights(self, u: torch.Tensor) -> torch.Tensor:
        return torch.ones(u.shape[0], device=u.device, dtype=u.dtype)

    def loss(self, u, noised, pred, true_clean):
        denom = rearrange((1.0 - u).clamp_min(self.denom_eps), "B -> B 1 1")
        w_tok = self.loss_weights(u)
        v_target = (true_clean - noised) / denom
        if self.pred_type == "x0":
            v_pred = (pred - noised) / denom
        else:
            v_pred = pred
        weighted = w_tok.unsqueeze(-1) * (v_target - v_pred).pow(2).mean(dim=-1)
        return weighted.mean(), {}

    @torch.no_grad()
    def compute_denoising_cost(
        self,
        z_target: torch.Tensor,
        grid_pos: torch.Tensor,
        backbone: nn.Module,
        class_label: int,
        u_end: float,
        num_u_steps: int,
    ) -> torch.Tensor:
        """Finite-cutoff weighted denoising cost for the linear schedule.

        ``(1/2) SNR'(u) = u / ((1-u)^3 s^2)``. Midpoint quadrature estimates
        ``integral_0^u_end (1/2) SNR'(u) E[||x_hat_0 - z||^2] du`` using
        one fresh epsilon per node. No endpoint/tail or density-normalization
        terms are included; this model-dependent diagnostic is not an NLL,
        KL, or MI bound. The cutoff controls the omitted clean-end interval.
        """
        if num_u_steps <= 0:
            raise ValueError("num_u_steps must be positive.")
        if u_end < 0.0 or u_end > 1.0:
            raise ValueError(f"u_end must lie in [0, 1], got {u_end}.")

        batch = z_target.shape[0]
        total = torch.zeros(batch, device=z_target.device, dtype=z_target.dtype)
        if z_target.numel() == 0 or u_end == 0.0:
            return total

        delta = torch.tensor(
            u_end / float(num_u_steps), device=z_target.device, dtype=z_target.dtype
        )
        u_nodes = (
            torch.arange(num_u_steps, device=z_target.device, dtype=z_target.dtype)
            + 0.5
        ) * delta
        noise_scale_sq = float(self.noise_scale**2)
        y_tensor = torch.full(
            (batch,), class_label, device=z_target.device, dtype=torch.long
        )

        for u_scalar in u_nodes:
            u_batch = u_scalar.expand(batch)
            noised, _ = self.noise(u_batch, z_target)
            pred = backbone(
                raw=noised,
                grid_pos=grid_pos,
                t=self.denoising_state(u_batch),
                y=y_tensor,
                context=None,
            )
            pred_x0 = self.pred_to_x0(pred, noised, u_batch)
            sq_error = (
                (pred_x0 - z_target).pow(2).sum(dim=tuple(range(1, z_target.ndim)))
            )
            coeff = u_scalar / (
                (1.0 - u_scalar).clamp_min(1e-6).pow(3) * noise_scale_sq
            )
            total = total + delta * coeff * sq_error

        return total


def make_timestep_schedule(
    schedule: str,
    steps: int,
    u_start: float,
    u_end: float,
    device: torch.device,
) -> torch.Tensor:
    """Map an integration grid from ``[0, 1]`` to ``[u_start, u_end]``."""
    steps = int(steps)
    if steps <= 0:
        raise ValueError(f"steps must be positive, got {steps}.")
    if not math.isfinite(float(u_start)) or not math.isfinite(float(u_end)):
        raise ValueError(
            f"u_start and u_end must be finite, got {u_start} and {u_end}."
        )

    t_norm = torch.linspace(0.0, 1.0, steps + 1, device=device)
    if schedule == "linear":
        mapped = t_norm
    elif schedule == "cosine":
        mapped = 0.5 * (1.0 - torch.cos(t_norm * math.pi))
    elif schedule == "quadratic":
        mapped = t_norm**2
    elif schedule == "cubic":
        mapped = t_norm**3
    elif schedule.startswith("power_"):
        power = float(schedule[len("power_") :])
        if not math.isfinite(power) or power <= 0.0:
            raise ValueError(
                f"Power timestep schedule exponent must be positive and finite, got {power}."
            )
        mapped = t_norm**power
    elif schedule == "sqrt":
        mapped = t_norm**0.5
    elif schedule == "log_snr":
        eps = 1e-4
        lo = torch.logit(torch.tensor(eps, device=device))
        hi = torch.logit(torch.tensor(1.0 - eps, device=device))
        mapped = torch.sigmoid(lo + t_norm * (hi - lo))
    else:
        raise ValueError(f"Unknown timestep schedule: {schedule!r}")
    return float(u_start) + (float(u_end) - float(u_start)) * mapped


class FlowSampler:
    """Low-level Euler/Heun sampler for the flow ODE."""

    def __init__(
        self,
        scheduler,
        steps: int = 50,
        cfg_scale: float = 1.0,
        denom_eps: float | None = None,
        method: str = "euler",
        schedule: str = "linear",
        cfg_rescale_phi: float = 0.0,
    ):
        self.scheduler = scheduler
        self.steps = steps
        self.cfg_scale = cfg_scale
        self.denom_eps = scheduler.denom_eps if denom_eps is None else float(denom_eps)
        self.cfg_rescale_phi = float(cfg_rescale_phi)
        self.method = method
        self.schedule = schedule

    def make_timestep_grid(
        self,
        u_start: float,
        u_end: float,
        device: torch.device,
        steps: int | None = None,
        schedule: str | None = None,
    ) -> torch.Tensor:
        """Return a (steps+1,) integration grid in u-space."""
        n = self.steps if steps is None else int(steps)
        sch = self.schedule if schedule is None else schedule
        return make_timestep_schedule(sch, n, u_start, u_end, device)

    def step(
        self,
        z_flow,
        grid_pos,
        u,
        u_next,
        denoiser,
        y,
        y_null,
        z_cond=None,
        method: str | None = None,
    ):
        """Dispatch a single integration step using the requested ``method``."""
        m = self.method if method is None else method
        if m == "heun":
            return self._heun_step(
                z_flow, grid_pos, u, u_next, denoiser, y, y_null, z_cond=z_cond
            )
        if m == "euler":
            return self._euler_step(
                z_flow, grid_pos, u, u_next, denoiser, y, y_null, z_cond=z_cond
            )
        raise ValueError(f"Unknown FlowSampler method: {m!r}")

    def _predict_x0(self, z_flow, grid_pos, u_scalar, denoiser, y, y_null, z_cond=None):
        B = z_flow.shape[0]
        u = u_scalar.expand(B)
        t_input = self.scheduler.denoising_state(u)  # mapping scalar directly
        alpha, sigma = self.scheduler.alpha_sigma(u)

        z_vp = z_flow * rearrange(alpha + sigma, "B -> B 1 1")

        raw_cond = denoiser(z_vp, grid_pos, t_input, y, context=z_cond)
        x0_cond = self.scheduler.pred_to_x0(raw_cond, z_vp, u)
        if self.cfg_scale == 1.0:
            return x0_cond

        # For context-conditioned decoders, proper CFG keeps the same class
        # label and drops only the cross-modal context. When no context is
        # present, use the class-unconditional branch.
        if z_cond is None:
            raw_unc = denoiser(z_vp, grid_pos, t_input, y_null, context=None)
        else:
            raw_unc = denoiser(z_vp, grid_pos, t_input, y, context=None)
        x0_unc = self.scheduler.pred_to_x0(raw_unc, z_vp, u)
        x0_cfg = x0_unc + self.cfg_scale * (x0_cond - x0_unc)
        if self.cfg_rescale_phi <= 0.0:
            return x0_cfg

        dims = tuple(range(1, x0_cfg.ndim))
        std_cond = x0_cond.std(dim=dims, keepdim=True).clamp_min(1e-6)
        std_cfg = x0_cfg.std(dim=dims, keepdim=True).clamp_min(1e-6)
        x0_rescaled = x0_cfg * (std_cond / std_cfg)
        return (
            self.cfg_rescale_phi * x0_rescaled + (1.0 - self.cfg_rescale_phi) * x0_cfg
        )

    def _predict_velocity(
        self, z_flow, grid_pos, u_scalar, denoiser, y, y_null, z_cond=None
    ):
        # Direct τ-space velocity for pred_type='v'. Avoids the v→x0→eps→v
        # round-trip; identity when noise_scale=1, affine in (v_u, z_flow) for
        # general s. For pred_type='x0' falls back to x0 → (z_flow - τx0)/(1-τ).
        if self.scheduler.pred_type != "v":
            B = z_flow.shape[0]
            x0 = self._predict_x0(
                z_flow, grid_pos, u_scalar, denoiser, y, y_null, z_cond=z_cond
            )
            alpha, sigma = self.scheduler.alpha_sigma(u_scalar.expand(B))
            tau = rearrange(alpha / (alpha + sigma), "B -> B 1 1")
            eps = (z_flow - tau * x0) / (1.0 - tau).clamp_min(self.denom_eps)
            return x0 - eps

        B = z_flow.shape[0]
        u = u_scalar.expand(B)
        t_input = self.scheduler.denoising_state(u)
        alpha, sigma = self.scheduler.alpha_sigma(u)
        asum_b = rearrange(alpha + sigma, "B -> B 1 1")
        z_vp = z_flow * asum_b

        raw_cond = denoiser(z_vp, grid_pos, t_input, y, context=z_cond)
        if self.cfg_scale == 1.0:
            return self._v_u_to_v_tau(raw_cond, z_flow, alpha, sigma)

        if z_cond is None:
            raw_unc = denoiser(z_vp, grid_pos, t_input, y_null, context=None)
        else:
            raw_unc = denoiser(z_vp, grid_pos, t_input, y, context=None)
        # The v_u→v_τ map is affine in v_u with a z_flow-only offset, so CFG
        # in v_u space and CFG in v_τ space are equivalent. Pre-mix in v_u
        # space to save one affine application.
        v_u_mix = raw_unc + self.cfg_scale * (raw_cond - raw_unc)
        return self._v_u_to_v_tau(v_u_mix, z_flow, alpha, sigma)

    def _v_u_to_v_tau(self, v_u, z_flow, alpha, sigma):
        # Closed-form conversion derived from x0 = z_vp + (1-u)*v_u and
        # v_τ = (x0 - z_flow)/(1-τ):
        #     v_τ = (α+σ)/s * v_u + (α+σ)*(s-1)/s * z_flow
        # For s=1 this collapses to the identity v_τ = v_u (since α+σ = 1).
        s = self.scheduler.noise_scale
        if abs(s - 1.0) < 1e-9:
            return v_u
        asum = alpha + sigma
        coef_v = rearrange(asum / s, "B -> B 1 1")
        coef_z = rearrange(asum * (s - 1.0) / s, "B -> B 1 1")
        return coef_v * v_u + coef_z * z_flow

    def _euler_step(
        self, z_flow, grid_pos, u, u_next, denoiser, y, y_null, z_cond=None
    ):
        B = z_flow.shape[0]
        v_tau = self._predict_velocity(
            z_flow, grid_pos, u, denoiser, y, y_null, z_cond=z_cond
        )

        alpha_c, sigma_c = self.scheduler.alpha_sigma(u.expand(B))
        alpha_n, sigma_n = self.scheduler.alpha_sigma(u_next.expand(B))
        tau_c = alpha_c / (alpha_c + sigma_c)
        tau_n = alpha_n / (alpha_n + sigma_n)
        d_tau = rearrange(tau_n - tau_c, "B -> B 1 1")
        return z_flow + d_tau * v_tau

    def _heun_step(self, z_flow, grid_pos, u, u_next, denoiser, y, y_null, z_cond=None):
        """2nd-order Heun integrator: averages the slope at t and the trial slope at t+dt."""
        B = z_flow.shape[0]
        v1 = self._predict_velocity(
            z_flow, grid_pos, u, denoiser, y, y_null, z_cond=z_cond
        )

        alpha_c, sigma_c = self.scheduler.alpha_sigma(u.expand(B))
        alpha_n, sigma_n = self.scheduler.alpha_sigma(u_next.expand(B))
        tau_c = alpha_c / (alpha_c + sigma_c)
        tau_n = alpha_n / (alpha_n + sigma_n)
        d_tau = rearrange(tau_n - tau_c, "B -> B 1 1")

        z_trial = z_flow + d_tau * v1
        v2 = self._predict_velocity(
            z_trial, grid_pos, u_next, denoiser, y, y_null, z_cond=z_cond
        )
        return z_flow + d_tau * (v1 + v2) * 0.5


# ── Modality Adapters ──
def _compute_encoder_grad_scale(
    u: torch.Tensor,
    p_u: torch.Tensor,
    noise_scale: float,
    encoder_grad_scale: float,
    max_scale: float = 10.0,
) -> torch.Tensor:
    """Return ``β·clamp(u / ((1-u)·s²·p(u)))`` for encoder-side ELBO gradients."""
    w = u / ((1.0 - u).clamp_min(1e-4) * noise_scale**2 * p_u.clamp_min(1e-4))
    w = w.clamp_max(max_scale)
    return encoder_grad_scale * w


class FlowPriorAdapter(BasePrior):
    """Flow prior over a latent sequence."""

    def __init__(
        self,
        backbone: nn.Module,
        class_label: int,
        noise_scale: float = 1.0,
        encoder_grad_scale: float = 1.0,
        max_grad_scale: float = 10.0,
        scheduler_eps: float = 5e-2,
        sampler_eps: float = 5e-2,
        sampler_endpoint_eps: float = 1e-3,
        diagnostic_endpoint_eps: float = 5e-2,
        p_mean: float = 0.0,
        p_std: float = 1.0,
        sampler_steps: int = 50,
        sampler_method: str = "euler",
        sampler_schedule: str = "linear",
        sampler_temperature: float = 1.0,
        pred_type: str = "x0",
        timestep_dist: str = "logit_normal",
        *,
        seq_len: int,
    ):
        super().__init__()
        self.backbone = backbone
        self.class_label = class_label
        self.seq_len = seq_len
        self.scheduler = JiTFlowScheduler(
            noise_scale=noise_scale,
            denom_eps=scheduler_eps,
            p_mean=p_mean,
            p_std=p_std,
            pred_type=pred_type,
            timestep_dist=timestep_dist,
        )
        self.encoder_grad_scale = float(encoder_grad_scale)
        self.max_grad_scale = float(max_grad_scale)
        self.sampler_eps = float(sampler_eps)
        self.sampler_endpoint_eps = float(sampler_endpoint_eps)
        self.diagnostic_endpoint_eps = float(diagnostic_endpoint_eps)
        self.sampler_steps = int(sampler_steps)
        self.sampler_method = sampler_method
        self.sampler_schedule = sampler_schedule
        self.sampler_temperature = float(sampler_temperature)

        self.register_buffer(
            "_grid_pos", make_sequence_grid_pos(seq_len, torch.device("cpu")),
            persistent=False,
        )

    def compute_loss(
        self,
        posterior: DiagonalGaussianPosterior,
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        z_latent = posterior.sample()
        batch_size = z_latent.shape[0]
        u, p_u = self.scheduler.sample_timestep(batch_size, z_latent.device)
        if z_latent.requires_grad:
            scale = _compute_encoder_grad_scale(
                u,
                p_u,
                self.scheduler.noise_scale,
                self.encoder_grad_scale,
                self.max_grad_scale,
            )
        else:
            scale = 1.0
        z_in = scale_grad(z_latent, scale)
        z_tokens, grid_pos = z_in, self._grid_pos
        noised, _ = self.scheduler.noise(u, z_tokens)
        t_input = self.scheduler.denoising_state(u)
        y = torch.full(
            (batch_size,), self.class_label, device=z_latent.device, dtype=torch.long
        )
        pred = self.backbone(
            raw=noised, grid_pos=grid_pos, t=t_input, y=y, context=None
        )
        flow_loss, _ = self.scheduler.loss(u, noised, pred, z_tokens)

        log_q = posterior.log_prob(z_latent)
        n_latent = max(1, posterior.latent_numel())
        log_q_loss = (log_q.mean() / n_latent).to(dtype=flow_loss.dtype)
        log_q_loss = log_q_loss * self.encoder_grad_scale
        return flow_loss + log_q_loss, {}

    @torch.no_grad()
    def compute_denoising_cost(
        self,
        posterior: DiagonalGaussianPosterior,
        num_u_steps: int = 16,
    ) -> torch.Tensor:
        z_target = posterior.sample()
        grid_pos = self._grid_pos
        u_end = max(0.0, min(1.0, 1.0 - self.diagnostic_endpoint_eps))
        return self.scheduler.compute_denoising_cost(
            z_target=z_target,
            grid_pos=grid_pos,
            backbone=self.backbone,
            class_label=self.class_label,
            u_end=u_end,
            num_u_steps=num_u_steps,
        )

    @torch.no_grad()
    def sample(
        self,
        batch_size: int = 1,
        device: torch.device | None = None,
        sampler_steps: int | None = None,
        endpoint_eps: float | None = None,
        method: str | None = None,
        schedule: str | None = None,
        temperature: float | None = None,
        sampler_eps: float | None = None,
    ) -> torch.Tensor:
        if device is None:
            device = next(self.parameters()).device
        y = torch.full((batch_size,), self.class_label, device=device, dtype=torch.long)
        y_null = torch.full_like(y, self.backbone.num_classes)

        steps = self.sampler_steps if sampler_steps is None else int(sampler_steps)
        eps = self.sampler_endpoint_eps if endpoint_eps is None else float(endpoint_eps)
        u_end = min(1.0, max(0.0, 1.0 - eps))
        method_eff = self.sampler_method if method is None else method
        schedule_eff = self.sampler_schedule if schedule is None else schedule
        temperature_eff = (
            self.sampler_temperature if temperature is None else float(temperature)
        )
        sampler_eps_eff = (
            self.sampler_eps if sampler_eps is None else float(sampler_eps)
        )

        d = self.backbone.token_dim
        seq_len, grid_pos = self.seq_len, self._grid_pos
        z_flow = torch.randn(batch_size, seq_len, d, device=device) * temperature_eff
        sampler = FlowSampler(
            self.scheduler,
            steps=steps,
            denom_eps=sampler_eps_eff,
            method=method_eff,
            schedule=schedule_eff,
        )
        ts = sampler.make_timestep_grid(0.0, u_end, device)
        for i in range(steps):
            z_flow = sampler.step(
                z_flow, grid_pos, ts[i], ts[i + 1], self.backbone, y, y_null
            )
        return z_flow


class FlowDecoderAdapter(BaseDecoder):
    """Conditional flow decoder for images or length-one feature sequences."""

    def __init__(
        self,
        backbone: nn.Module,
        class_label: int,
        noise_scale: float = 1.0,
        img_size: int | tuple[int, int] | list[int] = 256,
        patch_size: int = 16,
        in_channels: int = 3,
        encoder_grad_scale: float = 1.0,
        p_mean: float = 0.0,
        p_std: float = 1.0,
        sampler_steps: int = 50,
        sampler_method: str = "heun",
        sampler_schedule: str = "linear",
        sampler_temperature: float = 1.0,
        sampler_cfg_scale: float = 1.0,
        scheduler_eps: float = 5e-2,
        sampler_eps: float = 5e-2,
        pred_type: str = "x0",
        timestep_dist: str = "logit_normal",
        *,
        layout: str,
    ):
        super().__init__()
        if layout not in ("image", "feature"):
            raise ValueError("layout must be 'image' or 'feature'")
        self.layout = layout
        self.backbone = backbone
        self.class_label = int(class_label) if layout == "feature" else class_label
        if layout == "image":
            patch_size = int(patch_size)
            in_channels = int(in_channels)
            if in_channels <= 0:
                raise ValueError("in_channels must be positive")
            self.tokenizer = JiTTokenizer(
                img_size=img_size,
                patch_size=patch_size,
                in_channels=in_channels,
            )
        self.scheduler = JiTFlowScheduler(
            noise_scale=noise_scale,
            denom_eps=scheduler_eps,
            p_mean=p_mean,
            p_std=p_std,
            pred_type=pred_type,
            timestep_dist=timestep_dist,
        )
        self.encoder_grad_scale = (
            float(encoder_grad_scale) if layout == "feature" else encoder_grad_scale
        )
        self.sampler_steps = int(sampler_steps) if layout == "feature" else sampler_steps
        self.sampler_method = sampler_method
        self.sampler_schedule = sampler_schedule
        self.sampler_temperature = float(sampler_temperature)
        self.sampler_cfg_scale = float(sampler_cfg_scale)
        self.sampler_cfg_rescale_phi = 0.0
        self.sampler_eps = float(sampler_eps) if layout == "feature" else sampler_eps

    def compute_loss(
        self, target: torch.Tensor, z_cond: torch.Tensor, **kwargs
    ) -> tuple[torch.Tensor, dict[str, Any]]:
        z_in = scale_grad(z_cond, self.encoder_grad_scale)
        if self.layout == "image":
            target_raw, grid_pos = self.tokenizer.tokenize(target)
        else:
            target_raw, grid_pos = target.unsqueeze(1), None
        batch_size = target_raw.shape[0]
        u, _ = self.scheduler.sample_timestep(batch_size, target_raw.device)
        noised, _ = self.scheduler.noise(u, target_raw)
        if grid_pos is None:
            grid_pos = make_sequence_grid_pos(1, target_raw.device)
        t_input = self.scheduler.denoising_state(u)
        y = torch.full(
            (batch_size,), self.class_label, device=target.device, dtype=torch.long
        )
        pred = self.backbone(
            raw=noised, grid_pos=grid_pos, t=t_input, y=y, context=z_in
        )
        loss, metrics = self.scheduler.loss(u, noised, pred, target_raw)
        return loss, metrics

    @torch.no_grad()
    def sample(
        self,
        z_cond: torch.Tensor,
        cfg_scale: float | None = None,
        sampler_steps: int | None = None,
        method: str | None = None,
        schedule: str | None = None,
        temperature: float | None = None,
        init_noise: torch.Tensor | None = None,
        sampler_eps: float | None = None,
        cfg_rescale_phi: float | None = None,
        **kwargs,
    ) -> torch.Tensor:
        """Integrate to exactly u=1, using Euler for the terminal step."""
        if self.layout == "image":
            seq_len, grid_pos = self.tokenizer.seq_len, self.tokenizer.grid_pos
        else:
            seq_len, grid_pos = 1, None
        device = z_cond.device
        batch_size = z_cond.shape[0]
        y = torch.full(
            (batch_size,), self.class_label, device=device, dtype=torch.long
        )
        y_null = torch.full_like(y, self.backbone.num_classes)
        cfg_scale_eff = self.sampler_cfg_scale if cfg_scale is None else float(cfg_scale)
        steps = self.sampler_steps if sampler_steps is None else int(sampler_steps)
        method_eff = self.sampler_method if method is None else method
        schedule_eff = self.sampler_schedule if schedule is None else schedule
        temperature_eff = (
            self.sampler_temperature if temperature is None else float(temperature)
        )
        sampler_eps_eff = (
            self.sampler_eps if sampler_eps is None else float(sampler_eps)
        )
        cfg_rescale_eff = (
            self.sampler_cfg_rescale_phi
            if cfg_rescale_phi is None
            else float(cfg_rescale_phi)
        )

        expected_shape = (batch_size, seq_len, self.backbone.token_dim)
        if init_noise is None:
            z_flow = torch.randn(*expected_shape, device=device) * temperature_eff
        else:
            if tuple(init_noise.shape) != expected_shape:
                raise ValueError(
                    f"init_noise must have shape {expected_shape}, got {tuple(init_noise.shape)}."
                )
            z_flow = init_noise.to(device=device, dtype=z_cond.dtype).clone()
        if grid_pos is None:
            grid_pos = make_sequence_grid_pos(seq_len, device)

        sampler = FlowSampler(
            self.scheduler,
            steps=steps,
            cfg_scale=cfg_scale_eff,
            cfg_rescale_phi=cfg_rescale_eff,
            denom_eps=sampler_eps_eff,
            method=method_eff,
            schedule=schedule_eff,
        )
        ts = sampler.make_timestep_grid(0.0, 1.0, device)
        for i in range(steps - 1):
            z_flow = sampler.step(
                z_flow, grid_pos, ts[i], ts[i + 1], self.backbone, y, y_null,
                z_cond=z_cond,
            )
        # Also covers the one-step case. Heun cannot evaluate velocity at u=1.
        z_flow = sampler.step(
            z_flow, grid_pos, ts[-2], ts[-1], self.backbone, y, y_null,
            z_cond=z_cond, method="euler",
        )
        if self.layout == "image":
            return self.tokenizer.decode(z_flow)
        return z_flow.squeeze(1)
