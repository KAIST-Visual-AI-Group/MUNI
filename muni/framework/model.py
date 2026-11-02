"""MUNI multimodal composition root, factory, and EMA utilities."""

import contextlib
from collections.abc import Mapping
from typing import Any

import torch
import torch.nn as nn

from muni.checkpoint_utils import adapt_state_dict_keys
from muni.config_utils import get_obj_from_str
from muni.modules.core import (
    DiagonalGaussianPosterior,
    estimate_standard_normal_kl,
)


def _detach_aggregation_inputs(
    posteriors: dict[str, DiagonalGaussianPosterior],
    stop_grad_modalities: list[str],
) -> dict[str, DiagonalGaussianPosterior]:
    stop_set = set(stop_grad_modalities)
    return {
        modality: (
            DiagonalGaussianPosterior(
                latent_mean=posterior.latent_mean.detach(),
                latent_logvar=posterior.latent_logvar.detach(),
            )
            if modality in stop_set
            else posterior
        )
        for modality, posterior in posteriors.items()
    }


def _posterior_mean_sigreg(
    posterior: DiagonalGaussianPosterior,
    *,
    sketch_dim: int | None = None,
    mean_weight: float = 0.0,
) -> torch.Tensor:
    """Penalize one expert's batch covariance and mean, not its posterior variance."""
    if posterior.latent_mean.numel() == 0:
        return posterior.latent_mean.sum() * 0.0

    mu = posterior.latent_mean.flatten(1).float()
    batch_size, dim = mu.shape

    # The covariance and sketch matmuls must remain float32 under bfloat16 training.
    with torch.amp.autocast(device_type=mu.device.type, enabled=False):
        if sketch_dim is not None and int(sketch_dim) > 0 and dim > int(sketch_dim):
            proj_dim = int(sketch_dim)
            proj = torch.randn(dim, proj_dim, device=mu.device, dtype=mu.dtype)
            proj = torch.nn.functional.normalize(proj, p=2, dim=0, eps=1e-6)
            mu = mu @ proj
            dim = proj_dim

        mu_mean = mu.mean(dim=0, keepdim=True)
        mu_centered = mu - mu_mean
        denom = max(1, batch_size - 1)
        cov_mu = (mu_centered.T @ mu_centered) / float(denom)
        eye = torch.eye(dim, device=mu.device, dtype=mu_centered.dtype)

        # The unnormalized Frobenius penalty targets Cov_batch(mu)=I.
        cov_loss = torch.linalg.matrix_norm(cov_mu - eye, ord="fro")

        mean_loss = mu_mean.square().mean()
    return cov_loss + float(mean_weight) * mean_loss


class EMACheckpointError(RuntimeError):
    """An EMA payload does not cover the model's trainable parameters."""


class ParameterAverage:
    """Unregistered EMA state for the model's current trainable parameters."""

    def __init__(self, decay: float):
        self.decay = decay
        self.names: list[str] = []
        self.shadows: list[torch.Tensor] = []
        self.steps = 0

    @staticmethod
    def trainable(model):
        return ((name, param) for name, param in model.named_parameters() if param.requires_grad)

    @torch.no_grad()
    def initialize(self, model):
        trainable = list(self.trainable(model))
        self.names = [name for name, _ in trainable]
        self.shadows = [param.detach().clone() for _, param in trainable]
        self.steps = 0

    @torch.no_grad()
    def load(self, model, state_dict, device, *, history=None):
        """Load EMA weights and step history from a training checkpoint.

        Without history (evaluation), steps start at zero. Training checkpoints
        without ema_steps start at max(1, int(2 / max(1 - decay, 1e-12))).
        """
        names = [name for name, _ in self.trainable(model)]
        try:
            normalized, info = adapt_state_dict_keys(state_dict, names)
        except RuntimeError as error:
            raise EMACheckpointError(str(error)) from error
        if info["missing"]:
            raise EMACheckpointError(
                f"EMA checkpoint is missing required parameters: {info['missing']}"
            )
        steps = 0
        if history is not None:
            steps = int(history["ema_steps"]) if "ema_steps" in history else None
        self.names = names
        self.shadows = [normalized[name].to(device) for name in names]
        self.steps = (
            max(1, int(2.0 / max(1.0 - self.decay, 1e-12)))
            if steps is None else int(steps)
        )

    def export(self):
        return {
            name: shadow.detach()
            for name, shadow in zip(self.names, self.shadows, strict=True)
        }

    def _apply(self, fn):
        self.shadows = [fn(shadow) for shadow in self.shadows]

    @torch.no_grad()
    def update(self, model):
        trainable = [param for _, param in self.trainable(model)]
        self.steps += 1
        # β_eff(t)=min(β_max, (t-1)/t) avoids initialization bias.
        lerp_w = max(1.0 - self.decay, 1.0 / self.steps)
        torch._foreach_lerp_(self.shadows, trainable, lerp_w)

    @torch.no_grad()
    def install(self, model):
        for (_, param), shadow in zip(self.trainable(model), self.shadows, strict=True):
            param.copy_(shadow)

    @contextlib.contextmanager
    def evaluate(self, model):
        trainable = [param for _, param in self.trainable(model)]
        if len(trainable) != len(self.shadows):
            raise RuntimeError("EMA and live trainable parameter counts differ; call ema.initialize(model) after changing trainability.")
        live = [param.detach().clone() for param in trainable]
        try:
            self.install(model)
            yield
        finally:
            with torch.no_grad():
                for param, live_param in zip(trainable, live):
                    param.copy_(live_param)


class MultimodalWrapper(nn.Module):
    """Compose modality encoders, posterior aggregation, decoders, and prior."""

    def __init__(
        self,
        encoders: nn.ModuleDict,
        aggregator: nn.Module,
        decoders: nn.ModuleDict,
        prior: nn.Module,
        *,
        modalities: list[str],
        ema_decay: float,
        aggregated_kl_weight: float = 0.0,
        aggregated_sigreg_weight: float = 0.0,
        aggregated_sigreg_sketch_dim: int | None = None,
        aggregated_sigreg_mean_weight: float = 0.0,
        private_priors: nn.ModuleDict | None = None,
        private_dim: int = 0,
        shared_dim: int | None = None,
        private_kl_weight: float = 1.0,
    ):
        super().__init__()
        self.encoders = encoders
        self.aggregator = aggregator
        self.decoders = decoders
        self.prior = prior

        self.modalities = list(modalities)

        self.ema_decay = float(ema_decay)
        self.aggregated_kl_weight = float(aggregated_kl_weight)
        self.aggregated_sigreg_weight = float(aggregated_sigreg_weight)
        if aggregated_sigreg_sketch_dim is None:
            self.aggregated_sigreg_sketch_dim = None
        else:
            self.aggregated_sigreg_sketch_dim = int(aggregated_sigreg_sketch_dim)
        self.aggregated_sigreg_mean_weight = float(aggregated_sigreg_mean_weight)

        # MMVAE+ fuses shared latents and routes private latents separately.
        self.private_priors = private_priors
        self.private_dim = int(private_dim)
        self.shared_dim = None if shared_dim is None else int(shared_dim)
        self.private_kl_weight = float(private_kl_weight)

        self.ema = ParameterAverage(self.ema_decay)

    def _apply(self, fn, recurse=True):
        super()._apply(fn, recurse)
        self.ema._apply(fn)
        return self


    def trainable_state_dict(self) -> dict[str, torch.Tensor]:
        """Save trainable parameters and all persistent buffers.

        Frozen foundations are restored externally, but buffers may contain
        random architectural constants and must remain checkpoint-stable.
        """
        # Tied parameter aliases are not persistent buffers.
        param_names = {name for name, _ in self.named_parameters(remove_duplicate=False)}
        trainable_param_names = {name for name, _ in self.ema.trainable(self)}
        return {
            name: tensor
            for name, tensor in self.state_dict().items()
            if (name in trainable_param_names)
            or (name not in param_names)  # persistent buffer
        }

    def load_trainable_state_dict(self, state_dict, *, strict=True):
        """Restore saved model state while leaving frozen foundations in place."""
        adapted, info = adapt_state_dict_keys(state_dict, self.state_dict())
        missing, unexpected = self.load_state_dict(adapted, strict=False)
        if missing:
            parameters = dict(self.named_parameters(remove_duplicate=False))
            loaded_parameters = {
                id(parameters[name]) for name in adapted if name in parameters
            }
            missing = [
                name for name in missing
                if name not in parameters
                or (parameters[name].requires_grad and id(parameters[name]) not in loaded_parameters)
            ]
        if strict and (missing or unexpected or info["unused"]):
            raise RuntimeError(
                f"Incomplete model state: missing={missing}, "
                f"unexpected={unexpected}, unused={info['unused']}"
            )
        return missing, unexpected, info


    def _encode_modalities(
        self,
        dict_data: dict[str, torch.Tensor],
        enc_modalities: list[str],
    ) -> dict[str, DiagonalGaussianPosterior]:
        return {m: self.encoders[m].posterior(dict_data[m]) for m in enc_modalities}


    @staticmethod
    def _append_posterior_diagnostics(
        metrics,
        encoded_posteriors: dict[str, DiagonalGaussianPosterior],
    ):
        if not encoded_posteriors:
            return metrics

        metrics = dict(metrics)
        with torch.no_grad():
            for modality, posterior in encoded_posteriors.items():
                if posterior.latent_mean.numel() == 0:
                    continue
                mean = posterior.latent_mean.flatten(1).float()
                if mean.shape[0] < 2:
                    continue
                variance = posterior.latent_logvar.flatten(1).float().exp()
                mean_variance = mean.var(dim=0, unbiased=True).mean()
                mean_noise_variance = variance.mean()
                metrics[f"_global_post_var_mu/{modality}"] = float(
                    mean_variance.detach()
                )
                metrics[f"_global_post_snr/{modality}"] = float(
                    (mean_variance / mean_noise_variance.clamp(min=1e-12)).detach()
                )
        return metrics

    def forward(
        self,
        dict_data: dict[str, torch.Tensor],
        enc_modalities: list[str],
        tgt_modality: str,
        *,
        stop_grad_modalities: list[str] | None = None,
    ):
        """Compute the routed loss and its posterior regularization terms."""
        encoded_posteriors = self._encode_modalities(dict_data, enc_modalities)

        aggregation_inputs = encoded_posteriors
        if self.private_dim > 0:
            shared_posteriors, private_posteriors = {}, {}
            for modality, posterior in encoded_posteriors.items():
                shared_posteriors[modality], private_posteriors[modality] = self._split_shared_private(posterior)
            aggregation_inputs = shared_posteriors
        # Routing detaches aggregation inputs, not the separate private posteriors.
        if stop_grad_modalities:
            aggregation_inputs = _detach_aggregation_inputs(
                aggregation_inputs, stop_grad_modalities
            )
        aggregate_posterior = self.aggregator(aggregation_inputs, self.modalities)

        if tgt_modality == "prior":
            loss, metrics = self.prior.compute_loss(aggregate_posterior)
        else:
            z_cond = aggregate_posterior.sample()
            if self.private_dim > 0:
                if tgt_modality in enc_modalities:
                    private_sample = private_posteriors[tgt_modality].sample()
                else:
                    latent_extra = tuple(int(size) for size in z_cond.shape[1:-1])
                    private_sample = self.private_priors[tgt_modality].sample(
                        z_cond.shape[0], *latent_extra
                    )
                z_cond = torch.cat([z_cond, private_sample], dim=-1)
            loss, metrics = self.decoders[tgt_modality].compute_loss(
                target=dict_data[tgt_modality],
                z_cond=z_cond,
                available_modalities=list(dict_data.keys()),
            )

        if self.private_dim > 0 and self.private_kl_weight > 0.0 and enc_modalities:
            private_kl_terms = []
            for modality in enc_modalities:
                posterior = private_posteriors[modality]
                latent_numel = max(1, posterior.latent_numel())
                private_kl_terms.append(
                    self.private_priors[modality].kl(posterior).mean() / float(latent_numel)
                )
            # Average the per-dimension private KL across encoded experts.
            private_kl_per_dim = torch.stack(private_kl_terms).mean()
            loss = loss + private_kl_per_dim * self.private_kl_weight
            metrics = dict(metrics)
            metrics["_global_private_kl_per_dim"] = float(private_kl_per_dim.detach())

        if self.aggregated_kl_weight > 0.0:
            aggregate_kl = estimate_standard_normal_kl(aggregate_posterior)
            aggregate_kl_per_dim = aggregate_kl.mean() / max(1, aggregate_posterior.latent_numel())
            loss = loss + aggregate_kl_per_dim * self.aggregated_kl_weight
            metrics = dict(metrics)
            metrics["agg_kl_per_dim"] = float(aggregate_kl_per_dim.detach())

        # The existing aggregated_sigreg_* options control a sum over experts.
        if self.aggregated_sigreg_weight > 0.0:
            expert_mean_penalties = [
                _posterior_mean_sigreg(
                    posterior,
                    sketch_dim=self.aggregated_sigreg_sketch_dim,
                    mean_weight=self.aggregated_sigreg_mean_weight,
                )
                for posterior in aggregation_inputs.values()
            ]
            if expert_mean_penalties:
                expert_mean_penalty = torch.stack(expert_mean_penalties).sum()
                loss = loss + expert_mean_penalty * self.aggregated_sigreg_weight
                metrics = dict(metrics)
                metrics["_global_sigreg"] = expert_mean_penalty.detach().item()

        metrics = self._append_posterior_diagnostics(metrics, encoded_posteriors)
        return loss, metrics

    def shared_posterior(
        self, posterior: DiagonalGaussianPosterior
    ) -> DiagonalGaussianPosterior:
        """Return the prior-compatible shared slice in private-latent mode."""
        if self.private_dim <= 0:
            return posterior
        return DiagonalGaussianPosterior(
            latent_mean=posterior.latent_mean[..., :self.shared_dim],
            latent_logvar=posterior.latent_logvar[..., :self.shared_dim],
        )

    def _split_shared_private(
        self, posterior: DiagonalGaussianPosterior
    ) -> tuple[DiagonalGaussianPosterior, DiagonalGaussianPosterior]:
        D_w = int(self.shared_dim)
        return (
            self.shared_posterior(posterior),
            DiagonalGaussianPosterior(
                latent_mean=posterior.latent_mean[..., D_w:],
                latent_logvar=posterior.latent_logvar[..., D_w:],
            ),
        )

    def _assemble_z_cond(
        self,
        w: torch.Tensor,
        tgt_modality: str,
        *,
        z_private_self: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Combine shared ``w`` with a self posterior or target prior sample."""
        if z_private_self is not None:
            z_tgt = z_private_self
        else:
            B = int(w.shape[0])
            latent_extra = tuple(int(x) for x in w.shape[1:-1])
            z_tgt = self.private_priors[tgt_modality].sample(
                B, *latent_extra, dtype=w.dtype
            )
        return torch.cat([w, z_tgt], dim=-1)

    @torch.no_grad()
    def sample_prior(
        self,
        batch_size: int,
        device: torch.device,
        *,
        sample_kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        """Sample the learned latent prior (shared ``w`` in MMVAE+ mode)."""
        kwargs = dict(sample_kwargs or {})
        kwargs.update(batch_size=batch_size, device=device)
        return self.prior.sample(**kwargs)

    @torch.no_grad()
    def _sample_decoder(
        self,
        tgt_modality: str,
        z_cond: torch.Tensor,
        *,
        sample_kwargs: Mapping[str, Any] | None = None,
        deterministic: bool = False,
    ) -> Any:
        """Sample one decoder, optionally forcing categorical argmax output."""
        kwargs = dict(sample_kwargs or {})
        kwargs["deterministic"] = bool(deterministic)
        return self.decoders[tgt_modality].sample(z_cond=z_cond, **kwargs)

    @torch.no_grad()
    def decode_from_prior(
        self,
        tgt_modality: str,
        z_prior: torch.Tensor,
        *,
        sample_kwargs: Mapping[str, Any] | None = None,
    ) -> torch.Tensor:
        """Decode prior samples, adding a private target sample when required."""
        if self.private_dim > 0:
            z_cond = self._assemble_z_cond(z_prior, tgt_modality)
        else:
            z_cond = z_prior
        return self._sample_decoder(
            tgt_modality,
            z_cond,
            sample_kwargs=sample_kwargs,
            deterministic=False,
        )

    @torch.no_grad()
    def conditional_sample(
        self,
        dict_data: dict[str, torch.Tensor],
        src_modalities: list[str],
        tgt_modality: str,
        *,
        mode: str,
        decoder_sample_kwargs: Mapping[str, Any] | None = None,
        deterministic_encode: bool = False,
    ):
        """Decode an explicit reconstruction, direct, or aggregate route.

        Reconstruction uses the target's private posterior; other modes draw
        its private prior. Aggregate mode fuses even a singleton source.
        Conditional categorical decoders always use deterministic output.
        """
        if mode not in ("reconstruct", "direct", "aggregate"):
            raise ValueError(f"Unknown conditional sampling mode: {mode!r}")
        if mode != "aggregate":
            if len(src_modalities) != 1:
                raise ValueError(f"{mode} requires exactly one source modality")
            if mode == "reconstruct" and src_modalities[0] != tgt_modality:
                raise ValueError("reconstruct requires the target as its source")

        private_post = None
        if mode == "aggregate":
            posteriors = self._encode_modalities(dict_data, src_modalities)
            if self.private_dim > 0:
                posteriors = {
                    m: self.shared_posterior(p) for m, p in posteriors.items()
                }
            posterior = self.aggregator(posteriors, self.modalities)
        else:
            source = src_modalities[0]
            posterior = self.encoders[source].posterior(dict_data[source])
            if self.private_dim > 0:
                posterior, private_post = self._split_shared_private(posterior)

        z = posterior.latent_mean if deterministic_encode else posterior.sample()
        if self.private_dim > 0:
            private_sample = None
            if mode == "reconstruct":
                private_sample = (
                    private_post.latent_mean
                    if deterministic_encode
                    else private_post.sample()
                )
            z = self._assemble_z_cond(
                z, tgt_modality, z_private_self=private_sample
            )
        return self._sample_decoder(
            tgt_modality,
            z,
            sample_kwargs=decoder_sample_kwargs,
            deterministic=True,
        )




def build_model(cfg) -> MultimodalWrapper:
    # The global compile switch and per-module opt-out are both required.
    global_compile = bool(cfg.train.compile_backbones)
    backbones = nn.ModuleDict()
    for name, m_cfg in cfg.model.modules.items():
        # OmegaConf inheritance cannot delete modules; null disables them.
        if m_cfg is None:
            continue
        b_cls = get_obj_from_str(m_cfg["target"])
        backbone = b_cls(**(m_cfg.get("params", {}) or {}))
        per_compile = bool(m_cfg.get("compile", True))
        if global_compile and per_compile:
            backbone = torch.compile(backbone)
        backbones[name] = backbone

    routing = cfg.model.routing
    modalities = list(routing.modalities)

    encoders = nn.ModuleDict()
    for mod in modalities:
        e_conf = routing.encoders[mod]
        e_cls = get_obj_from_str(e_conf.adapter)
        e_backbone = None if e_conf.get("module") is None else backbones[e_conf.module]
        encoders[mod] = e_cls(e_backbone, **(e_conf.get("params", {}) or {}))

    agg_conf = routing.aggregator
    agg_cls = get_obj_from_str(agg_conf.adapter)
    aggregator = agg_cls(**agg_conf.get("params", {}))

    decoders = nn.ModuleDict()
    for mod in modalities:
        d_conf = routing.decoders[mod]
        d_cls = get_obj_from_str(d_conf.adapter)
        params = dict(d_conf.get("params", {}) or {})
        # Resolve auxiliary backbone references such as projector_module.
        for key in list(params):
            if key.endswith("_module") and isinstance(params[key], str):
                name = params.pop(key)
                params[key[: -len("_module")]] = backbones[name]
        decoders[mod] = d_cls(backbones[d_conf.module], **params)

    p_conf = routing.prior
    p_cls = get_obj_from_str(p_conf.adapter)
    p_backbone = None if p_conf.get("module") is None else backbones[p_conf.module]
    prior_params = dict(p_conf.get("params", {}) or {})
    if p_conf.adapter == "muni.modules.flow.FlowPriorAdapter" and "layout" in prior_params:
        # Configs may specify the only supported prior layout, sequence, explicitly.
        if prior_params.pop("layout") != "sequence":
            raise ValueError("FlowPriorAdapter prior layout must be 'sequence'.")
    prior = p_cls(p_backbone, **prior_params)

    private_priors_module: nn.ModuleDict | None = None
    private_dim = 0
    shared_dim_arg: int | None = None
    private_kl_weight = 1.0

    pp_conf = routing.get("private_priors", None)
    if pp_conf is not None:
        private_priors_module = nn.ModuleDict()
        for mod in modalities:
            m_cfg = pp_conf[mod]
            cls = get_obj_from_str(m_cfg.target)
            private_priors_module[mod] = cls(**(m_cfg.get("params", {}) or {}))

        dims = {mod: int(p.dim) for mod, p in private_priors_module.items()}
        unique_dims = set(dims.values())
        if len(unique_dims) != 1:
            raise ValueError(
                f"All ModalityPrivatePrior instances must share the same dim; got {dims}."
            )
        private_dim = unique_dims.pop()

        # shared_dim: private_priors -> routing -> latent_dim - private_dim.
        shared_dim_arg = pp_conf.get("shared_dim", routing.get("shared_dim", None))
        if shared_dim_arg is None:
            shared_dim_arg = int(cfg.model.latent_dim) - private_dim
        shared_dim_arg = int(shared_dim_arg)
        if shared_dim_arg <= 0:
            raise ValueError("MMVAE+ shared_dim must be positive.")

        # KL weight: private_priors.kl_weight -> routing.private_kl_weight -> 1.
        private_kl_weight = float(
            pp_conf.get("kl_weight", routing.get("private_kl_weight", 1.0))
        )

    wrapper = MultimodalWrapper(
        encoders,
        aggregator,
        decoders,
        prior,
        modalities=modalities,
        ema_decay=cfg.train.ema_decay,
        aggregated_kl_weight=float(routing.get("aggregated_kl_weight", 0.0)),
        aggregated_sigreg_weight=float(routing.get("aggregated_sigreg_weight", 0.0)),
        aggregated_sigreg_sketch_dim=routing.get("aggregated_sigreg_sketch_dim", None),
        aggregated_sigreg_mean_weight=float(
            routing.get("aggregated_sigreg_mean_weight", 0.0)
        ),
        private_priors=private_priors_module,
        private_dim=private_dim,
        shared_dim=shared_dim_arg,
        private_kl_weight=private_kl_weight,
    )
    return wrapper
