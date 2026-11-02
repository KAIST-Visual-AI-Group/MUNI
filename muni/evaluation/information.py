"""Finite-cutoff prior denoising diagnostics minus posterior entropy."""

import torch
import muni.util.misc as misc


class PriorDenoisingCostEvaluator:
    """Average denoising cost minus entropy; not an NLL, KL, or MI bound."""

    def __init__(
        self,
        num_samples: int = 5000,
        num_u_steps: int = 16,
    ):
        if num_samples <= 0:
            raise ValueError("num_samples must be positive.")

        self.num_samples = int(num_samples)
        self.num_u_steps = int(num_u_steps)
        if self.num_u_steps <= 0:
            raise ValueError("num_u_steps must be positive.")

    @torch.no_grad()
    def evaluate(
        self,
        model,
        data_loader,
        device: torch.device,
        modality: str,
        *,
        num_samples: int | None = None,
    ) -> dict[str, torch.Tensor]:
        local_num_samples = (
            self.num_samples if num_samples is None else int(num_samples)
        )

        total_cost_minus_entropy = torch.zeros((), device=device, dtype=torch.float64)
        total_per_dim = torch.zeros((), device=device, dtype=torch.float64)
        total_samples = 0

        if local_num_samples == 0:
            return {
                "cost_minus_entropy_sum": total_cost_minus_entropy,
                "cost_minus_entropy_per_dim_sum": total_per_dim,
                "count": torch.zeros((), device=device, dtype=torch.long),
            }

        for batch in data_loader:
            if total_samples >= local_num_samples:
                break

            dict_data = misc.extract_dict_data(batch)
            if modality not in dict_data:
                continue  # pair-group batch missing this modality
            inputs = misc.move_to_device(dict_data[modality], device)

            if isinstance(inputs, dict):
                ref = next(iter(inputs.values()))
            else:
                ref = inputs
            remaining = local_num_samples - total_samples
            if remaining < int(ref.shape[0]):
                if isinstance(inputs, dict):
                    inputs = {k: v[:remaining] for k, v in inputs.items()}
                else:
                    inputs = inputs[:remaining]

            posterior = model.encoders[modality].posterior(inputs)
            # MMVAE+ priors model only the shared posterior slice.
            posterior = model.shared_posterior(posterior)
            cost_minus_entropy, per_dim = model.prior.compute_denoising_cost_minus_entropy(
                posterior,
                num_u_steps=self.num_u_steps,
            )

            total_cost_minus_entropy += cost_minus_entropy.sum().to(dtype=torch.float64)
            total_per_dim += per_dim.sum().to(dtype=torch.float64)
            total_samples += int(cost_minus_entropy.shape[0])

        return {
            "cost_minus_entropy_sum": total_cost_minus_entropy,
            "cost_minus_entropy_per_dim_sum": total_per_dim,
            "count": torch.tensor(total_samples, device=device, dtype=torch.long),
        }
