"""Task-independent TensorBoard evaluation loggers."""

import math

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402 — must follow backend selection
import numpy as np
import torch
import torchvision.utils as vutils
from einops import rearrange


def _square_nrow(n, h, w):
    """Compute nrow so the final grid is close to square in pixel resolution."""
    return max(1, min(n, round(math.sqrt(n * h / w))))


class BaseLogger:
    def log(self, writer, step, preds, targets, prefix_name, **kwargs):
        raise NotImplementedError


class ImageGridLogger(BaseLogger):
    """Generic mapping for unconditioned generative arrays bounding variables correctly."""

    def __init__(self, num_vis_samples: int = 16, **kwargs):
        self.num_vis_samples = num_vis_samples

    def log(self, writer, step, preds, targets, prefix_name, **kwargs):
        B = preds.shape[0]
        n = min(B, self.num_vis_samples)
        h, w = preds.shape[2], preds.shape[3]
        grid = vutils.make_grid(preds[:n].clamp(-1, 1) * 0.5 + 0.5, nrow=_square_nrow(n, h, w), normalize=False)
        writer.add_image(f"{prefix_name}/grid", grid, step)


class ImageTripletLogger(BaseLogger):
    """Pairs Image targets identically visually aligning original inputs, generated outcomes, and residual losses."""

    def __init__(self, num_vis_samples: int = 8, **kwargs):
        self.num_vis_samples = num_vis_samples

    def log(self, writer, step, preds, targets, prefix_name, **kwargs):
        B = preds.shape[0]
        n = min(B, self.num_vis_samples, int(targets.shape[0]))

        gt = targets[:n] * 0.5 + 0.5
        recon = preds[:n] * 0.5 + 0.5
        err = torch.abs(gt - recon)

        combo = torch.cat([gt, recon, err], dim=0)
        grid = vutils.make_grid(combo, nrow=n, normalize=False)
        writer.add_image(f"{prefix_name}/triplet", grid, step)


class ClassHistogramLogger(BaseLogger):
    """Outputs categorical distributions as bar charts with class labels."""

    def __init__(self, class_labels: list[str] | None = None, **kwargs):
        self.class_labels = class_labels

    def log(self, writer, step, preds, targets, prefix_name, **kwargs):
        preds_np = preds.flatten().cpu().numpy().astype(int)
        num_classes = len(self.class_labels) if self.class_labels else int(max(preds_np.max(), 0) + 1)
        if self.class_labels is None:
            self.class_labels = [str(i) for i in range(num_classes)]

        x = np.arange(num_classes)
        width = 0.35
        fig, ax = plt.subplots(figsize=(max(6, num_classes * 0.8), 4))

        pred_counts = np.bincount(preds_np, minlength=num_classes)[:num_classes].astype(float)
        pred_dist = pred_counts / max(pred_counts.sum(), 1)

        if targets is not None:
            tgts_np = targets.flatten().cpu().numpy().astype(int)
            gt_counts = np.bincount(tgts_np, minlength=num_classes)[:num_classes].astype(float)
            gt_dist = gt_counts / max(gt_counts.sum(), 1)
            ax.bar(x - width / 2, gt_dist, width, alpha=0.7, label="GT")
            ax.bar(x + width / 2, pred_dist, width, alpha=0.7, label="Pred")
            l1_dist = float(np.abs(gt_dist - pred_dist).sum())
            ax.set_title(f"L1 Distance: {l1_dist:.4f}", fontsize=10)
        else:
            ax.bar(x, pred_dist, width, alpha=0.7, label="Pred")

        ax.set_xticks(x)
        ax.set_xticklabels(self.class_labels, rotation=45, ha="right", fontsize=8)
        ax.set_ylabel("Proportion")
        ax.legend(loc="upper right")
        fig.tight_layout()

        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)  # type: ignore
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        plt.close(fig)

        img_t = rearrange(torch.from_numpy(img), "H W C -> C H W")
        writer.add_image(f"{prefix_name}/histogram", img_t, step)


class LabeledImageGridLogger(BaseLogger):
    """Image grid with per-image text labels derived from source class indices."""

    def __init__(self, class_labels: list[str] | None = None, num_vis_samples: int = 16, **kwargs):
        self.class_labels = class_labels
        self.num_vis_samples = num_vis_samples

    def log(self, writer, step, preds, targets, prefix_name, sources=None, **kwargs):
        B = preds.shape[0]
        n = min(B, self.num_vis_samples)
        images = preds[:n].clamp(-1, 1) * 0.5 + 0.5

        labels = []
        if sources is not None:
            src = sources[:n].reshape(-1).cpu().tolist()
            if self.class_labels:
                labels = [self.class_labels[int(i)] for i in src]
            else:
                labels = [str(int(i)) for i in src]

        h, w = images.shape[2], images.shape[3]
        ncols = _square_nrow(n, h, w)
        nrows = math.ceil(n / ncols)
        fig, axes = plt.subplots(nrows, ncols, figsize=(ncols * 2.5, nrows * 2.8))
        axes = np.atleast_2d(np.asarray(axes, dtype=object)).flatten()

        for i in range(n):
            img_np = images[i].permute(1, 2, 0).cpu().numpy().clip(0, 1)
            axes[i].imshow(img_np)
            axes[i].axis("off")
            if i < len(labels):
                axes[i].set_title(labels[i], fontsize=10, fontweight="bold")

        for i in range(n, len(axes)):
            axes[i].set_visible(False)

        fig.tight_layout()
        fig.canvas.draw()
        img = np.frombuffer(fig.canvas.buffer_rgba(), dtype=np.uint8)
        img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))[..., :3]
        plt.close(fig)

        img_t = rearrange(torch.from_numpy(img.copy()), "H W C -> C H W")
        writer.add_image(f"{prefix_name}/grid", img_t, step)

