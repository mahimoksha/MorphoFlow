"""Latent prior of MorphoFlow: an autoregressive rational-quadratic spline flow whose base
distribution is a zero-mean Gaussian with learnable per-dimension scales (automatic relevance
determination, ARD).
"""
import math

import torch
from nflows.distributions.base import Distribution
from nflows.flows import Flow
from nflows.transforms import CompositeTransform
from nflows.transforms.autoregressive import \
    MaskedPiecewiseRationalQuadraticAutoregressiveTransform

# Lower bound for the log standard deviations of the ARD base distribution
MIN_LOG_SCALE = -8.0


class LearnableDiagonalNormal(Distribution):
    """N(0, diag(sigma^2)) with learnable log(sigma)."""
    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim
        self.register_buffer('loc', torch.zeros(dim))
        self.log_scale = torch.nn.Parameter(torch.zeros(dim))

    def log_prob(self, inputs, context=None):
        log_scale = torch.clamp(self.log_scale, min=MIN_LOG_SCALE)
        var = torch.exp(log_scale) ** 2
        return -0.5 * (
            (inputs ** 2) / (var + 1e-8)
            + 2 * log_scale
            + math.log(2.0 * math.pi)
        ).sum(dim=-1)

    def sample(self, num_samples, context=None):
        eps = torch.randn(num_samples, self.dim, device=self.log_scale.device)
        return eps * torch.exp(self.log_scale)


def build_ar_spline_flow_with_ard(latent_dim: int, num_layers: int = 4,
                                  hidden_features: int = 128, num_bins: int = 8,
                                  tail_bound: float = 3.0) -> Flow:
    transforms = [
        MaskedPiecewiseRationalQuadraticAutoregressiveTransform(
            features=latent_dim,
            hidden_features=hidden_features,
            num_bins=num_bins,
            tails='linear',
            tail_bound=tail_bound
        )
        for _ in range(num_layers)
    ]
    return Flow(CompositeTransform(transforms), LearnableDiagonalNormal(latent_dim))


def ard_gamma_hyperprior(log_scale: torch.Tensor, gamma_a: float,
                         gamma_b: float) -> torch.Tensor:
    """Negative log of a Gamma(a, b) hyperprior on the ARD precisions alpha = sigma^-2 (up to a
    constant), summed over latent dimensions."""
    alpha = torch.exp(-2.0 * log_scale)
    return torch.sum(-(gamma_a - 1.0) * torch.log(alpha + 1e-8) + gamma_b * alpha)


def num_effective_dims(log_scale: torch.Tensor, fraction: float) -> int:
    """Smallest number of ARD dimensions that explain `fraction` of the total prior variance."""
    sigma2_sorted = torch.sort(torch.exp(log_scale.detach().cpu()) ** 2, descending=True)[0]
    ratio = torch.cumsum(sigma2_sorted, dim=0) / sigma2_sorted.sum()
    return int((ratio >= fraction).nonzero()[0].item()) + 1
