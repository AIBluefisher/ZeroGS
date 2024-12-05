# pylint: disable=E1101

import math
import torch
import torch.nn as nn
import numpy as np

from typing import List, Callable, Dict


class SinusoidalEncoder(nn.Module):
    """Sinusoidal Positional Encoder used in Nerf."""

    def __init__(self, x_dim, min_deg, max_deg, use_identity: bool = True):
        super().__init__()
        self.x_dim = x_dim
        self.min_deg = min_deg
        self.max_deg = max_deg
        self.use_identity = use_identity
        self.c2f = None
        self.register_buffer(
            "scales", torch.tensor([2**i for i in range(min_deg, max_deg)])
        )

    @property
    def latent_dim(self) -> int:
        return (
            int(self.use_identity) + (self.max_deg - self.min_deg) * 2
        ) * self.x_dim

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: [..., x_dim]
        Returns:
            latent: [..., latent_dim]
        """
        if self.max_deg == self.min_deg:
            return x
        xb = torch.reshape(
            (x[Ellipsis, None, :] * self.scales[:, None]),
            list(x.shape[:-1]) + [(self.max_deg - self.min_deg) * self.x_dim],
        )
        latent = torch.sin(torch.cat([xb, xb + 0.5 * math.pi], dim=-1))
        if self.use_identity:
            latent = torch.cat([x] + [latent], dim=-1)
        return latent


class ProgressiveSinusoidalEncoder(SinusoidalEncoder):
    """
    Coarse-to-fine positional encodings.
    """

    def __init__(
        self,
        x_dim: int,
        min_deg: int,
        max_deg: int,
        use_identity: bool = True,
        c2f: List = [0.1, 0.5],
        half_dim: bool = False,
    ):
        super().__init__(x_dim, min_deg, max_deg, use_identity)

        # Use nn.Parameter so it could be checkpointed.
        self.progress = torch.nn.Parameter(torch.tensor(0.), requires_grad=False)

        self.c2f = c2f
        self.half_dim = half_dim

    @property
    def latent_dim(self) -> int:
        latent_dim = super().latent_dim
        if self.half_dim:
            latent_dim = (latent_dim - self.x_dim) // 2 + self.x_dim
        return latent_dim

    def anneal(
        self,
        iteration: int,
        max_iteration: int,
        factor: float = 1.0,
        reduction: float = 0.0,
        bias: float = 0.0,
        anneal_surface: bool = False,
    ):
        """
        Gradually increase the controllable parameter during training.
        """
        if anneal_surface:
            if iteration > max_iteration // 2:
                progress_data = 1.0
            else:
                progress_data = 0.5 + float(iteration) / float(max_iteration)
        else:
            # For camera pose annealing.
            progress_data = float(iteration) / float(max_iteration)

        progress_data = factor * (progress_data - reduction) + bias

        self.progress.data.fill_(progress_data)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        latent = super().forward(x)
        latent_dim = super().latent_dim

        # Computing weights.
        start, end = self.c2f
        alpha = (self.progress.data - start) / (end - start) * self.max_deg
        ks = torch.arange(self.min_deg, self.max_deg,
                          dtype=torch.float32, device=x.device)
        weight = (
            1.0 - (alpha - ks).clamp_(min=0, max=1).mul_(np.pi).cos_()
        ) / 2.0

        # Apply weight to positional encodings.
        shape = latent.shape
        L = self.max_deg - self.min_deg

        if self.use_identity:
            latent_freq = latent[:, self.x_dim:].reshape(-1, L)
            latent_freq = (
                latent_freq * weight).reshape(shape[0], shape[-1] - self.x_dim)
            latent[:, self.x_dim:] = latent_freq
        else:
            latent = (latent.reshape(-1, L) * weight).reshape(*shape)

        if self.half_dim:
            half_freq = L // 2
            # input coordinates are excluded.
            half_latent_dim = (latent_dim - self.x_dim) // 2
            num_feat_each_band = (latent_dim - self.x_dim) // L
            half_latent = latent[:, self.x_dim:].view(-1, L, num_feat_each_band)[
                :, :half_freq, :].view(-1, half_latent_dim)

            half_latent_contg = latent[:, self.x_dim:].view(-1, L, num_feat_each_band)[
                :, :half_freq, :].view(-1, half_latent_dim).contiguous()
            half_latent_contg = (
                half_latent_contg.view(-1, half_freq) * weight[:half_freq]
            ).view(-1, half_latent_dim)
            flag = weight[:half_freq].tile(shape[0], num_feat_each_band, 1).transpose(
                1, 2).contiguous().view(-1, half_latent_dim)
            half_latent = torch.where(
                flag > 0.01, half_latent, half_latent_contg)
            latent = torch.cat([latent[:, :self.x_dim], half_latent], dim=-1)

        return latent


class GaussianEncoder(nn.Module):
    """
    Gaussian encodings.
    """

    def __init__(
        self,
        x_dim: int,
        feature_dim: int,
        init_func: Callable = nn.init.uniform_,
        init_range: float = 0.1,
        sigma: float = 0.1,
    ) -> None:
        super().__init__()

        self.init_func = init_func
        self.init_range = init_range
        self.sigma = sigma
        self.sigma_square = sigma ** 2
        self.latent_dim = feature_dim

        gaussian_linear = torch.nn.Linear(x_dim, feature_dim)
        self.init_func(gaussian_linear.weight, -
                       self.init_range, self.init_range)
        self.gaussian_linear = nn.utils.weight_norm(gaussian_linear)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.gaussian_linear(x)
        mu = torch.mean(x, axis=-1).unsqueeze(-1)
        x = torch.exp(
            -0.5 * ((x - mu) ** 2) / self.sigma_square
        )
        return x


def create_encoder(x_dim: int, config: Dict):
    """
    Factory function for creating encodings that applied to coordinate input.
    """
    encoder_type = config["type"]
    if encoder_type == "sinusoidal":
        return SinusoidalEncoder(
            x_dim=x_dim,
            min_deg=config["min_deg"],
            max_deg=config["max_deg"],
            use_identity=config["use_identity"],
        )
    elif encoder_type == "progressive":
        return ProgressiveSinusoidalEncoder(
            x_dim=x_dim,
            min_deg=config["min_deg"],
            max_deg=config["max_deg"],
            use_identity=config["use_identity"],
            c2f=config["c2f"],
            half_dim=config["half_dim"],
        )
    elif encoder_type == "gaussian":
        return GaussianEncoder(
            x_dim=x_dim,
            feature_dim=config["feature_dim"] // 2,
            init_range=config["init_range"],
            sigma=config["sigma"],
        )
    else:
        raise NotImplementedError
