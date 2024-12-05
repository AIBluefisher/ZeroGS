import torch
import torch.nn as nn


class Gaussian(nn.Module):
    """
    Gaussian activation function.
    """
    def __init__(
        self,
        mean: float = 0.0,
        sigma: float = 0.1,
    ) -> None:
        super().__init__()
        
        self.mean = mean
        self.sigma = sigma
        self.sigma_square = self.sigma ** 2

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return ( # torch.exp(
            -0.5 * (x ** 2) / self.sigma_square
        ).exp()
