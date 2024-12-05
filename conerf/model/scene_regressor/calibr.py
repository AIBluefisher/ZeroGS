# pylint: disable=[E1101]

import torch
import torch.nn as nn


class Calibr(nn.Module):
    """
    A modular class for calibration refinement.
    NOTE:
        The class assumes that:
            (1) the principle point is in the center;
            (2) pixels are unskewed and square;
            (3) image distortion is not modelled.
    """

    def __init__(self, device: str = "cuda"):
        super(Calibr, self).__init__()

        self.device = device
        self.scaler = torch.nn.Parameter(torch.tensor(0.), requires_grad=True)

    def forward(self, heights: torch.Tensor, widths: torch.Tensor) -> torch.Tensor:
        batch_size = heights.shape[0]
        # The initial focal length is set to 70% of the image diagonal.
        focal_lengths_init = 0.7 * torch.sqrt(heights ** 2 + widths ** 2)

        # assume principle point is in the center.
        cxs = widths / 2
        cys = heights / 2

        focal_lengths = focal_lengths_init * (1 + self.scaler)

        Ks = torch.eye(3, device=self.device)[None, ...].repeat(batch_size, 1, 1)
        Ks[:, 0, 0] = Ks[:, 1, 1] = focal_lengths
        Ks[:, 0, 2] = cxs
        Ks[:, 1, 2] = cys

        return Ks
