# Copyright © Niantic, Inc. 2022.
# pylint: disable=[E1101]

import torch

from conerf.datasets.utils import store_ply


def get_pixel_grid(image_height: int, image_width: int):
    """
    Generate target pixel positions according to image height and width, assuming
    prediction at center pixel.
    """
    ys = torch.arange(image_height, dtype=torch.float32)
    xs = torch.arange(image_width, dtype=torch.float32)
    yy, xx = torch.meshgrid(ys, xs, indexing='ij')

    return torch.stack([xx, yy]) + 0.5


# def get_pixel_grid(subsampling_factor):
#     """
#     Generate target pixel positions according to a subsampling factor, assuming prediction
#     at center pixel.
#     """
#     pix_range = torch.arange(np.ceil(5000 / subsampling_factor), dtype=torch.float32)
#     yy, xx = torch.meshgrid(pix_range, pix_range, indexing='ij')

#     return subsampling_factor * (torch.stack([xx, yy]) + 0.5)


def to_homogeneous(input_tensor, dim=1):
    """
    Converts tensor to homogeneous coordinates by adding ones to the specified dimension
    """
    ones = torch.ones_like(input_tensor.select(dim, 0).unsqueeze(dim))
    output = torch.cat([input_tensor, ones], dim=dim)

    return output


def save_point_cloud(points3d: torch.Tensor, colors: torch.Tensor = None, path: str = ""):
    """Save point cloud to '.ply' file.
    """
    if isinstance(points3d, torch.Tensor):
        points3d = points3d.detach().cpu().numpy()

    if colors is not None:
        if isinstance(colors, torch.Tensor):
            colors = colors.detach().cpu().numpy()
    
    store_ply(path, points3d, colors)
