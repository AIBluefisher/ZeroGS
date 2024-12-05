# pylint: disable=[E1101]

import os
import math
import copy

import torch
import torch.nn as nn
import torchvision.transforms as transforms

from pytorch3d.transforms.se3 import se3_exp_map
from packaging import version as pver

from conerf.geometry.pose_util import projection_matrix


def custom_meshgrid(*args):
    """
    Hack the `meshgrid' function in torch with version lower than 1.10
    """
    # ref: https://pytorch.org/docs/stable/generated/torch.meshgrid.html?highlight=meshgrid#torch.meshgrid
    if pver.parse(torch.__version__) < pver.parse('1.10'):
        return torch.meshgrid(*args)
    else:
        return torch.meshgrid(*args, indexing='ij')


def meshgrid(B: int, H: int, W: int, dtype: torch.dtype, device: str, normalized: bool = False):
    """ Create meshgrid with a specific resolution
    Parameters:
        B: batch size
        H: image height
        W: image width
        dtype: meshgrid type
        device: meshgrid device
        normalized: True if grid is normalized between -1 and 1
    Return:
        xs: torch.Tensor [B,1,W]
        ys: torch.Tensor [B,H,1]
    """
    if normalized:
        xs = torch.linspace(-1, 1, W, device=device, dtype=dtype)
        ys = torch.linspace(-1, 1, H, device=device, dtype=dtype)
    else:
        xs = torch.linspace(0, W-1, W, device=device, dtype=dtype)
        ys = torch.linspace(0, H-1, H, device=device, dtype=dtype)
    ys, xs = custom_meshgrid([ys, xs]) # torch.meshgrid([ys, xs])
    return xs.repeat([B, 1, 1]), ys.repeat([B, 1, 1])


def image_grid(B: int, H: int, W: int, dtype: torch.dtype, device: str, normalized: bool = False):
    """ Create an image grid with a specific resolution.
    Parameters:
        B: batch size
        H: image height
        W: image width
        dtype: meshgrid type
        device: meshgrid device
        normalized: True if grid is normalized between -1 and 1
    
    Returns:
        grid: torch.Tensor [B,3,H,W] Image grid containing a meshgrid in x, y and 1
    """
    xs, ys = meshgrid(B, H, W, dtype, device, normalized=normalized)
    ones = torch.ones_like(xs)
    grid = torch.stack([xs + 0.5, ys + 0.5, ones], dim=1)
    return grid


def scale_intrinsics(K, x_scale, y_scale):
    """Scale intrinsics given x_scale and y_scale factors."""
    K[..., 0, 0] *= x_scale
    K[..., 1, 1] *= y_scale
    K[..., 0, 2] = (K[..., 0, 2] + 0.5) * x_scale - 0.5
    K[..., 1, 2] = (K[..., 1, 2] + 0.5) * y_scale - 0.5
    return K


def focal_length_to_fov(focal_length, pixels):
    return 2 * math.atan(pixels / (2 * focal_length))


class Camera(nn.Module):
    def __init__(
        self,
        image_index: int,
        world_to_camera: torch.Tensor,
        fx: float,
        fy: float,
        cx: float,
        cy: float,
        image: torch.Tensor,
        width: int = None,
        height: int = None,
        normal: torch.Tensor = None,
        device: str = "cuda:0",
    ) -> None:
        super().__init__()

        assert image is not None or (width is not None and height is not None)

        self.device = device
        self.image_index = image_index
        self.image = (copy.copy(image) / 255.0).clamp(0.0, 1.0).to(device) \
            if image is not None else None
        self.normal = None
        if normal is not None:
            self.normal = (copy.copy(normal) / 255.0).clamp(0.0, 1.0).to(device)

        if self.image is not None:
            self.width = self.image.shape[1]
            self.height = self.image.shape[0]
        else:
            self.width = width
            self.height = height

        self.fx = fx
        self.fy = fy
        self.cx = cx
        self.cy = cy
        self.fov_x = focal_length_to_fov(self.fx, self.width)
        self.fov_y = focal_length_to_fov(self.fy, self.height)

        self.znear = 0.01
        self.zfar = 100.0

        self.R = world_to_camera[:3, :3].to(device)
        self.t = world_to_camera[:3,  3].to(device)
        self.world_to_camera = copy.copy(world_to_camera).transpose(0, 1).to(device)
        self.projection_matrix = projection_matrix(
            self.znear, self.zfar, self.fov_x, self.fov_y).transpose(0, 1).to(device)
        self.projective_matrix = self.world_to_camera @ self.projection_matrix
        self.camera_center = self.world_to_camera.inverse()[3, :3]

        # optimizable parameters.
        self.delta_rot = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )
        self.delta_trans = nn.Parameter(
            torch.zeros(3, requires_grad=True, device=device)
        )

        # self.exposure_a = nn.Parameter(
        #     torch.tensor([0.0], requires_grad=True, device=device)
        # )
        # self.exposure_b = nn.Parameter(
        #     torch.tensor([0.0], requires_grad=True, device=device)
        # )

    @torch.no_grad()
    def downsample(self, resolution: int = 1):
        fx = self.fx / resolution
        fy = self.fy / resolution
        cx = self.cx / resolution
        cy = self.cy / resolution
        height = math.ceil(self.height / resolution)
        width = math.ceil(self.width / resolution)
        transform = transforms.Resize((height, width))
        image = transform(self.image.permute(2, 0, 1)).permute(1, 2, 0)
        camera = Camera(
            self.image_index, self.world_to_camera.transpose(0, 1),
            fx, fy, cx, cy, image * 255.0,
            device=self.image.device
        )
        return camera

    def compose_state_dict(self):
        state_dict = {
            'world_to_camera': torch.concat([
                torch.concat([self.R, self.t[..., None]], dim=-1),
                torch.tensor([[0., 0., 0., 1.]])
            ], dim=0),
            'image_index': self.image_index,
            'width': self.width,
            'height': self.height,
            'image': self.image * 255.,
            'fx': self.fx,
            'fy': self.fy,
            'cx': self.cx,
            'cy': self.cy,
        }

        # Normal is optional.
        if self.normal is not None:
            state_dict['normal'] = self.normal * 255.

        return state_dict

    @torch.no_grad()
    def update_camera_pose(self):
        tau = torch.cat([self.delta_rot, self.delta_trans], dim=0).unsqueeze(0)
        w2c = torch.eye(4, device=tau.device)
        w2c[:3, :3] = self.R
        w2c[:3,  3] = self.t

        new_w2c = (se3_exp_map(tau).squeeze(0).transpose(0, 1) @ w2c).to(self.device)
        # new_w2c = (w2c @ se3_exp_map(tau).squeeze(0).transpose(0, 1)).to(self.device)

        self.R = new_w2c[:3, :3]
        self.t = new_w2c[:3,  3]

        self.world_to_camera = new_w2c.transpose(0, 1)
        self.projective_matrix = self.world_to_camera @ self.projection_matrix
        self.camera_center = self.world_to_camera.inverse()[3, :3]

        self.delta_rot.data.fill_(0)
        self.delta_trans.data.fill_(0)

    @classmethod
    def read(cls, path: str, read_image: bool = True, read_normal: bool = False, device='cuda'):
        state_dict = torch.load(path)

        # Normal is optional.
        normal = None
        if read_normal and 'normal' in state_dict.keys():
            normal = state_dict['normal']

        camera = Camera(
            image_index=state_dict['image_index'],
            width=state_dict['width'],
            height=state_dict['height'],
            world_to_camera=state_dict['world_to_camera'],
            fx=state_dict['fx'],
            fy=state_dict['fy'],
            cx=state_dict['cx'],
            cy=state_dict['cy'],
            image=state_dict['image'] if read_image else None,
            normal=normal,
            device=device,
        )
        return camera

    def write(self, path: str, index: int):
        camera_path = os.path.join(path, f'camera_{index}.pt')
        if not os.path.exists(camera_path):
            state_dict = self.compose_state_dict()
            torch.save(state_dict, camera_path)

    def copy_to_device(self, device = "cuda:0"):
        """Deep copy camera data to device."""
        image = self.image.to(device) * 255.0 if self.image is not None else None
        normal = self.normal.to(device) * 255.0 if self.normal is not None else None
        world_to_camera = self.world_to_camera.transpose(0, 1).to(device)
        camera = Camera(
            image_index=self.image_index,
            width=self.width,
            height=self.height,
            world_to_camera=world_to_camera,
            fx=self.fx,
            fy=self.fy,
            cx=self.cx,
            cy=self.cy,
            image=image,
            normal=normal,
            device=device,
        )
        return camera

    def to_device(self, device = "cuda:0"):
        """Move camera data to device."""
        self.image = self.image.to(device) if self.image is not None else None
        self.normal = self.normal.to(device) if self.normal is not None else None
        self.world_to_camera = self.world_to_camera.to(device)
        self.projection_matrix = self.projection_matrix.to(device)
        self.projective_matrix = self.projective_matrix.to(device)
        self.camera_center = self.camera_center.to(device)

    def to_cpu(self):
        """Move camera data to cpu."""
        self.image = self.image.to('cpu') if self.image is not None else None
        self.normal = self.normal.to('cpu') is self.normal if not None else None
        self.world_to_camera = self.world_to_camera.to('cpu')
        self.projection_matrix = self.projection_matrix.to('cpu')
        self.projective_matrix = self.projective_matrix.to('cpu')
        self.camera_center = self.camera_center.to('cpu')

    @property
    def K(self): # pylint: disable=C0103
        """Camera intrinsic matrix"""
        K = torch.eye(3, device=self.device).unsqueeze(0)
        K[:, 0, 0] = self.fx
        K[:, 1, 1] = self.fy
        K[:, 0, 2] = self.cx
        K[:, 1, 2] = self.cy
        return K
    
    @property
    def Kinv(self): # pylint: disable=C0103
        """Inverse intrinsics (for lifting)"""
        Kinv = self.K.clone()
        Kinv[:, 0, 0] = 1. / self.fx
        Kinv[:, 1, 1] = 1. / self.fy
        Kinv[:, 0, 2] = -1. * self.cx / self.fx
        Kinv[:, 1, 2] = -1. * self.cy / self.fy
        return Kinv
    
    def reconstruct(self, depth: torch.Tensor, max_depth: float = 1000.0):
        """ Reconstructs pixel-wise 3D points from a depth map.

        Parameters:
            depth: [B,1,H,W] Depth map for the camera
            max_depth: the maximum depth threshold
        Returns:
            points: torch.Tensor [B,3,H,W] Pixel-wise 3D points.
        """
        # TODO(chenyu): Remove the batch dimension.
        B, C, H, W = depth.shape # pylint: disable=C0103
        assert C == 1

        # Create flat index grid
        grid = image_grid(B, H, W, depth.dtype, depth.device, normalized=False) # [B,3,H,W]
        flat_grid = grid.view(B, 3, -1) # [B,3,HW]

        # Estimate the outward rays in the camera frame.
        xnorm = (self.Kinv.bmm(flat_grid)).view(B, 3, H, W)
        # Scale rays to metric depth.
        Xc = xnorm * depth # [B,3,H,W]

        Xc = Xc.permute(1, 0, 2, 3) # [3,B,H,W]
        depth = depth.permute(1, 0, 2, 3).squeeze(0) # [1,B,H,W]
        # Mind pixels with invalid depth.
        Xc[:, depth == 0] = 0
        Xc[:, depth > max_depth] = 0
        Xc = Xc.permute(1, 0, 2, 3) # [B,3,H,W]

        # Project to world space.
        cam_to_world = torch.inverse(self.world_to_camera.transpose(0, 1))
        Rcw, tcw = cam_to_world[:3, :3], cam_to_world[:3, 3].unsqueeze(-1)
        Xc = Xc.reshape(B, 3, -1)
        Xw = (Rcw @ Xc + tcw).reshape(B, 3, H, W)
        
        return Xw
    
    def project(self, X: torch.Tensor, normalize: bool = True):
        """Projects 3D points onto the image plane.

        Parameters:
            X: [B,3,H,W] 3D points to be projected.
        
        Returns:
            points: torch.Tensor [B,H,W,2] 2D projected points that are within the image boundaries.
        """
        # TODO(chenyu): Remove the batch dimension.
        B, C, H, W = X.shape
        assert C == 3

        # Project 3D points onto the camera image plane.
        world_to_cam = self.world_to_camera.transpose(0, 1)
        Rwc, twc = world_to_cam[:3, :3], world_to_cam[:3, 3].unsqueeze(-1)
        Xw = X.reshape(B, C, -1)
        Xc = Rwc @ Xw + twc
        Xc = self.K @ Xc

        # Normalize points
        X = Xc[:, 0]
        Y = Xc[:, 1]
        Z = Xc[:, 2].clamp(min=1e-5)
        if normalize:
            Xnorm = 2 * (X / Z) / (W - 1) - 1.
            Ynorm = 2 * (Y / Z) / (H - 1) - 1.
        else:
            Xnorm = X / Z
            Ynorm = Y / Z
        
        # Return pixel coordinates
        return torch.stack([Xnorm, Ynorm], dim=-1).view(B, H, W, 2)
