# pylint: disable=[E1101,E1102]

import torch
import torch.nn as nn

from conerf.geometry.pose_util import se3_exp_map
from conerf.model.backbone.mlp import MLP


class PoseRefineNetwork(nn.Module):
    """
    Optimize the 6DoF camera poses.
    """

    def __init__(self, input_dim: int = 12, output_dim: int = 6, hidden_dim: int = 128):
        super(PoseRefineNetwork, self).__init__()

        self.hidden_dim = hidden_dim

        self.mlp = MLP(
            input_dim=input_dim,
            output_dim=output_dim,
            net_depth=6,  # hard-coded.
            net_width=hidden_dim,
            skip_layer=3,  # hard-coded.
            # TODO(chenyu): check with the hidden activation since it is not mentioned in the paper.
            hidden_activation=nn.ReLU(),
        )

    def forward(self, poses: torch.Tensor):
        """
        Parameters:
            @param poses: [N,3/4,4]
        Returns:
            optimized poses [N,4,4]
        """
        batch_size = poses.shape[0]

        poses_3x4 = poses[:, :3, :].reshape(batch_size, -1)  # [B,12]
        delta_se3 = self.mlp(poses_3x4)  # [B,6]
        delta_pose_4x4 = se3_exp_map(delta_se3)  # [B,4,4]

        updated_poses = poses @ delta_pose_4x4

        return updated_poses, delta_se3

        # poses = poses[:, :3, :].reshape(batch_size, -1)  # [B,12]
        # poses = self.mlp(poses)  # [B,12]
        # poses = poses.reshape(batch_size, 3, 4)  # [B,3,4]

        # # Retraction to recover the rotational part.
        # Us, _, Vhs = torch.linalg.svd(poses[:, :3, :3]) # pylint: disable=C0103

        # updated_poses = torch.eye(4, device=poses.device).reshape(-1, 4).repeat(batch_size, 1, 1)

        # # R = U @ V^T.
        # # Construct Z to fix the orientation of R to get det(R) = 1.
        # Z = torch.eye(3, device=poses.device).reshape(-1, 3).repeat(batch_size, 1, 1)
        # Z[:, -1, -1] = Z [:, -1, -1] * torch.sign(torch.linalg.det(Us @ Vhs))
        # updated_poses[:, :3, :3] = Us @ Z @ Vhs

        # # Copy translational part.
        # updated_poses[:, :3, 3:] = poses[:, :3, 3:]

        # return updated_poses
