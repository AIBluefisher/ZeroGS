# pylint: disable=[E1101,E1102,E0401,W0201]

import os
import logging
import time
import random
import copy
from typing import List

from omegaconf import OmegaConf

import numpy as np
import tqdm
import torch
import torch.backends
import torch.optim as optim
import torchvision.transforms.functional as TF
from torch.cuda.amp import GradScaler, autocast
from torch.utils.data import DataLoader
from torch.utils.data import sampler
from torchvision.transforms import ToTensor

import dsacstar

from conerf.datasets.ace_camera_loc_dataset import CamLocDataset
from conerf.geometry.camera import Camera
from conerf.geometry.align_poses import align_ate_c2b_use_a2b
from conerf.geometry.pose_util import rotation_distance
from conerf.model.scene_regressor.ace_network import Regressor
from conerf.model.scene_regressor.ace_loss import ReproLoss
from conerf.model.scene_regressor.ace_util import (
    get_pixel_grid, to_homogeneous, save_point_cloud
)
from conerf.model.scene_regressor.calibr import Calibr
from conerf.model.scene_regressor.depth_network import DepthNetwork
from conerf.model.scene_regressor.pose_refine_network import PoseRefineNetwork
from conerf.trainers.trainer import BaseTrainer
from conerf.utils.utils import colorize, save_images
from conerf.visualization.pose_visualizer import visualize_cameras

_logger = logging.getLogger(__name__)


def normalize_shape(tensor_in):
    """Bring tensor from shape [B,C,H,W] to [N,C]"""
    return tensor_in.transpose(0, 1).flatten(1).transpose(0, 1)


@torch.no_grad()
def depth_to_points3D(
    image_index, depth, pose, calibr, width, height, depth_max, device
):
    K = calibr(
        torch.tensor([height], device=device),
        torch.tensor([width], device=device)
    )[0]

    camera = Camera(
        image_index=image_index,
        world_to_camera=pose,
        fx=K[0, 0],
        fy=K[1, 1],
        cx=K[0, 2],
        cy=K[1, 2],
        image=None,
        width=width,
        height=height,
        device=device,
    )

    # The point clouds generated from the depth image served as an pseudo label to
    # initialize the scene regressor network.
    points3D = camera.reconstruct(depth, max_depth=depth_max)  # [B=1,3,H,W]
    points3D = points3D.permute(0, 2, 3, 1).flatten(0, 2).float()  # [B*H*W,3]

    return points3D


@torch.no_grad()
def register_image(
    regressor: torch.nn.Module,
    image: torch.Tensor,
    calib_refiner: torch.nn.Module,
    hypotheses: float,
    threshold: float,
    inlier_alpha: float,
    max_pixel_error: float,
    factor: int,
    max_hypotheses_tries: int,
    device: str,
):
    image = image.to(device, non_blocking=True)

    # Predict scene coordinates.
    with autocast(enabled=True):
        scene_coordinates = regressor(image)  # [1,3,H,W]
    # We need them on the CPU to run RANSAC.
    scene_coordinates = scene_coordinates.float().cpu()

    H, W = scene_coordinates.shape[-2:]  # pylint: disable=C0103
    intrinsics = calib_refiner(
        torch.tensor([H], device=device),
        torch.tensor([W], device=device)
    )[0]

    # Compute the pose via RANSAC.
    focal_length = intrinsics[0, 0].item()
    cx, cy = intrinsics[0, 2].item(), intrinsics[1, 2].item()
    out_pose = torch.zeros((4, 4))
    inlier_count = dsacstar.forward_rgb(
        scene_coordinates,
        out_pose,
        # ACE0 halves the number of hypotheses drawn from 64 to 32.
        hypotheses,
        threshold,
        focal_length,
        focal_length,
        cx, cy,
        inlier_alpha,
        max_pixel_error,
        factor,
        max_hypotheses_tries,
        False,
    )

    inlier_ratio = inlier_count / (H * W)

    return out_pose, scene_coordinates, inlier_count, inlier_ratio


def compute_reproj_error(
    pred_scene_coords_b31: torch.Tensor,
    target_px_b2: torch.Tensor,
    Ks_b33: torch.Tensor,  # pylint: disable=C0103
    poses_b44: torch.Tensor,
    depth_min: float,
):
    # Inverse camera pose from world to camera.
    inv_poses_b34 = torch.linalg.inv(poses_b44)[:, :3, :]

    # Make 3D points homogeneous so that we can easily matrix-multiply them.
    pred_scene_coords_b41 = to_homogeneous(pred_scene_coords_b31)

    # Scene coordinates to camera coordinates.
    pred_cam_coords_b31 = torch.bmm(inv_poses_b34, pred_scene_coords_b41)

    pred_px_b31 = torch.bmm(Ks_b33, pred_cam_coords_b31)

    # Avoid division by zero.
    # Note: negative values are also clamped at +self.config.regressor.depth_min.
    # The predicted pixel would be wrong, but that's fine since we mask them out later.
    pred_px_b31[:, 2].clamp_(min=depth_min)

    # Dehomogenise.
    pred_px_b21 = pred_px_b31[:, :2] / pred_px_b31[:, 2, None]

    reprojection_error_b2 = pred_px_b21.squeeze() - target_px_b2
    reprojection_error_b1 = torch.norm(
        reprojection_error_b2, dim=1, keepdim=True, p=1)

    return pred_cam_coords_b31, reprojection_error_b1


def compute_points_loss(gt_points, pred_points, depth_min, depth_max):
    invalid_min_depth_b1 = pred_points[:, 2] < depth_min
    invalid_max_depth_b1 = pred_points[:, 2] > depth_max
    invalid_mask = (invalid_min_depth_b1 | invalid_max_depth_b1)
    valid_mask = ~invalid_mask
    coords_square_error = torch.square(pred_points - gt_points)
    loss_valid = coords_square_error[valid_mask].sum()
    coords_abs_error = torch.abs(pred_points - gt_points)
    loss_invalid = coords_abs_error[invalid_mask].sum()
    loss = loss_valid + loss_invalid

    return loss


@torch.no_grad()
def pose_dict_to_tensor(pose_dict: dict):
    idxs = []
    poses = []
    for idx, pose in pose_dict.items():
        idxs.append(int(idx))
        poses.append(pose.to('cpu'))

    idxs = torch.tensor(idxs, dtype=torch.int32)
    poses = torch.stack(poses)

    return idxs, poses


@torch.no_grad()
def evaluate_camera_alignment(aligned_pred_poses, poses_gt):
    """
    measure errors in rotation and translation
    """
    R_aligned, t_aligned = aligned_pred_poses.split([3, 1], dim=-1)
    R_gt, t_gt = poses_gt.split([3, 1], dim=-1)

    R_error = rotation_distance(R_aligned[..., :3, :3], R_gt[..., :3, :3])
    t_error = (t_aligned - t_gt)[..., 0].norm(dim=-1)

    mean_rotation_error = np.rad2deg(R_error.mean().cpu()).item()
    mean_position_error = t_error.mean().item()
    med_rotation_error = np.rad2deg(R_error.median().cpu()).item()
    med_position_error = t_error.median().item()

    return {'R_error_mean': mean_rotation_error, "t_error_mean": mean_position_error,
            'R_error_med': med_rotation_error, 't_error_med': med_position_error}


class AceZeroTrainer(BaseTrainer):
    """
    Class for training ACE0 given a set of unordered images.
    """

    def __init__(
        self,
        config: OmegaConf,
        prefetch_dataset: bool = True,
        trainset=None,
        valset=None,
    ) -> None:
        # Properties that will be recorded into state_dicts are
        # initialized before the super's constructor.
        self.num_low_error_consecutive_batches = 0

        super().__init__(config, prefetch_dataset, trainset, valset)

        self.base_seed = 2089

        # Used to generate batch indices.
        self.batch_generator = torch.Generator()
        self.batch_generator.manual_seed(self.base_seed + 1023)

        # Dataloader generator, used to seed individual workers by the dataloader.
        self.loader_generator = torch.Generator()
        self.loader_generator.manual_seed(self.base_seed + 511)

        # Generator used to sample random features (runs on the GPU).
        self.sampling_generator = torch.Generator(device=self.device)
        self.sampling_generator.manual_seed(self.base_seed + 4095)

        # Generator used to permute the feature indices during each training epoch.
        self.training_generator = torch.Generator()
        self.training_generator.manual_seed(self.base_seed + 8191)

        self.config.trainer.max_iterations = self.config.trainer.epochs * \
            self.config.trainer.training_buffer_size // self.config.trainer.batch_size

        _logger.info(
            "Loaded training images from: %s -- %d images",
            self.config.dataset.scene, len(self.train_dataset),
        )

        # Will be filled at the beginning of the training process.
        self.training_buffer = None
        self.pixel_grid = get_pixel_grid(
            self.train_dataset.origin_image_height,
            self.train_dataset.origin_image_width,
        )  # [2,H,W]

        self.seed_image_idx = -1
        self.reg_image_idx_to_pose = {}
        self.prev_num_reg_images = 0
        self.finalize = False  # Whether to finalize the training.

    def load_dataset(self):
        self.train_dataset = CamLocDataset(
            root_fp=self.config.dataset.root_dir,
            subject_id=self.config.dataset.scene,
            use_half=self.config.trainer.get("use_half", False),
            factor=self.config.dataset.factor,
            augment=self.config.dataset.use_aug,
            aug_rotation=self.config.dataset.aug_rotation,
            aug_scale_max=self.config.dataset.aug_scale,
            aug_scale_min=1 / self.config.dataset.aug_scale,
        )

        self.register_dataset = CamLocDataset(
            root_fp=self.config.dataset.root_dir,
            subject_id=self.config.dataset.scene,
            use_half=self.config.trainer.get("use_half", False),
            factor=self.config.dataset.factor,
            augment=False,
            aug_rotation=False,
        )

    def build_pose_refiner(self):
        self.pose_refiner = PoseRefineNetwork().to(self.device)
        self.pose_refiner.train()

    def build_networks(self):
        # (1) Scene Coordinate Regressor.
        # Create network using the state dict of the pretrained encoder.
        encoder_state_dict = torch.load(
            self.config.dataset.encoder_path, map_location='cpu')
        self.regressor = Regressor.create_from_encoder(
            encoder_state_dict=encoder_state_dict,
            num_head_blocks=self.config.regressor.num_head_blocks,
            use_homogeneous=self.config.regressor.use_homogeneous,
        ).to(self.device)
        self.regressor.train()

        # (2) Camera pose refine network.
        self.build_pose_refiner()

        # (3) Calibration refinement embedding.
        self.calib_refiner = Calibr(self.device).to(self.device)
        self.calib_refiner.train()

    def setup_optimizer(self):
        # (1) Optimizer for scene coordinate regressor.
        self.optimizer = optim.AdamW(
            self.regressor.parameters(), lr=self.config.optimizer.lr_sc_min)
        steps_per_epoch = self.config.trainer.training_buffer_size // self.config.trainer.batch_size
        self.scheduler = optim.lr_scheduler.OneCycleLR(
            self.optimizer,
            max_lr=self.config.optimizer.lr_sc_max,
            epochs=self.config.trainer.epochs,
            steps_per_epoch=steps_per_epoch,
            cycle_momentum=False,
        )

        # (2) Optimizer for pose refine network.
        self.pose_optimizer = optim.AdamW(
            self.pose_refiner.parameters(),
            lr=self.config.optimizer.lr_pr,
            weight_decay=0.01,
        )

        # (3) Optimizer for calibration refiner.
        self.calib_optimizer = optim.AdamW(
            self.calib_refiner.parameters(),
            lr=self.config.optimizer.lr_cr,
            weight_decay=0.01,
        )

        # Gradient scaler in case we train with half precision.
        self.scaler = GradScaler(enabled=self.config.trainer.use_half)

    def setup_loss_functions(self):
        self.repro_loss = ReproLoss(
            total_iterations=self.config.trainer.max_iterations_per_epoch,
            soft_clamp=self.config.loss.repro_loss_soft_clamp,
            soft_clamp_min=self.config.loss.repro_loss_soft_clamp_min,
            type=self.config.loss.repro_loss_type,
            circle_schedule=(
                self.config.loss.repro_loss_scheduler == 'circle'),
        )

    def _setup_visualizer(self):
        super()._setup_visualizer()

        # Generate video of training process.
        self.ace_visualizer = None

    def _clear_training_buffer(self):
        torch.cuda.reset_peak_memory_stats()
        peak_memory = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        print(f'PEAK MEMORY BEFORE: {peak_memory}')

        self.training_buffer['features'] = self.training_buffer['features'].to(
            'cpu')
        self.training_buffer['target_px'] = self.training_buffer['target_px'].to(
            'cpu')
        self.training_buffer['sample_idxs'] = self.training_buffer['sample_idxs'].to(
            'cpu')
        self.training_buffer['image_idxs'] = self.training_buffer['image_idxs'].to(
            'cpu')
        self.training_buffer['rotations'] = self.training_buffer['rotations'].to(
            'cpu')
        self.training_buffer['heights'] = self.training_buffer['heights'].to(
            'cpu')
        self.training_buffer['widths'] = self.training_buffer['widths'].to(
            'cpu')
        self.training_buffer = None
        torch.cuda.empty_cache()

        torch.cuda.reset_peak_memory_stats()
        peak_memory = torch.cuda.max_memory_allocated() / (1024.0 ** 2)
        print(f'PEAK MEMORY AFTER: {peak_memory}')

    def _finish_training(self):
        # Training terminated if all images have been registered or less than 1% of
        # additional views are registered compared to the last relocalization stage.
        num_reg_images = len(self.reg_image_idx_to_pose)
        if num_reg_images == len(self.train_dataset.rgb_files):
            _logger.info("All images have been registered!")
            return True

        reg_ratio = (self.prev_num_reg_images -
                     num_reg_images) / num_reg_images
        if abs(reg_ratio) < 0.01:
            _logger.info("Too few images can be registered after training!")
            return True

        return False

    def _registration_succeed(self, inlier_count):
        return inlier_count >= self.config.pose_estimator.min_inlier_count

    @torch.no_grad()
    def _check_early_stopping(self, errors, error_thresh: float) -> bool:
        batch_size = errors.shape[0]
        num_low_reproj_error_pixels = (errors[:] <= error_thresh).sum()
        ratio_low_reproj_error_pixels = num_low_reproj_error_pixels / batch_size

        if ratio_low_reproj_error_pixels >= 0.7:
            self.num_low_error_consecutive_batches += 1
        else:
            self.num_low_error_consecutive_batches = 0
            self.early_stopping = False

        if self.num_low_error_consecutive_batches >= 100:
            self.num_low_error_consecutive_batches = 0
            self.early_stopping = True

    @torch.no_grad()
    def _single_view_reconstruction(self, depth_network, height, width, seed_index):
        pose = torch.eye(4).to(self.device)

        image = self.train_dataset.image(seed_index)
        image = TF.to_pil_image(image)
        color_image = ToTensor()(image)  # [3,H,W]
        colors = color_image.permute(1, 2, 0).reshape(-1, 3)  # [H*W,3]
        # Rescale the image such that it aligns with the shape input to
        # the scene regressor network.
        resized_image = TF.resize(
            image, [height, width], interpolation=TF.InterpolationMode.NEAREST)
        resized_colors = ToTensor()(resized_image).permute(
            1, 2, 0).reshape(-1, 3)  # [h*w,3]

        depth, _, _ = depth_network.infer(  # [depth, confidence, output_dict]
            color_image.unsqueeze(0).to(self.device))  # [B=1,1,H,W]
        color_depth = colorize(depth.cpu().squeeze(
            0).squeeze(0), cmap_name="jet")  # [H,W,3]
        resized_depth = TF.resize(depth, [height, width])  # [h,w,1]
        resized_color_depth = colorize(resized_depth.cpu().squeeze(
            0).squeeze(0), cmap_name="jet")  # [h,w,3]
        save_images(
            save_dir=self.output_path,
            image_dict={
                "rgb": color_image.permute(1, 2, 0),
                "depth": color_depth,
                "resized_depth": resized_color_depth,
            },
            index=seed_index,
        )

        all_scene_coords = depth_to_points3D(
            seed_index, depth, pose, self.calib_refiner,
            color_image.shape[2], color_image.shape[1],
            self.config.regressor.depth_max, self.device,
        )

        pseudo_gt_scene_coords = depth_to_points3D(
            seed_index, resized_depth, pose, self.calib_refiner, width, height,
            self.config.regressor.depth_max, self.device,
        )

        full_seed_recon_path = os.path.join(
            self.output_path, f"full_seed_{seed_index}.ply")
        save_point_cloud(all_scene_coords, colors=colors,
                         path=full_seed_recon_path)

        gt_seed_recon_path = os.path.join(
            self.output_path, f"gt_seed_{seed_index}.ply")
        save_point_cloud(pseudo_gt_scene_coords,
                         colors=resized_colors, path=gt_seed_recon_path)
        _logger.info(
            "pseudo ground truth point clouds are saved to: %s", gt_seed_recon_path)

        return pseudo_gt_scene_coords, all_scene_coords, colors

    # @torch.no_grad()
    def _get_registered_image_poses(
        self, image_idxs: torch.Tensor, rotations: torch.Tensor = None
    ) -> torch.Tensor:
        """Get camera poses from registered images.
        Param:
            @param image_idxs: [B]
            @param rotations: [B,4,4]
        Return:
            camera poses: [B,4,4]
        """
        poses = []
        with torch.no_grad():
            for image_idx in image_idxs:
                pose = self.reg_image_idx_to_pose[int(
                    image_idx)].to(self.device)
                poses.append(pose)
            poses = torch.stack(poses, dim=0)  # [B,4,4]

            # The camera poses should be the poses without augmentation right multiply
            # the rotations that augment the image.
            if rotations is not None:
                poses = poses @ rotations

        return poses

        # se3_refine = self.delta_pose.weight[image_idxs]
        # pose_refine = se3_exp_map(se3_refine)
        # updated_poses = poses @ pose_refine

        # return updated_poses

    def _compute_reproj_loss(
        self,
        invKs_b33: torch.Tensor,  # pylint: disable=C0103
        target_px_b2: torch.Tensor,
        pred_cam_coords_b31: torch.Tensor,
        reprojection_error_b1: torch.Tensor,
    ):
        batch_size = invKs_b33.shape[0]
        invalid_min_depth_b1 = pred_cam_coords_b31[:,
                                                   2] < self.config.regressor.depth_min
        invalid_max_depth_b1 = pred_cam_coords_b31[:,
                                                   2] > self.config.regressor.depth_max
        invalid_repro_b1 = reprojection_error_b1 > self.config.loss.repro_loss_hard_clamp
        invalid_mask_b1 = (invalid_min_depth_b1 |
                           invalid_repro_b1 | invalid_max_depth_b1)
        valid_mask_b1 = ~invalid_mask_b1

        valid_reprojection_error_b1 = reprojection_error_b1[valid_mask_b1]
        loss_valid = self.repro_loss.compute(
            valid_reprojection_error_b1, self.iteration)

        # Handle the invalid predictions: generate proxy coordinate targets with constant depth
        # assumption.
        pixel_grid_crop_b31 = to_homogeneous(target_px_b2.unsqueeze(2))
        target_camera_coords_b31 = self.config.regressor.depth_target * \
            torch.bmm(invKs_b33, pixel_grid_crop_b31)

        # Compute the distance to target camera coordinates.
        invalid_mask_b11 = invalid_mask_b1.unsqueeze(2)
        loss_invalid = torch.abs(
            target_camera_coords_b31 - pred_cam_coords_b31
        ).masked_select(invalid_mask_b11).sum()

        loss = loss_valid + loss_invalid
        loss = loss / batch_size

        return loss_valid, loss_invalid, loss

    def _create_new_training_session(self):
        # Load the pretrained feature encoder.
        encoder_state_dict = torch.load(
            self.config.dataset.encoder_path, map_location='cpu')

        regressor = Regressor.create_from_encoder(
            encoder_state_dict=encoder_state_dict,
            num_head_blocks=self.config.regressor.num_head_blocks,
            use_homogeneous=self.config.regressor.use_homogeneous,
        )
        regressor = regressor.to(self.device)
        regressor.train()

        optimizer = optim.AdamW(
            regressor.parameters(), lr=self.config.optimizer.lr_sc_min)
        scaler = GradScaler(enabled=self.config.trainer.use_half)

        steps_per_epoch = self.config.trainer.training_buffer_size \
            // self.config.trainer.batch_size
        epochs = self.config.trainer.max_iterations_per_epoch // steps_per_epoch
        scheduler = optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=self.config.optimizer.lr_sc_max,
            epochs=epochs,
            steps_per_epoch=steps_per_epoch,
            cycle_momentum=False,
        )

        return regressor, optimizer, scheduler, scaler

    def _create_depth_network(self):
        depth_network = DepthNetwork(
            self.config.regressor.depth_net_method,
            self.config.regressor.depth_net_type,
            pretrain=True,
            depth_min=self.config.regressor.depth_min,
            depth_max=self.config.regressor.depth_max,
            device=self.device,
        )
        return depth_network

    @torch.no_grad()
    def update_scene_center(self):
        _, pred_poses = pose_dict_to_tensor(self.reg_image_idx_to_pose)
        centers = pred_poses[:, :3, 3]

        self.regressor.heads.mean = torch.mean(
            centers, dim=0).view(1, 3, 1, 1).to(self.device)
        print(f'scene center: {self.regressor.heads.mean}')

    def initialize(self):
        """
        Initialize the seed reconstruction.
        In the seed iteration, ACE0 optimizes neither the seed pose (which is a identity)
        nor the initial calibration parameters.
        """

        depth_network = self._create_depth_network()

        rand_image_indices = np.random.permutation(
            len(self.register_dataset)).tolist()
        num_seed_image_trials = self.config.regressor.num_seed_image_trials
        seed_image_indices = rand_image_indices[:num_seed_image_trials]

        # Cache the relocalized images into GPU memory
        rand_reloc_image_indices = rand_image_indices[
            num_seed_image_trials:num_seed_image_trials +
            self.config.regressor.num_reloc_images_max
        ]
        reloc_image_indices, reloc_images = [], []
        for reloc_image_index in rand_reloc_image_indices:
            image = self.register_dataset.resized_grayscale_image(
                reloc_image_index, self.register_dataset.image_height
            )[-1].to(self.device)
            reloc_image_indices.append(reloc_image_index)
            reloc_images.append(image)

        num_relocalized_images_max = 0
        reg_images_for_best_model = {}
        for seed_index in seed_image_indices:
            self.reg_image_idx_to_pose.clear()
            self.reg_image_idx_to_pose[seed_index] = torch.eye(4)
            dataset = copy.deepcopy(self.register_dataset)
            self.create_training_buffer(dataset)

            training_buffer_size = len(self.training_buffer['features'])
            H, W = self.training_buffer['H'], self.training_buffer['W']

            pseudo_gt_scene_coords, _, _ = self._single_view_reconstruction(
                depth_network, H, W, seed_index,
            )

            # Start a new session for different seed candidates.
            regressor, optimizer, scheduler, scaler = self._create_new_training_session()

            # Training the initial regressor.
            iteration = 0
            while iteration < self.config.trainer.max_iterations_per_epoch:
                random_indices = torch.randperm(
                    training_buffer_size, generator=self.training_generator
                )

                for batch_start in range(0, training_buffer_size, self.config.trainer.batch_size):
                    batch_end = batch_start + self.config.trainer.batch_size
                    if batch_end > training_buffer_size:
                        continue
                    random_batch_indices = random_indices[batch_start:batch_end]

                    pixel_idxs = self.training_buffer['sample_idxs'][random_batch_indices]
                    features = self.training_buffer['features'][random_batch_indices].contiguous(
                    )
                    gt_scene_coords = pseudo_gt_scene_coords[pixel_idxs].contiguous(
                    )

                    # Reshape to a "fake" BCHW shape, since it's faster to run through the network
                    # compared to the original shape.
                    batch_size, channels = features.shape
                    features = features[None, None, ...].view(
                        -1, 16, 32, channels).permute(0, 3, 1, 2)

                    with autocast(enabled=self.config.trainer.use_half):
                        pred_scene_coords = regressor.get_scene_coordinates(
                            features)

                    pred_scene_coords = pred_scene_coords.permute(
                        0, 2, 3, 1).flatten(0, 2).float()  # [B*H*W,3]

                    with torch.no_grad():
                        target_px_b2 = self.training_buffer['target_px'][random_batch_indices].contiguous(
                        )
                        heights = self.training_buffer['heights'][random_batch_indices].contiguous(
                        )
                        widths = self.training_buffer['widths'][random_batch_indices].contiguous(
                        )
                        Ks_b33 = self.calib_refiner(heights, widths)  # [B,3,3]
                        poses_b44 = torch.eye(4, device=self.device)[
                            None, ...].repeat(batch_size, 1, 1)

                    _, reprojection_error_b1 = compute_reproj_error(
                        pred_scene_coords.unsqueeze(
                            -1), target_px_b2, Ks_b33, poses_b44,
                        self.config.regressor.depth_min,
                    )
                    min_reproj_error = reprojection_error_b1.min()
                    mean_reproj_error = reprojection_error_b1.mean()

                    loss_type = "Euclidean"
                    loss = compute_points_loss(
                        gt_scene_coords, pred_scene_coords,
                        self.config.regressor.depth_min, self.config.regressor.depth_max)

                    old_optimizer_step = optimizer._step_count  # pylint: disable=W0212

                    optimizer.zero_grad(set_to_none=True)
                    scaler.scale(loss).backward()
                    scaler.step(optimizer)
                    scaler.update()

                    if old_optimizer_step < optimizer._step_count < scheduler.total_steps:  # pylint: disable=W0212
                        scheduler.step()

                    if iteration and iteration % (self.config.trainer.n_tensorboard * 5) == 0:
                        _logger.info(  # pylint: disable=W1203
                            f"[Init] Iteration: {iteration:6d}, min_reproj_error: {min_reproj_error:.5f}, " +
                            f"mean_reproj_error: {mean_reproj_error:.5f}, Loss Type: {loss_type}, Loss: {loss:.5f}")

                    iteration += 1
            # End of a initialize session.

            # Relocalization test after training the seed scene regressor.
            regressor.eval()
            num_relocalized_images, cur_reg_images = self.register_init_image_batches(
                regressor, seed_index, reloc_image_indices, reloc_images)
            _logger.info("%d images are registered for seed #%d",
                         len(cur_reg_images), seed_index)

            if num_relocalized_images_max < num_relocalized_images:
                num_relocalized_images_max = num_relocalized_images
                reg_images_for_best_model = cur_reg_images
                self.seed_image_idx = seed_index
                self.regressor.load_state_dict(regressor.state_dict())

        self.reg_image_idx_to_pose = reg_images_for_best_model

        _logger.info(
            "Image #%d is selected as the seed image, " +
            "%d images are registered after initialization.",
            self.seed_image_idx, len(self.reg_image_idx_to_pose)
        )

        self.visualize_seed_reconstruction()

    @torch.no_grad()
    def register_init_image_batches(self, regressor, seed_index, image_indices, images):
        num_relocalized_images = 0
        cur_reg_images = {seed_index: torch.eye(4)}
        for image_idx, reloc_image in zip(image_indices, images):
            pose, _, inlier_count, _ = register_image(
                regressor,
                reloc_image.unsqueeze(0),  # [1,3,H,W]
                self.calib_refiner,
                self.config.pose_estimator.hypotheses,
                self.config.pose_estimator.reproj_thresh,
                self.config.pose_estimator.inlier_alpha,
                self.config.pose_estimator.max_pixel_error,
                1,  # self.config.dataset.factor,
                max_hypotheses_tries=1000000,
                device=self.device,
            )  # pose is from camera to world.

            if self._registration_succeed(inlier_count):
                num_relocalized_images += 1
                cur_reg_images[image_idx] = pose.to(self.device)

        return num_relocalized_images, cur_reg_images

    @torch.no_grad()
    def visualize_seed_reconstruction(self):
        seed_image_pil, seed_image = self.register_dataset.resized_grayscale_image(
            self.seed_image_idx, self.register_dataset.image_height)
        seed_image = seed_image.to(self.device)

        with autocast(enabled=True):
            pred_scene_coords = self.regressor(
                seed_image.unsqueeze(0))  # [B,3,H,W]

        H, W = pred_scene_coords.shape[-2:]
        seed_image_pil = TF.resize(
            seed_image_pil, [H, W], interpolation=TF.InterpolationMode.NEAREST)
        colors = ToTensor()(seed_image_pil).permute(
            1, 2, 0).reshape(-1, 3)  # [H*W,3]

        pred_scene_coords = pred_scene_coords.float().cpu().permute(
            0, 2, 3, 1).flatten(0, 2).float()  # [B*H*W,3]

        pred_seed_recon_path = os.path.join(
            self.output_path, f"pred_seed_{self.seed_image_idx}.ply")
        save_point_cloud(
            pred_scene_coords,
            colors=colors,
            path=pred_seed_recon_path,
        )
        _logger.info(
            "predicted initial point clouds are saved to: %s", pred_seed_recon_path)

        idxs, pred_poses = pose_dict_to_tensor(self.reg_image_idx_to_pose)
        gt_poses = self.train_dataset.gt_camtoworlds[idxs]
        pred_poses = align_ate_c2b_use_a2b(pred_poses, gt_poses)[0]

        if self.config.trainer.enable_visdom:
            visualize_cameras(
                self.visdom, 0, poses=[pred_poses, gt_poses], colors=["blue", "magenta"],
                cam_depth=0.1, plot_dist=True,
            )
        pose_error = evaluate_camera_alignment(pred_poses, gt_poses)
        _logger.info(
            f"rotation error: {pose_error['R_error_mean']:.4f}, " +
            f"translation error: {pose_error['t_error_mean']:.4f}"
        )

    @torch.no_grad()
    def register_images(self, image_idxs: List = None):
        """
        Batched incremental registration of images.
        """
        num_reg_images = 1
        self.prev_num_reg_images = len(self.reg_image_idx_to_pose)
        self.reg_image_idx_to_pose = {}

        # Registration dataloader. Batch size 1 by default.
        register_dataloader = DataLoader(
            dataset=self.register_dataset,
            shuffle=False,
            num_workers=8,
        )

        pbar = tqdm.trange(
            len(self.register_dataset), desc="Registering images", leave=False)
        for image, _, _, idx, _ in register_dataloader:
            idx = int(idx)
            # Only register the specified images.
            if image_idxs is not None and idx not in image_idxs:
                continue

            image = image.to(self.device, non_blocking=True)  # [B,1,H,W]

            out_pose, _, inlier_count, _ = register_image(
                self.regressor,
                image,
                self.calib_refiner,
                self.config.pose_estimator.hypotheses // 2,
                self.config.pose_estimator.reproj_thresh,
                self.config.pose_estimator.inlier_alpha,
                self.config.pose_estimator.max_pixel_error,
                1,  # self.config.dataset.factor,
                max_hypotheses_tries=16,
                device=self.device,
            )

            # TODO(chenyu): using a tighter threshold for the last iteration.
            if self._registration_succeed(inlier_count):
                num_reg_images += 1
                self.reg_image_idx_to_pose[idx] = out_pose.to(self.device)

            pbar.update(1)

        idxs, pred_poses = pose_dict_to_tensor(self.reg_image_idx_to_pose)
        gt_poses = self.train_dataset.gt_camtoworlds[idxs]
        pred_poses = align_ate_c2b_use_a2b(pred_poses, gt_poses)[0]

        if self.config.trainer.enable_visdom:
            visualize_cameras(
                self.visdom, self.iteration, poses=[pred_poses, gt_poses],
                colors=["blue", "magenta"], cam_depth=0.1, plot_dist=True,
            )

        _logger.info("Registered %d images.", num_reg_images)

        pose_error = evaluate_camera_alignment(pred_poses, gt_poses)
        _logger.info(
            f"rotation error: {pose_error['R_error_mean']:.4f}, " +
            f"translation error: {pose_error['t_error_mean']:.4f}"
        )

    @torch.no_grad()
    def update_learning_rate(self, iteration: int):
        """
        Approximate the one-cycle learning rate schedule in a linear fashion:
          - (1) increase the learning rate in the first 1K iterations;
          - (2) decrease the learning rate within 5K iterations when the early stopping 
                criterion has been met.
        """
        max_anneal_inc_steps = 1000
        max_anneal_dec_steps = 5000

        if iteration >= max_anneal_dec_steps and self.anneal_status == 'decrease':
            self.anneal_status = 'increase'

        if self.early_stopping and iteration >= self.config.trainer.min_iterations_per_epoch:
            self.anneal_status = 'decrease'

        if self.anneal_status == 'increase':
            self.cur_lr = self.config.optimizer.lr_sc_min + (
                self.config.optimizer.lr_sc_max - self.config.optimizer.lr_sc_min
            ) * max(iteration / max_anneal_inc_steps, 1)
        else:
            self.cur_lr = self.config.optimizer.lr_sc_min + (
                self.config.optimizer.lr_sc_max - self.config.optimizer.lr_sc_min
            ) * max(max_anneal_dec_steps - iteration, 0) / max_anneal_dec_steps

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.cur_lr

    def train(self):
        """
         and subsequently trains a scene
        coordinate regression head.
        """

        iter_start = self.load_checkpoint(
            load_optimizer=not self.config.trainer.no_load_opt,
            load_scheduler=not self.config.trainer.no_load_scheduler
        )

        if iter_start < self.config.trainer.n_checkpoint:
            self.initialize()
            self.prev_reg_image_idxs = [self.seed_image_idx]

        self.anneal_status = 'increase'
        self.cur_lr = self.config.optimizer.lr_sc_min

        while True:
            # Enable benchmarking since all operations work on the same tensor size.
            torch.backends.cudnn.benchmark = True

            # Fills a feature buffer using the pretrained encoder.
            # The training buffer should be re-created at the beginning
            # of each training iteration. Ref: Sec.3.2 at Page 8.
            self.create_training_buffer(self.train_dataset)

            self.update_scene_center()

            training_buffer_size = len(self.training_buffer['features'])

            iteration = 0
            while iteration < self.config.trainer.max_iterations_per_epoch:
                random_indices = torch.randperm(
                    training_buffer_size, generator=self.training_generator)

                for batch_start in range(0, training_buffer_size, self.config.trainer.batch_size):
                    batch_end = batch_start + self.config.trainer.batch_size

                    # Drop last batch if not full.
                    if batch_end > training_buffer_size:
                        continue

                    random_batch_indices = random_indices[batch_start:batch_end]

                    self.train_iteration(random_batch_indices)

                    iteration += 1
                    self.iteration += 1

                    self.update_learning_rate(iteration)

                    if self.iteration % self.config.trainer.n_checkpoint == 0:
                        self.save_checkpoint(score=0)

                    if self.early_stopping and iteration >= self.config.trainer.min_iterations_per_epoch:
                        break

                if self.early_stopping and iteration >= self.config.trainer.min_iterations_per_epoch:
                    _logger.info(
                        "Early stopped current epoch due to low reprojection error!")
                    self.early_stopping = False
                    break
            # End of a training epoch.
            self.epoch += 1

            image_idxs_to_register = list(self.reg_image_idx_to_pose.keys())
            # Re-estimate camera poses for registered images.
            self.register_images(image_idxs_to_register)

            # At the end of each training epoch, we re-estimate the poses of all views,
            # even if they have been registered before. This gives the pipeline the ability
            # to correct previous outlier estimates.
            self.prev_reg_image_idxs = image_idxs_to_register
            self.register_images()

            # Check wether terminate the training.
            if self._finish_training():
                break

            # if self.finalize is True:
            #     break

            # # Check wether terminate the training.
            # if self._finish_training():
            #     _logger.info("Finalizing the network training!")
            #     self.regressor, self.optimizer, self.scheduler, self.scaler = \
            #         self._create_new_training_session()
            #     self.config.trainer.early_stop_thresh /= 2
            #     self.finalize = True

        # In case we lost the last checkpoint.
        self.save_checkpoint(score=0)

        self.train_done = True

    def create_training_buffer(self, dataset):
        """
        Create training buffer at the begins of each training iteration with the registered
        images in the previous relocalization iteration.
        """
        # Refresh the available training images from the registered images from previous
        # relocalization iteration.
        dataset.valid_file_indices = {}
        for idx, _ in self.reg_image_idx_to_pose.items():
            index = len(dataset.valid_file_indices)
            dataset.valid_file_indices[index] = idx

        # Disable benchmarking, since we have variable tensor sizes.
        torch.backends.cudnn.benchmark = False

        batch_sampler = sampler.BatchSampler(
            sampler.RandomSampler(dataset,
                                  generator=self.batch_generator),
            batch_size=1,
            drop_last=False
        )

        # Used to seed workers in a reproducible manner.
        def seed_worker(worker_id):
            # Different seed per epoch. Initial seed is generated by the main process consuming
            # one random number from the dataloader generator.
            worker_seed = torch.initial_seed() % 2 ** 32
            np.random.seed(worker_seed)
            random.seed(worker_seed)

        # Batching is handled at the dataset level (the dataset __getitem__ receives a list of
        # indices, because we need to rescale all images in the batch to the same size).
        training_dataloader = DataLoader(
            dataset=dataset,
            sampler=batch_sampler,
            batch_size=None,
            worker_init_fn=seed_worker,
            generator=self.loader_generator,
            pin_memory=True,
            num_workers=self.config.trainer.num_workers,
            persistent_workers=self.config.trainer.num_workers > 0,
            timeout=60 if self.config.trainer.num_workers > 0 else 0,
        )

        # Create a training buffer that lives on the GPU.
        buffer_size = min(
            len(dataset) * self.config.trainer.max_patch_loops_per_epoch *
            self.config.trainer.samples_per_image,
            self.config.trainer.training_buffer_size
        )
        self.training_buffer = {
            'features': torch.empty(
                (buffer_size, self.regressor.feature_dim),
                dtype=(torch.float32, torch.float16)[
                    self.config.trainer.use_half],
                device=self.device,
            ),
            'target_px': torch.empty(
                (buffer_size, 2),
                dtype=torch.float32,
                device=self.device,
            ),
            'sample_idxs': torch.empty(buffer_size, dtype=torch.int64, device=self.device),
            'image_idxs': torch.empty(buffer_size, dtype=torch.int64),
            'rotations': torch.empty((buffer_size, 4, 4,), dtype=torch.float32, device=self.device),
            "heights": torch.empty(buffer_size, dtype=torch.float32, device=self.device),
            "widths": torch.empty(buffer_size, dtype=torch.float32, device=self.device),
        }

        # Features are computed in evaluation mode.
        self.regressor.eval()

        buffer_start_time = time.time()
        pbar = tqdm.trange(
            buffer_size, desc="Filling training buffers", leave=False)

        # The encoder is pretrained, so we don't compute any gradient.
        with torch.no_grad():
            # Iterate until the training buffer is full.
            buffer_idx = 0
            dataset_passes = 0

            while buffer_idx < buffer_size:
                # In the beginning of the reconstruction, we have very few mapping images,
                # thus we loop over the mapping images at 10 times when sampling patches,
                # or until 8M patches have been sampled.
                dataset_passes += 1
                if dataset_passes > self.config.trainer.max_patch_loops_per_epoch:
                    break

                for image, image_mask, rotation, image_idx, _ in training_dataloader:
                    image = image.to(
                        self.device, non_blocking=True)  # [B,1,H,W]
                    image_mask = image_mask.to(
                        self.device, non_blocking=True)  # [B,1,H,W]

                    # Compute image features.
                    with autocast(enabled=self.config.trainer.use_half):
                        features = self.regressor.get_features(
                            image)  # [B,C,H,W]

                    # Dimensions after the network's downsampling.
                    B, C, H, W = features.shape  # pylint: disable=[E0103,C0103]

                    # The image_mask needs to be downsampled to the actual output resolution
                    # and cast to bool.
                    image_mask = TF.resize(
                        image_mask, [
                            H, W], interpolation=TF.InterpolationMode.NEAREST
                    )
                    image_mask = image_mask.bool()

                    # If the current mask has no valid pixels, continue.
                    if image_mask.sum() == 0:
                        continue

                    # Create a tensor with the pixel coordinates of every feature vector.
                    # [2,H,W]
                    pixel_positions = self.pixel_grid[:, :H, :W].clone()
                    pixel_positions = pixel_positions[None]  # [1,2,H,W]
                    pixel_positions = pixel_positions.expand(
                        B, 2, H, W)  # [B,2,H,W]

                    batch_data = {
                        'features': normalize_shape(features),
                        'target_px': normalize_shape(pixel_positions).to(self.device),
                    }

                    # Turn image mask into sampling weights (all equal).
                    image_mask = image_mask.float()
                    image_mask_N1 = normalize_shape(image_mask)

                    # Over-sample according to image mask.
                    features_to_select = self.config.trainer.samples_per_image * B
                    features_to_select = min(
                        features_to_select, buffer_size - buffer_idx)

                    # Sample indices uniformly, with replacement.
                    sample_idxs = torch.multinomial(
                        image_mask_N1.view(-1),
                        features_to_select,
                        replacement=True,
                        generator=self.sampling_generator,
                    ).to(self.device)

                    # Select the data to put in the buffer.
                    for k in batch_data:
                        batch_data[k] = batch_data[k][sample_idxs]
                    batch_data['sample_idxs'] = sample_idxs
                    batch_data['image_idxs'] = torch.tensor(
                        [image_idx for _ in range(features_to_select)], dtype=torch.int64)
                    batch_data['rotations'] = rotation.expand(
                        features_to_select, -1, -1).to(self.device)
                    batch_data['heights'] = torch.tensor([H]).expand(
                        features_to_select).to(self.device)
                    batch_data['widths'] = torch.tensor([W]).expand(
                        features_to_select).to(self.device)

                    # Write to training buffer. Start at buffer_idx and end at buffer_offset - 1.
                    buffer_offset = buffer_idx + features_to_select
                    for k, data in batch_data.items():
                        self.training_buffer[k][buffer_idx:buffer_offset] = data

                    pbar.update(features_to_select)
                    buffer_idx = buffer_offset
                    if buffer_idx >= buffer_size:
                        break

        buffer_memory = sum([v.element_size() * v.nelement()
                            for k, v in self.training_buffer.items()])
        buffer_memory /= 1024 * 1024 * 1024

        self.training_buffer['H'] = H
        self.training_buffer['W'] = W

        buffer_end_time = time.time()
        creating_buffer_time = buffer_end_time - buffer_start_time

        _logger.info(f"Created buffer of {buffer_memory:.2f}GB " +
                     f"with {dataset_passes} passes over the training data in " +
                     f"{creating_buffer_time:.1f} s.")
        self.regressor.train()

    def train_iteration(self, data_batch) -> None:
        """
        Run one iteration of training, computing the reprojection error and minimizing it.
        """
        # torch.autograd.set_detect_anomaly(True)

        batch_indices = data_batch
        features = self.training_buffer['features'][batch_indices].contiguous()
        target_px_b2 = self.training_buffer['target_px'][batch_indices].contiguous(
        )
        image_idxs = self.training_buffer['image_idxs'][batch_indices].contiguous(
        )
        rotations = self.training_buffer['rotations'][batch_indices].contiguous(
        )
        heights = self.training_buffer['heights'][batch_indices].contiguous()
        widths = self.training_buffer['widths'][batch_indices].contiguous()

        _, channels = features.shape

        #################################################################################
        # The forward pass to regress camera poses given only image features.
        #################################################################################

        #
        # (1) Estimate the scene coordinates.
        #
        # Reshape to a "fake" BCHW shape, since it is faster to run through the network
        # compared to the original shape.
        features = features[None, None, ...].view(
            -1, 16, 32, channels).permute(0, 3, 1, 2)  # [B,C,H,W]
        with autocast(enabled=self.config.trainer.use_half):
            pred_scene_coords = self.regressor.get_scene_coordinates(
                features)  # [B,3,H,W]

        # Back to the original shape. Convert to float32 as well.
        pred_scene_coords_b31 = pred_scene_coords.permute(
            0, 2, 3, 1).flatten(0, 2).unsqueeze(-1).float()

        # (2) Estimate the intrinsics. We assume all cameras in the same scene share
        # a same intrinsics.
        Ks_b33 = self.calib_refiner(heights, widths)  # [B,3,3]
        invKs_b33 = torch.linalg.inv(Ks_b33)          # [B,3,3]

        # (3) Get the estimated poses from registered images.
        init_poses_b44 = self._get_registered_image_poses(
            image_idxs, rotations)

        # (4) Optimize the initial camera poses using pose refiner.
        pred_poses_b44, delta_se3 = self.pose_refiner(init_poses_b44)

        #################################################################################
        #                           End of the forward pass
        #################################################################################

        pred_cam_coords_b31, reprojection_error_b1 = compute_reproj_error(
            pred_scene_coords_b31, target_px_b2, Ks_b33, pred_poses_b44,
            self.config.regressor.depth_min,
        )

        loss_valid, loss_invalid, loss = self._compute_reproj_loss(
            invKs_b33, target_px_b2, pred_cam_coords_b31, reprojection_error_b1
        )

        ######################################## End of Loss computation ###########################
        ############################################################################################

        # Jointly optimize the scene regressor, pose refiner, calibration network.
        self.optimizer.zero_grad(set_to_none=True)
        self.pose_optimizer.zero_grad(set_to_none=True)
        self.calib_optimizer.zero_grad(set_to_none=True)

        self.scaler.scale(loss).backward()

        self.scaler.step(self.optimizer)
        self.pose_optimizer.step()
        self.calib_optimizer.step()

        self.scaler.update()

        self.scalars_to_log['train/loss_valid'] = loss_valid.detach().item()
        self.scalars_to_log['train/loss_invalid'] = loss_invalid.detach().item()
        self.scalars_to_log['train/loss'] = loss.detach().item()

        if self.iteration % (self.config.trainer.n_tensorboard) == 0:
            delta_pose_norm = torch.linalg.norm(delta_se3).detach().item() \
                / delta_se3.shape[0]
            _logger.info(
                f"[Mapping] Epoch: {self.epoch}, Iter: {self.iteration:6d}, " +
                f"loss valid: {loss_valid:.3f}, loss invalid: {loss_invalid:.3f}, loss: {loss:.3f}, " +
                f"pose change: {delta_pose_norm:.5f}, " +
                f"fs: {self.calib_refiner.scaler.detach().item():.3f}, lr: {self.cur_lr:.4f}"
            )

        self._check_early_stopping(
            reprojection_error_b1, error_thresh=self.config.trainer.early_stop_thresh)

    def compose_state_dicts(self) -> None:
        self.state_dicts = {
            "models": dict(),
            "optimizers": dict(),
            "schedulers": dict(),
            "meta_data": dict(),
        }

        self.state_dicts["models"]["regressor"] = self.regressor
        self.state_dicts["models"]["pose_refiner"] = self.pose_refiner
        self.state_dicts["models"]["calib_refiner"] = self.calib_refiner

        self.state_dicts["optimizers"]["optimizer"] = self.optimizer
        self.state_dicts["optimizers"]["pose_optimizer"] = self.pose_optimizer
        self.state_dicts["optimizers"]["calib_optimizer"] = self.calib_optimizer

        self.state_dicts["schedulers"]["scheduler"] = self.scheduler

        self.state_dicts["meta_data"]["num_low_error_consecutive_batches"] = \
            self.num_low_error_consecutive_batches

    def update_meta_data(self):
        pass
