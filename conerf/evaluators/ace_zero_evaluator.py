# pylint: disable=E1101

import os
import copy
import logging
import json
from typing import List, Literal
from omegaconf import OmegaConf
from pathlib import Path

import torch
import tqdm
from torch.utils.data import DataLoader
from torchvision.transforms import ToTensor
import numpy as np
import matplotlib.pyplot as plt

from conerf.base.checkpoint_manager import CheckPointManager
from conerf.datasets.ace_camera_loc_dataset import CamLocDataset
from conerf.datasets.utils import save_colmap_ply, save_colmap_images
from conerf.evaluators.evaluator import Evaluator
from conerf.geometry.align_poses import align_ate_c2b_use_a2b
from conerf.model.scene_regressor.ace_network import Regressor
from conerf.model.scene_regressor.ace_util import save_point_cloud, get_pixel_grid
from conerf.model.scene_regressor.calibr import Calibr
from conerf.model.scene_regressor.pose_refine_network import PoseRefineNetwork
from conerf.trainers.ace_zero_trainer import (
    register_image, pose_dict_to_tensor, evaluate_camera_alignment,
    compute_reproj_error,
)
from conerf.visualization.pose_visualizer import plot_save_poses
from scripts.preprocess.read_write_model import Camera, write_cameras_text

_logger = logging.getLogger(__name__)


class AceZeroEvaluator(Evaluator):
    """Evaluator for ace0.
    """

    def __init__(
        self,
        config: OmegaConf,
        load_train_data: bool = False,
        trainset=None,
        load_val_data: bool = True,
        valset=None,
        load_test_data: bool = False,
        testset=None,
        models: List = None,
        meta_data: List = None,
        verbose: bool = False,
        device: str = "cuda"
    ) -> None:
        super().__init__(
            config,
            load_train_data,
            trainset,
            load_val_data,
            valset,
            load_test_data,
            testset,
            [],
            meta_data,
            verbose,
            device
        )

    def _build_networks(self, *args, **kwargs):
        # (1) Scene Coordinate Regressor.
        # Create network using the state dict of the pretrained encoder.
        encoder_state_dict = torch.load(
            self.config.dataset.encoder_path, map_location='cpu')
        regressor = Regressor.create_from_encoder(
            encoder_state_dict=encoder_state_dict,
            num_head_blocks=self.config.regressor.num_head_blocks,
            use_homogeneous=self.config.regressor.use_homogeneous,
        ).to(self.device)
        regressor.eval()

        # (2) Camera pose refine network.
        pose_refiner = PoseRefineNetwork().to(self.device)
        pose_refiner.eval()

        # (3) Calibration refinement embedding.
        calib_refiner = Calibr(self.device).to(self.device)
        calib_refiner.eval()

        return regressor, pose_refiner, calib_refiner

    def setup_metadata(self):
        pass

    def load_dataset(
        self,
        load_train_data: bool = False,
        load_val_data: bool = True,
        load_test_data: bool = False,
    ):
        self.register_dataset = CamLocDataset(
            root_fp=self.config.dataset.root_dir,
            subject_id=self.config.dataset.scene,
            use_half=self.config.trainer.get("use_half", False),
            factor=self.config.dataset.factor,
            scale=self.config.dataset.get("scale", True),
            rotate=self.config.dataset.get("rotate", True),
            augment=False,
            aug_rotation=False,
        )

    def load_model(self, path: str):
        self.meta_data, self.models = [], []  # pylint: disable=W0201
        ckpt_manager = CheckPointManager(verbose=False)

        local_config = copy.deepcopy(self.config.trainer)
        local_config.ckpt_path = os.path.join(path)
        assert os.path.exists(local_config.ckpt_path), \
            f"checkpoint does not exist: {local_config.ckpt_path}"

        regressor, pose_refiner, calib_refiner = self._build_networks()
        iteration = ckpt_manager.load(
            local_config,
            models={
                'regressor': regressor,
                'pose_refiner': pose_refiner,
                'calib_refiner': calib_refiner
            },
            optimizers=None,
            schedulers=None,
        )

        return regressor, pose_refiner, calib_refiner, iteration

    @torch.no_grad()
    def eval(
        self, iteration: int = None, split: Literal["val", "test"] = "val",
    ):
        eval_dir = os.path.join(self.eval_dir, "val")
        os.makedirs(eval_dir, exist_ok=True)

        image_dir = os.path.join(eval_dir, "images")
        os.makedirs(image_dir, exist_ok=True)

        metrics = dict()

        register_dataloader = DataLoader(
            dataset=self.register_dataset,
            shuffle=False,
            num_workers=8,
        )

        model_dir = os.path.join(
            self.config.dataset.root_dir, 'out', self.config.expname, 'model')
        models_path = sorted(Path(model_dir).iterdir())
        num_models = len(models_path)
        points3d, colors = [], []

        pbar = tqdm.trange(num_models, desc="Registering images", leave=False)

        for i, model_path in enumerate(models_path):
            reg_image_idx_to_pose = {}
            filtered_reg_image_idx_to_pose = {}
            regressor, pose_refiner, calib_refiner, iteration = self.load_model(
                path=model_path)

            for image, _, _, idx, _ in register_dataloader:
                idx = int(idx)
                image = image.to(self.device, non_blocking=True)  # [B,1,H,W]

                out_pose, scene_coords, inlier_count, _ = register_image(
                    regressor,
                    image,
                    calib_refiner,
                    self.config.pose_estimator.hypotheses // 2,
                    self.config.pose_estimator.reproj_thresh,
                    self.config.pose_estimator.inlier_alpha,
                    self.config.pose_estimator.max_pixel_error,
                    1,  # self.config.dataset.factor,
                    max_hypotheses_tries=16,
                    device=self.device,
                )  # pose is from camera to world.

                out_pose, _ = pose_refiner(out_pose.unsqueeze( # pylint: disable=E1102
                    0).to(self.device))
                out_pose = out_pose.squeeze(0)
                reg_image_idx_to_pose[idx] = out_pose
                if inlier_count >= self.config.pose_estimator.min_inlier_count:
                    filtered_reg_image_idx_to_pose[idx] = out_pose

                image_height, image_width = image.shape[-2:]
                if i == num_models - 1:
                    feat_height, feat_width = scene_coords.shape[-2:]
                    target_px_b2 = get_pixel_grid(
                        feat_height, feat_width).permute(1, 2, 0).reshape(-1, 2)
                    color_image = self.register_dataset.resized_image(
                        idx, feat_height, feat_width)
                    color = ToTensor()(color_image).permute(1, 2, 0).reshape(-1, 3)
                    points = scene_coords.squeeze(
                        0).permute(1, 2, 0).reshape(-1, 3)

                    num_points = points.shape[0]
                    poses_b44 = out_pose[None, ...].repeat(
                        num_points, 1, 1).to('cpu')
                    Ks = calib_refiner(  # pylint: disable=E1102
                        torch.tensor([feat_height], device=self.device),
                        torch.tensor([feat_width], device=self.device)
                    )
                    Ks_b33 = Ks.repeat(num_points, 1, 1).to('cpu')
                    _, reproj_error_b1 = compute_reproj_error(
                        points.unsqueeze(-1),
                        target_px_b2,
                        Ks_b33,
                        poses_b44,
                        self.config.regressor.depth_min,
                    )
                    valid_points_mask = reproj_error_b1.squeeze(-1) < 2
                    # TODO(chenyu): remove points with large reprojection errors.
                    colors.append(color[valid_points_mask])
                    points3d.append(points[valid_points_mask])

            if len(filtered_reg_image_idx_to_pose) > 0:
                filtered_idxs, filtered_pred_poses = pose_dict_to_tensor(
                    filtered_reg_image_idx_to_pose)
                filtered_gt_poses = self.register_dataset.gt_camtoworlds[filtered_idxs]
                filtered_pred_poses = align_ate_c2b_use_a2b(
                    filtered_pred_poses, filtered_gt_poses)[0]

                fig = plt.figure(figsize=(16, 8))
                plot_save_poses(
                    self.config.dataset.cam_depth, fig, filtered_pred_poses.to('cpu'),
                    pose_ref=filtered_gt_poses.to('cpu'),
                    path=image_dir, ep=f'filtered_pose_{i:03d}', axis_len=self.config.dataset.axis_len,
                )

            if len(reg_image_idx_to_pose) > 0:
                idxs, pred_poses = pose_dict_to_tensor(reg_image_idx_to_pose)
                gt_poses = self.register_dataset.gt_camtoworlds[idxs]
                pred_poses = align_ate_c2b_use_a2b(pred_poses, gt_poses)[0]

                fig = plt.figure(figsize=(16, 8))
                plot_save_poses(
                    self.config.dataset.cam_depth, fig, pred_poses.to('cpu'),
                    pose_ref=gt_poses.to('cpu'),
                    path=image_dir, ep=f'pose_{i:03d}', axis_len=self.config.dataset.axis_len,
                )

                pose_error = evaluate_camera_alignment(pred_poses, gt_poses)
                metrics[iteration] = pose_error
                metrics[iteration]['Reg Images'] = \
                    f'{str(len(filtered_pred_poses))}/{len(self.register_dataset)}'
                _logger.info(
                    f"[Iter {iteration}] rotation error: {pose_error['R_error_mean']:.4f}, " +
                    f"[Iter {iteration}] translation error: {pose_error['t_error_mean']:.4f}"
                )
            
                plt.cla()
                plt.close(fig)

            # Export to COLMAP formats for the last checkpoint.
            if i == num_models - 1:
                sparse_model_dir = os.path.join(eval_dir, "ACE0_COLMAP")
                os.makedirs(sparse_model_dir, exist_ok=True)

                # TODO(chenyu): need to align with the resolution of original image shape.
                K = calib_refiner(  # pylint: disable=E1102
                    torch.tensor([image_height], device=self.device),
                    torch.tensor([image_width], device=self.device)
                )[0].to('cpu').numpy()
                fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2]
                params = [fx, fy, cx, cy]
                camera = Camera(
                    id=1, model="PINHOLE", width=image_width, height=image_height, params=params)

                points3d = torch.concat(points3d, dim=0)
                colors = torch.concat(colors, dim=0)

                ply_path = os.path.join(sparse_model_dir, "points3D.ply")
                save_point_cloud(points3d, colors, ply_path)
                write_cameras_text({0: camera}, os.path.join(
                    sparse_model_dir, "cameras.txt"))
                save_colmap_images([pred_poses], len(
                    pred_poses), sparse_model_dir, write_empty=False)
                save_colmap_ply(points3d, (colors.to('cpu').numpy() * 255).astype(np.uint8),
                                os.path.join(sparse_model_dir, "points3D.txt"))

            pbar.update(1)

        # Convert pose images to a video.
        video_name = os.path.join(eval_dir, "poses.mp4")
        os.system(
            f"ffmpeg -framerate 5 -i {image_dir}/pose_%3d.png " +
            f"-c:v libx264 -pix_fmt yuv420p {video_name}"
        )
        filtered_video_name = os.path.join(eval_dir, "filtered_poses.mp4")
        os.system(
            f"ffmpeg -framerate 5 -i {image_dir}/filtered_pose_%3d.png " +
            f"-c:v libx264 -pix_fmt yuv420p {filtered_video_name}"
        )

        # Save metrics file.
        metric_file = os.path.join(eval_dir, 'metrics.json')
        json_obj = json.dumps(metrics, indent=4)
        if self.verbose:
            print(f'Saving metrics to {metric_file}')
        with open(metric_file, 'w', encoding='utf-8') as json_file:
            json_file.write(json_obj)

        return metrics
