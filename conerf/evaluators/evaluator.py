# pylint: disable=[E1101,W0621]

import os
import copy
import json

from typing import List, Literal
from omegaconf import OmegaConf

import tqdm
import torch
import torch.nn.functional as F
import numpy as np
import lpips

from conerf.datasets.utils import create_dataset
from conerf.loss.ssim_torch import ssim
from conerf.base.checkpoint_manager import CheckPointManager
from conerf.utils.utils import get_subdirs


def color_correct(img: np.ndarray, ref: np.ndarray, num_iters: int = 5, eps: float = 0.5 / 255):
    """Warp `img` to match the colors in `ref_img`."""
    if img.shape[-1] != ref.shape[-1]:
        raise ValueError(
            f'img\'s {img.shape[-1]} and ref\'s {ref.shape[-1]} channels must match'
        )
    num_channels = img.shape[-1]
    img_mat = img.reshape([-1, num_channels])
    ref_mat = ref.reshape([-1, num_channels])
    is_unclipped = lambda z: (z >= eps) & (z <= (1 - eps))  # z \in [eps, 1-eps].
    mask0 = is_unclipped(img_mat)

    # Because the set of saturated pixels may change after solving for a
    # transformation, we repeatedly solve a system `num_iters` times and update
    # our estimate of which pixels are saturated.
    for _ in range(num_iters):
        # Construct the left hand side of a linear system that contains a quadratic
        # expansion of each pixel of `img`.
        a_mat = []
        for c in range(num_channels):
            a_mat.append(img_mat[:, c:(c + 1)] * img_mat[:, c:])  # Quadratic term.
        a_mat.append(img_mat)  # Linear term.
        a_mat.append(np.ones_like(img_mat[:, :1]))  # Bias term.
        a_mat = np.concatenate(a_mat, axis=-1)
        warp = []
        for c in range(num_channels):
            # Construct the right hand side of a linear system containing each color
            # of `ref`.
            b = ref_mat[:, c]
            # Ignore rows of the linear system that were saturated in the input or are
            # saturated in the current corrected color estimate.
            mask = mask0[:, c] & is_unclipped(img_mat[:, c]) & is_unclipped(b)
            ma_mat = np.where(mask[:, None], a_mat, 0)
            mb = np.where(mask, b, 0) # pylint: disable=C0103
            # Solve the linear system. We're using the np.lstsq instead of np because
            # it's significantly more stable in this case, for some reason.
            w = np.linalg.lstsq(ma_mat, mb, rcond=-1)[0]
            assert np.all(np.isfinite(w))
            warp.append(w)
        warp = np.stack(warp, axis=-1)
        # Apply the warp to update img_mat.
        img_mat = np.clip(np.matmul(a_mat, warp), 0, 1)
    corrected_img = np.reshape(img_mat, img.shape)

    return corrected_img


def compute_psnr(gt_image, pred_image, eps=1e-6):
    """
    Args:
        gt_image: ground truth image
        pred_image: predicted image
        eps: parameter to prevent dividing by zero.
    Return:
        PSNR
    """
    mse = F.mse_loss(gt_image, pred_image)
    psnr = -10.0 * torch.log(mse + eps) / np.log(10.0)
    return psnr


def compute_ssim(gt_image, pred_image):
    """
    Args:
        gt_image: ground truth image
        pred_image: predicted image
    Return:
        SSIM
    """
    return ssim(gt_image, pred_image).item()


def compute_lpips(lpips_loss, gt_image, pred_image):
    """
    Args:
        lpips_loss: loss model to compute lpips
        gt_image: ground truth image
        pred_image: predicted image
    Return:
        LPIPs
    """
    return lpips_loss(gt_image * 2 - 1, pred_image * 2 - 1).item()


class Evaluator():
    """Abstract class for evaluating implicit neural representation models."""

    def __init__(
        self,
        config: OmegaConf,
        load_train_data: bool = False,
        trainset = None,
        load_val_data: bool = True,
        valset = None,
        load_test_data: bool = False,
        testset = None,
        models: List = None,
        meta_data: List = None,
        verbose: bool = False,
        device: str = "cuda",
    ) -> None:
        self.config = config
        self.device = device
        self.verbose = verbose

        self.models = models
        self.model_iterations = list()
        self.meta_data = meta_data
        self.train_dataset = trainset
        self.val_dataset = valset
        self.lpips_loss = lpips.LPIPS(net="alex", verbose=False).to(self.device)

        self.output_dir = os.path.join(
            self.config.dataset.root_dir, "out", config.expname)
        os.makedirs(self.output_dir, exist_ok=True)

        self.eval_dir = os.path.join(self.output_dir, "eval")
        os.makedirs(self.eval_dir, exist_ok=True)

        if load_train_data and trainset is None:
            self.load_dataset(True, False, False)
        if load_val_data and valset is None:
            self.load_dataset(False, True, False)
        if load_test_data and testset is None:
            self.load_dataset(False, False, True)

        self.setup_metadata()
        if models is None:
            self.load_model()

        self.global_model = None

    def load_dataset(
        self,
        load_train_data: bool = False,
        load_val_data: bool = True,
        load_test_data: bool = False,
    ):
        """Loading train dataset or validation dataset if required."""
        if load_train_data:
            self.train_dataset = create_dataset(
                config=self.config,
                split=self.config.dataset.train_split,
                num_rays=self.config.dataset.num_rays,
                apply_mask=self.config.dataset.apply_mask,
                device=self.device
            )

        if load_val_data:
            self.val_dataset = create_dataset(
                config=self.config,
                split=self.config.dataset.val_split,
                num_rays=None,
                apply_mask=self.config.dataset.apply_mask,
                device=self.device,
            )

        if load_test_data:
            self.test_dataset = create_dataset(
                config=self.config,
                split="test",
                num_rays=None,
                apply_mask=self.config.dataset.apply_mask,
                device=self.device,
            )

    def setup_metadata(self):
        """Set up meta data that are required to initialize/evaluate a model."""
        # meta data for construction models.
        meta_data = {
            "aabb": None,
            "unbounded": None,
            "grid_resolution": None,
            "contraction_type": None,
            "near_plane": None,
            "far_plane": None,
            "render_step_size": None,
            "alpha_thre": None,
            "cone_angle": None,
            "grid_levels": None,
            "camera_poses": None,
        }
        if self.config.dataset.multi_blocks:
            meta_data["block_id"] = None

        return meta_data

    def load_model(self):
        """Function for loading model (Should be implemented for child class)."""
        kwargs = self._prepare_model_init_params(model_type="local")

        self.meta_data, self.models = [], []
        ckpt_manager = CheckPointManager(verbose=False)

        input_model_dir = os.path.join(
            self.config.dataset.root_dir, 'out', self.config.expname)

        assert os.path.exists(input_model_dir), \
            f"input model directory does not exist: {input_model_dir}"
        if self.config.dataset.multi_blocks:
            model_dirs = get_subdirs(input_model_dir, "block_")
        else:
            model_dirs = [input_model_dir]

        pbar = tqdm.trange(len(model_dirs), desc="Loading Models", leave=True)
        for model_dir in model_dirs:
            local_config = copy.deepcopy(self.config.trainer)
            local_config.ckpt_path = os.path.join(model_dir, 'model.pth')
            assert os.path.exists(local_config.ckpt_path), \
                f"checkpoint does not exist: {local_config.ckpt_path}"

            # Load meta data at first.
            meta_data = self.setup_metadata()
            iteration = ckpt_manager.load(local_config, meta_data=meta_data)
            self.meta_data.append(meta_data)
            self.model_iterations.append(iteration)

            # Load model parameters.
            if "appear_embedding" in self.config:
                kwargs["appear_embedding"] = self.config.appear_embedding
                kwargs["num_images"] = meta_data['camera_poses'].shape[0]

            model = self._build_networks(
                self.config.neural_field_type,
                meta_data['aabb'],
                meta_data['grid_resolution'],
                meta_data['grid_levels'],
                **kwargs,
            )
            ckpt_manager.load(local_config, models={'model': model})

            self.models.append(model)

            pbar.update(1)

    def _prepare_model_init_params(self, model_type: Literal["global", "local"] = "local"):
        raise NotImplementedError

    def _build_networks(self, *args, **kwargs):
        raise NotImplementedError

    @torch.no_grad()
    def _eval(self, data, model, meta_data, eval_dir, image_index):
        raise NotImplementedError

    @torch.no_grad()
    def _export_mesh(self, model, iteration, mesh_dir):
        raise NotImplementedError

    def eval(
        self,
        iteration: int = None,
        split: Literal["val", "test"] = "val",
    ) -> dict:
        """
        Main logic for evaluation.
        """
        metrics = dict()
        if split == "val":
            assert self.val_dataset is not None
            dataset = self.val_dataset
        elif split == "test":
            assert self.test_dataset is not None
            dataset = self.test_dataset
        else:
            if self.verbose:
                print(f'[WARNING] {split} set does not exist!')
            return

        eval_dir = os.path.join(self.eval_dir, split)
        os.makedirs(eval_dir, exist_ok=True)

        num_blocks = len(self.models)
        if self.global_model is not None:
            self.models.append(self.global_model)

        meta_data = self.meta_data[0]
        for k, model in enumerate(self.models):
            model.eval()

            # meta_data = self.meta_data[k]
            iteration = self.model_iterations[k] if iteration is None else iteration

            val_dir = eval_dir
            if k >= num_blocks and self.global_model is not None:
                val_dir = os.path.join(eval_dir, "global")
            elif self.config.dataset.multi_blocks:
                val_dir = os.path.join(eval_dir, f"block_{k}")
            os.makedirs(val_dir, exist_ok=True)

            if self.verbose:
                print(f'Results are saving to: {val_dir}')

            image_dir = os.path.join(val_dir, "images")
            os.makedirs(image_dir, exist_ok=True)

            pbar = tqdm.trange(
                len(dataset), desc=f"Validating {self.config.expname}", leave=False
            )
            psnrs, ssims, lpips = [], [], []

            for i in range(len(dataset)):  # pylint: disable=C0200
                data = dataset[i]

                psnr, ssim, lpip = self._eval(
                    data, model, meta_data, image_dir, i)

                psnrs.append(psnr)
                ssims.append(ssim)
                lpips.append(lpip)

                pbar.update(1)

            avg_psnr = sum(psnrs) / len(psnrs)
            avg_ssim = sum(ssims) / len(ssims)
            avg_lpips = sum(lpips) / len(lpips)

            metric_key = k if k < num_blocks else "global"
            metrics[metric_key] = {
                'iteration': iteration,
                'psnr': avg_psnr,
                'ssim': avg_ssim,
                'lpips': avg_lpips,
            }

        if self.global_model is not None:
            self.models.pop(-1)

        metric_file = os.path.join(eval_dir, 'metrics.json')
        json_obj = json.dumps(metrics, indent=4)
        if self.verbose:
            print(f'Saving metrics to {metric_file}')
        with open(metric_file, 'a', encoding='utf-8') as file:
            file.write(json_obj)

        return metrics

    def export_mesh(self, iteration: int = None):
        """
        Exporting mesh.
        """
        num_blocks = len(self.models)
        if self.global_model is not None:
            self.models.append(self.global_model)

        for k, model in enumerate(self.models):
            model.eval()

            iteration = self.model_iterations[k] if iteration is None else iteration

            val_dir = self.eval_dir
            if k >= num_blocks and self.global_model is not None:
                val_dir = os.path.join(self.eval_dir, "global")
            elif self.config.dataset.multi_blocks:
                val_dir = os.path.join(self.eval_dir, f"block_{k}")
            os.makedirs(val_dir, exist_ok=True)

            if self.verbose:
                print(f'Mesh are saving to: {val_dir}')

            mesh_dir = os.path.join(val_dir, "meshes")
            os.makedirs(mesh_dir, exist_ok=True)
            self._export_mesh(model, iteration, mesh_dir)

        if self.global_model is not None:
            self.models.pop(-1)
