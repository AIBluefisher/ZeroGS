# pylint: disable=E1101

import logging
import random
import math

import imageio
import torch
import torchvision.transforms.functional as TF
from skimage import color
from skimage import io
from skimage.transform import rotate
from torch.utils.data import Dataset
from torch.utils.data.dataloader import default_collate
from torchvision import transforms

from conerf.datasets.load_colmap import load_colmap

_logger = logging.getLogger(__name__)


class CamLocDataset(Dataset):
    """Camera localization dataset.

    Access to image, calibration and ground truth data given a dataset directory.
    """

    def __init__(
        self,
        # root_dir: str,
        root_fp: str,
        subject_id: str,
        val_interval: int = 0,
        scale: bool = True,
        rotate: bool = True,
        augment: bool = False,
        aug_rotation: int = 15,
        aug_scale_min: float = 2 / 3,
        aug_scale_max: float = 3 / 2,
        aug_black_white: float = 0.1,
        aug_color: float = 0.3,
        factor: int = 8,
        use_half: bool = True,
    ):
        """
        Params:
            root_dir: Folder of the data (training or test).
            augment: Use random data augmentation, note: note supported for mode=2 (RGB-D) since 
                pre-generated eye coordinates cannot be augmented
            aug_rotation: Max 2D image rotation angle, sampled uniformly around 0, both 
                directions, degrees
            aug_scale_min: Lower limit of image scale factor for uniform sampling
            aug_scale_max: Upper limit of image scale factor for uniform sampling
            aug_black_white: Max relative scale factor for image brightness/contrast sampling, 
                e.g. 0.1 -> [0.9,1.1]
            aug_color: Max relative scale factor for image saturation/hue sampling, e.g. 
                0.1 -> [0.9,1.1]
            image_height: RGB images are rescaled to this maximum height (if augmentation is 
                disabled, and in the range [aug_scale_min * image_height, aug_scale_max *
                image_height] otherwise).
            use_half: Enabled if training with half-precision floats.
        """

        self.use_half = use_half
        self.factor = factor
        self.augment = augment
        self.aug_rotation = aug_rotation
        self.aug_scale_min = aug_scale_min
        self.aug_scale_max = aug_scale_max
        self.aug_black_white = aug_black_white
        self.aug_color = aug_color

        data = load_colmap(
            root_fp, subject_id, split='train', factor=factor,
            val_interval=val_interval, scale=scale, rotate=rotate,
        )
        self.rgb_files = data['image_paths']
        self.gt_camtoworlds = data['poses']

        # We use this to iterate over all frames.
        self.valid_file_indices = {i: i for i in range(len(self.rgb_files))}

        # Try to read an image and get its width and height.
        image = imageio.imread(self.rgb_files[0]) # [H,W,3]
        # Use a fixed 480px image height since the convolutional feature backbone
        # is pretrained to ingest images scaled to 480px.
        self.origin_image_height, self.origin_image_width = image.shape[:2]
        # self.image_height = image.shape[0]
        # self.image_width = image.shape[1]
        self.image_height = 480

        # Image transformations. Excluding scale since that can vary batch-by-batch.
        if self.augment:
            self.image_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ColorJitter(
                    brightness=self.aug_black_white, contrast=self.aug_black_white),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4], std=[0.25]),
            ])
        else:
            self.image_transform = transforms.Compose([
                transforms.Grayscale(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4], std=[0.25]),
            ])

    def image(self, idx):
        idx = self.valid_file_indices[idx]
        return self._load_image(idx)

    def image_tensor(self, idx):
        return torch.from_numpy(self.image(idx))

    def resized_image(self, idx, image_height: int, image_width: int = None):
        image = self.image(idx)
        return self._resize_image(image, image_height, image_width)

    def resized_grayscale_image(self, idx, image_height: int):
        color_image_pil = self.resized_image(idx, image_height)
        return color_image_pil, self.image_transform(color_image_pil)

    @staticmethod
    def _resize_image(image, image_height: int, image_width: int = None):
        # Resize a numpy image as PIL. Works slightly better than resizing the tensor
        # using torch's internal function.
        image = TF.to_pil_image(image)
        image = TF.resize(image, image_height) if image_width is None else \
                TF.resize(image, [image_height, image_width])
        return image

    @staticmethod
    def _rotate_image(image, angle, order, mode='constant'):
        # Image is a torch tensor (CxHxW), convert it to numpy as HxWxC.
        image = image.permute(1, 2, 0).numpy()
        # Apply rotation.
        image = rotate(image, angle, order=order, mode=mode)
        # Back to torch tensor.
        image = torch.from_numpy(image).permute(2, 0, 1).float()
        return image

    def _load_image(self, idx):
        image = io.imread(self.rgb_files[idx])

        if len(image.shape) < 3:
            # Convert to RGB if needed.
            image = color.gray2rgb(image)

        return image

    def _get_single_item(self, idx, image_height):
        # Apply index indirection.
        idx = self.valid_file_indices[idx]

        # Load image.
        image = self._load_image(idx)

        # Rescale image.
        image = self._resize_image(image, image_height)

        # Create mask of the same size as the resized image (it's a PIL image at this point).
        image_mask = torch.ones((1, image.size[1], image.size[0]))

        # Apply remaining transforms.
        image = self.image_transform(image)

        pose_rot = torch.eye(4)

        # Apply data augmentation if necessary.
        if self.augment:
            # Generate a random rotation angle.
            angle = random.uniform(-self.aug_rotation, self.aug_rotation)

            # Rotate input image and mask.
            image = self._rotate_image(image, angle, 1, 'reflect')
            image_mask = self._rotate_image(image_mask, angle, order=1, mode='constant')

            # Provide the rotation as well.
            # - pose = pose @ pose_rot
            angle = angle * math.pi / 180.
            pose_rot[0, 0] = math.cos(angle)
            pose_rot[0, 1] = -math.sin(angle)
            pose_rot[1, 0] = math.sin(angle)
            pose_rot[1, 1] = math.cos(angle)

        # Convert to half precision if needed.
        if self.use_half and torch.cuda.is_available():
            image = image.half()

        # Binarize the mask.
        image_mask = image_mask > 0

        # TODO(chenyu): shall we return the augmented status for latter 3D Gaussian Splatting?

        return image, image_mask, pose_rot, idx, str(self.rgb_files[idx])

    def __len__(self):
        return len(self.valid_file_indices)

    def __getitem__(self, idx):
        if self.augment:
            scale_factor = random.uniform(self.aug_scale_min, self.aug_scale_max)
        else:
            scale_factor = 1

        # Target image height. We compute it here in case we are asked for a full batch of tensors
        # because we need to apply the same scale factor to all of them.
        image_height = int(self.image_height * scale_factor)

        if type(idx) == list:
            # Whole batch.
            tensors = [self._get_single_item(i, image_height) for i in idx]
            return default_collate(tensors)
        else:
            # Single element.
            return self._get_single_item(idx, image_height)
