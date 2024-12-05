# pylint: disable=[E1101,C0415,C0200,E1102,C0103]

import os
import collections
import math
from typing import List, NamedTuple
from PIL import Image

import imageio
import tqdm
import trimesh
import torch
import numpy as np

from omegaconf import OmegaConf
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R

from conerf.geometry.camera import Camera


Rays = collections.namedtuple("Rays", ("origins", "viewdirs"))


class BasicPointCloud(NamedTuple):
    points : np.array
    colors : np.array
    normals : np.array


def namedtuple_map(func, tup):
    """Apply `func` to each element of `tup` and cast to `tup`'s namedtuple."""
    return type(tup)(*(None if x is None else func(x) for x in tup))


def visualize_poses(poses, size=0.1, bounding_box=None):
    """
    Visualize camera poses in the axis-aligned bounding box, which
    can be utilized to tune the aabb size.
    """
    # poses: [B, 4, 4]

    axes = trimesh.creation.axis(axis_length=4)
    # box = trimesh.primitives.Box(bounds=[[-1.2, -1.2, -1.2], [1.2, 1.2, 1.2]]).as_outline()
    box = trimesh.primitives.Box(bounds=bounding_box).as_outline()
    box.colors = np.array([[128, 128, 128]] * len(box.entities))
    objects = [axes, box]

    for pose in poses:
        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        segs = np.array([[pos, a], [pos, b], [pos, c], [pos, d], [a, b], [b, c], [c, d], [d, a]])
        segs = trimesh.load_path(segs)
        objects.append(segs)

    trimesh.Scene(objects).show()


def save_colmap_ply(xyz, rgb, path: str):
    num_points = xyz.shape[0]

    file = open(path, 'w')
    file.write("# 3D point list with one line of data per point:\n")
    file.write("#   POINT3D_ID, X, Y, Z, R, G, B, ERROR, " + \
               "TRACK[] as (IMAGE_ID, POINT2D_IDX)\n")
    file.write(f"# Number of points: {num_points}, mean track length: 0\n")
    for i in range(num_points):
        file.write(f'{i} ')
        file.write(f'{xyz[i][0]} {xyz[i][1]} {xyz[i][2]} ' +
                   f'{rgb[i][0]} {rgb[i][1]} {rgb[i][2]} 0 \n')

    file.close()


def save_colmap_images(
    camtoworlds_list: List,
    num_unique_images: int,
    save_dir: str,
    write_empty: bool = False
):
    image_file_path = os.path.join(save_dir, "exp_images.txt")
    file = open(image_file_path, 'w')
    file.write("# Image list with two lines of data per image:\n")
    file.write("#   IMAGE_ID, QW, QX, QY, QZ, TX, TY, TZ, CAMERA_ID, NAME\n")
    file.write("#   POINTS2D[] as (X, Y, POINT3D_ID)\n")
    file.write(f"# Number of images: {num_unique_images}, mean observations per image: {0.0}\n")
    image_id, camera_id = 1, 1
    labels = dict()
    for block_id, camtoworlds in enumerate(camtoworlds_list):
        num_images = camtoworlds.shape[0]
        worldtocams = np.linalg.inv(camtoworlds.numpy())
        rotations, tvecs = worldtocams[:, :3, :3], worldtocams[:, :3, -1]
        for i in range(num_images):
            qvec, tvec = R.from_matrix(rotations[i]).as_quat(), tvecs[i]
            file.write(f"{image_id} {qvec[3]} {qvec[0]} {qvec[1]} {qvec[2]} " + \
                       f"{tvec[0]} {tvec[1]} {tvec[2]} {camera_id} image_{image_id}.jpg \n")
            if write_empty:
                file.write("\n")
            labels[image_id] = block_id
            image_id += 1
    file.close()

    split_result_file = open(
        os.path.join(save_dir, "exp_cluster.txt"),
        "w",
        encoding="utf-8",
    )
    for image_id, label in labels.items():
        print(f'{image_id} {label}', file=split_result_file)
    split_result_file.close()


def compute_rainbow_color(block_id: int, freq: float = 0.4):
    color = torch.zeros(1, 3)
    color[0, 0] = math.sin(freq * block_id + 0) * 0.5 + 0.5
    color[0, 1] = math.sin(freq * block_id + 2) * 0.5 + 0.5
    color[0, 2] = math.sin(freq * block_id + 4) * 0.5 + 0.5
    color *= 255.0
    return color


def minify(basedir, factors=[], resolutions=[], image_dir="images"):
    need_to_load = False

    for factor in factors:
        scaled_image_dir = os.path.join(basedir, f'{image_dir}_{factor}')
        if not os.path.exists(scaled_image_dir):
            need_to_load = True

    for factor in resolutions:
        scaled_image_dir = os.path.join(basedir, f'{image_dir}_{factor[1]}x{factor[0]}')
        if not os.path.exists(scaled_image_dir):
            need_to_load = True

    if not need_to_load:
        return image_dir

    image_dir = os.path.join(basedir, image_dir)
    original_image_dir = image_dir

    imgs = []
    for root, dirs, files in os.walk(image_dir): # pylint: disable=W0612
        for file in files:
            imgs.append(os.path.join(root, file))

    imgs = [
        f for f in imgs if any(f.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG'])
    ]

    for resolution in factors + resolutions:
        if isinstance(resolution, int):
            name = f'{image_dir}_{resolution}'
        else:
            name = f'{image_dir}_{resolution[1]}x{resolution[0]}'
        image_dir = os.path.join(basedir, name)
        if os.path.exists(image_dir):
            continue

        os.makedirs(image_dir)
        pbar = tqdm.trange(len(imgs), desc=f"Resizing images at resolution x{resolution}")

        for img_path in imgs:
            img = Image.open(img_path)
            width, height = img.size
            resize_height = math.ceil(height / int(resolution))
            resize_width = math.ceil(width / int(resolution))
            resized_img = img.resize((resize_width, resize_height))

            # img_name = os.path.split(img_path)[-1]
            img_name_start_index = len(original_image_dir)
            img_name = img_path[img_name_start_index+1:]
            new_img_path = os.path.join(image_dir, img_name)
            new_img_dir = os.path.split(new_img_path)[0]
            if not os.path.exists(new_img_dir):
                os.makedirs(new_img_dir, exist_ok=True)

            imageio.imwrite(new_img_path, resized_img)
            pbar.update(1)

    return image_dir


def compute_nerf_plus_plus_norm(cameras: List[Camera]):
    def get_center_and_diag(camera_centers):
        camera_centers = np.hstack(camera_centers)
        avg_camera_center = np.mean(camera_centers, axis=1, keepdims=True)
        center = avg_camera_center
        dist = np.linalg.norm(camera_centers - center, axis=0, keepdims=True)
        diagonal = np.max(dist)

        return center.flatten(), diagonal

    camera_centers = []
    for camera in cameras:
        camera_centers.append(camera.camera_center.unsqueeze(-1).transpose(0, 1).cpu().numpy())

    diagonal = get_center_and_diag(camera_centers)[-1]
    radius = diagonal * 1.1

    return radius


def fetch_ply(path: str):
    ply_data = PlyData.read(path)
    vertices = ply_data['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T

    return BasicPointCloud(points=positions, colors=colors, normals=normals)


def store_ply(path: str, xyz: np.ndarray, color: np.ndarray):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]

    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, color), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)


def get_block_info_dir(data_dir: str, num_blocks: int = 1, mx: int = None, my: int = None):
    block_info_dir = ""

    if mx is not None and my is not None:
        # assert num_blocks == 1
        block_info_dir = os.path.join(data_dir, f"blocks_{mx}x{my}")
    elif num_blocks > 1:
        assert mx is None and my is None
        block_info_dir = os.path.join(data_dir, f"bipartite_{num_blocks}")

    assert block_info_dir != "", "Invalid configurations!"
    return block_info_dir


def create_dataset(
    config: OmegaConf,
    # train: bool = True,
    split: str = "train",
    num_rays: int = None,
    apply_mask: bool = False,
    device: str = 'cuda:0',
):
    return None
