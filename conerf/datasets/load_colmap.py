# pylint: disable=[E1101,C0206,E0401,C0103]

import os
import math
import random
from typing import List, Dict

import imageio
import numpy as np
import torch
import torch.nn.functional as F
import tqdm

from conerf.datasets.utils import minify, store_ply, fetch_ply
from conerf.pycolmap.pycolmap.scene_manager import SceneManager


def find_all_images_with_subdir(image_dir: str) -> List:
    image_names = []
    for root, dirs, files in os.walk(image_dir): # pylint: disable=W0612
        for file in files:
            full_image_name = os.path.join(root, file)
            image_name_start_index = len(image_dir)
            if any(full_image_name.endswith(ex) for ex in ['JPG', 'jpg', 'png', 'jpeg', 'PNG']):
                image_names.append(full_image_name[image_name_start_index+1:])
    image_names.sort()

    return image_names


def load_images(image_indices: np.array, image_paths: List, image_split: str = "train"):
    images = []
    if len(image_paths) == 0:
        return images
    
    pbar = tqdm.trange(len(image_indices), desc=f"Loading {image_split} images")
    for image_index, image_path in enumerate(image_paths):
        if image_index in image_indices:
            # images.append(imageio.imread(image_path))
            images.append(torch.from_numpy(imageio.imread(image_path)).to(torch.uint8))
            pbar.update(1)
    # images = np.stack(images, axis=0)
    return images


def scratch_data(
    indices: np.array,
    image_paths: List,
    normal_paths: List,
    poses: np.array,
    intrinsics: np.array,
    split: str = "train"
) -> Dict:
    val_images = load_images(indices, image_paths, split)
    val_normals = load_images(indices, normal_paths, split+"/normal")
    camtoworlds = torch.from_numpy(poses[indices]).float()
    intrinsics = torch.from_numpy(intrinsics[indices]).float()

    return {
        "rgbs": val_images,
        "normals": val_normals,
        "poses": camtoworlds,
        "intrinsics": intrinsics,
        "image_paths": [image_paths[i] for i in indices],
    }


def load_colmap(
    root_fp: str,
    subject_id: str,
    split: str,
    factor: int = 1,
    val_interval: int = 0,
    multi_blocks: bool = False,
    num_blocks: int = 1,
    bbox_scale_factor: List = [1.0,1.0,1.0],
    scale: bool = True,
    rotate: bool = True,
    model_folder: str = 'sparse',
    load_specified_images: bool = False,
    load_normal: bool = False,
    mx: int = None,
    my: int = None,
):
    assert factor in [1, 2, 4, 8]

    data_dir = os.path.join(root_fp, subject_id)
    colmap_dir = os.path.join(data_dir, model_folder, "0")

    train_image_name = []
    if load_specified_images:
        images_to_train_path = os.path.join(data_dir, "images_to_train.txt")
        if os.path.exists(images_to_train_path):
            file = open(images_to_train_path, 'r', encoding='utf-8')
            line = file.readline()
            while line:
                data = line.split(' ')
                image_id, image_path = int(data[0]), str(data[1])
                train_image_name.append(os.path.split(image_path.strip())[1])
                line = file.readline()

    if mx is not None and my is not None:
        assert num_blocks == mx * my

    if factor != 1:
        minify(basedir=data_dir, factors=[factor])

    ply_path = os.path.join(colmap_dir, "points3D.ply")

    manager = SceneManager(colmap_dir, load_points=True)
    manager.load()

    colmap_image_data = manager.images
    colmap_camera_data = manager.cameras
    image_name_to_image_id = manager.name_to_image_id
    image_names, w2c_mats, intrinsics = [], [], []
    image_index_to_image_id = {}

    # Extract extrinsic & intrinsic matrices.
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in colmap_image_data:
        im_data = colmap_image_data[k]
        if load_specified_images and im_data.name not in train_image_name:
            continue
        image_names.append(im_data.name)

        camera_id = im_data.camera_id
        camera = colmap_camera_data[camera_id]

        intrinsics.append(np.array([
            [camera.fx / factor, 0, camera.cx / factor],
            [0, camera.fy / factor, camera.cy / factor],
            [0, 0, 1]]
        ))

        w2c = np.concatenate([
            np.concatenate([im_data.R(), im_data.tvec.reshape(3, 1)], 1), bottom
        ], axis=0)
        w2c_mats.append(w2c)
    w2c_mats = np.stack(w2c_mats, axis=0)
    intrinsics = np.stack(intrinsics, axis=0)

    # Convert extrinsics to camera-to-world.
    camtoworlds = np.linalg.inv(w2c_mats)

    # Previous Nerf results were generated with images sorted by filename,
    # ensure metrics are reported on the same test set.
    inds = np.argsort(image_names)
    image_names = [image_names[i] for i in inds]
    camtoworlds = camtoworlds[inds]
    intrinsics = intrinsics[inds]
    for image_index, image_name in enumerate(image_names):
        image_id = image_name_to_image_id[image_name]
        image_index_to_image_id[image_index] = image_id

    if scale:
        # normalize the scene
        T, scale = similarity_from_cameras(
            camtoworlds, strict_scaling=False
        )
        camtoworlds = np.einsum("nij, ki -> nkj", camtoworlds, T)
        camtoworlds[:, :3, 3:4] *= scale

        points3d = scale * (T[:3, :3] @ manager.points3D.T + T[:3, 3][..., None]).T # [Np, 3]

        if rotate:
            # Rotate the scene to align with ground plane.
            camtoworlds, points3d, _, _ = normalize_poses(
                torch.from_numpy(camtoworlds).float(),
                torch.from_numpy(points3d).float(),
                up_est_method="ground",
                center_est_method="lookat",
            )
            camtoworlds = camtoworlds.numpy()

    if not os.path.exists(ply_path):
        points3d, colors = manager.points3D, manager.point3D_colors
        print("Converting point3d.bin to .ply " +
              "will happen only the first time you open the scene.")
        store_ply(ply_path, points3d, colors)
    else:
        pcd = fetch_ply(ply_path)
        points3d = pcd.points
        colors = pcd.colors

    image_dir_suffix = f"_{factor}" if factor > 1 else ""
    image_folder = "images" if root_fp.find('MatrixCity') < 0 else ""
    colmap_image_dir = os.path.join(data_dir, image_folder).rstrip('/')
    image_dir = os.path.join(data_dir, image_folder + image_dir_suffix).rstrip('/')
    for d in [image_dir, colmap_image_dir]:
        if not os.path.exists(d):
            raise ValueError(f"Image folder {d} does not exist.")

    # Downsampled images may have different names vs images used for COLMAP,
    # so we need to map between the two sorted lists of files.
    colmap_files = find_all_images_with_subdir(colmap_image_dir)
    image_files = find_all_images_with_subdir(image_dir)
    colmap_to_image = dict(zip(colmap_files, image_files))
    image_paths = [
        os.path.join(image_dir, colmap_to_image[f]) for f in image_names
    ]

    normal_paths = []
    if load_normal:
        normal_dir = os.path.join(data_dir, "normals" + image_dir_suffix)
        for image_path in image_paths:
            ext = os.path.splitext(image_path)[1]
            normal_path = image_path.replace(image_dir, normal_dir).replace(ext, '.png')
            normal_paths.append(normal_path)

    ################################## For test set #################################
    if split == 'test':
        n_test_traj_steps = 60
        camtoworlds = create_spheric_poses(
            torch.from_numpy(camtoworlds)[:, :3, 3], n_steps=n_test_traj_steps
        ).float()
        dummy_image = imageio.imread(image_paths[0])
        test_images = torch.zeros(
            (n_test_traj_steps, *(dummy_image.shape)), dtype=torch.uint8)

        # visualize_poses(camtoworlds)
        return {
            "rgbs": test_images,
            "poses": camtoworlds,
            "intrinsics": torch.from_numpy(intrinsics).float()
        }

    num_images = len(image_paths)
    all_indices = np.arange(num_images)
    all_val_indices = all_indices[all_indices % val_interval == 0] \
        if val_interval > 0 else np.array([])

    ################################# For validation set ###############################
    if split == "val":
        return scratch_data(
            all_val_indices, image_paths, normal_paths, camtoworlds, intrinsics, split)

    ############################ For train set (Non-block mode) ########################
    if not multi_blocks:
        indices = set(all_indices) - set(all_val_indices)
        indices = np.array(list(indices), dtype=np.int64)
        return scratch_data(indices, image_paths, normal_paths, camtoworlds, intrinsics, split)


def similarity_from_cameras(c2w, strict_scaling):
    """
    reference: nerf-factory
    Get a similarity transform to normalize dataset
    from c2w (OpenCV convention) cameras
    :param c2w: (N, 4)
    :return T (4,4) , scale (float)
    """
    t = c2w[:, :3, 3]
    R = c2w[:, :3, :3]

    # (1) Rotate the world so that z+ is the up axis
    # we estimate the up axis by averaging the camera up axes
    ups = np.sum(R * np.array([0, -1.0, 0]), axis=-1)
    world_up = np.mean(ups, axis=0)
    world_up /= np.linalg.norm(world_up)

    up_camspace = np.array([0.0, -1.0, 0.0])
    c = (up_camspace * world_up).sum()
    cross = np.cross(world_up, up_camspace)
    skew = np.array(
        [
            [0.0, -cross[2], cross[1]],
            [cross[2], 0.0, -cross[0]],
            [-cross[1], cross[0], 0.0],
        ]
    )
    if c > -1:
        R_align = np.eye(3) + skew + (skew @ skew) * 1 / (1 + c) # pylint: disable=C0103
    else:
        # In the unlikely case the original data has y+ up axis,
        # rotate 180-deg about x axis
        R_align = np.array( # pylint: disable=C0103
            [[-1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]])

    #  R_align = np.eye(3) # DEBUG
    R = R_align @ R
    fwds = np.sum(R * np.array([0, 0.0, 1.0]), axis=-1)
    t = (R_align @ t[..., None])[..., 0]

    # (2) Recenter the scene using camera center rays
    # find the closest point to the origin for each camera's center ray
    nearest = t + (fwds * -t).sum(-1)[:, None] * fwds

    # median for more robustness
    translate = -np.median(nearest, axis=0)

    #  translate = -np.mean(t, axis=0)  # DEBUG

    transform = np.eye(4)
    transform[:3, 3] = translate
    transform[:3, :3] = R_align

    # (3) Rescale the scene using camera distances
    scale_fn = np.max if strict_scaling else np.median
    scale = 1.0 / scale_fn(np.linalg.norm(t + translate, axis=-1))

    return transform, scale


def get_center(pts):
    center = pts.mean(0)
    dis = (pts - center[None, :]).norm(p=2, dim=-1)
    mean, std = dis.mean(), dis.std()
    q25, q75 = torch.quantile(dis, 0.25), torch.quantile(dis, 0.75)
    valid = (dis > mean - 1.5 * std) & (dis < mean + 1.5 * std) & \
            (dis > mean - (q75 - q25) * 1.5) & (dis < mean + (q75 - q25) * 1.5)
    center = pts[valid].mean(0)
    return center


def normalize_poses(
    poses,
    pts,
    up_est_method: str = "ground",
    center_est_method: str = "lookat"
):
    if center_est_method == 'camera':
        # estimation scene center as the average of all camera positions
        center = poses[..., :3, 3].mean(0)
    elif center_est_method == 'lookat':
        # estimation scene center as the average of the intersection of
        # selected pairs of camera rays
        cams_ori = poses[..., :3, 3]
        cams_dir = poses[:, :3, :3] @ torch.as_tensor([0., 0., -1.])
        cams_dir = F.normalize(cams_dir, dim=-1)
        A = torch.stack([cams_dir, -cams_dir.roll(1, 0)], dim=-1)
        b = -cams_ori + cams_ori.roll(1, 0)
        t = torch.linalg.lstsq(A, b).solution
        center = (torch.stack([cams_dir, cams_dir.roll(1, 0)], dim=-1) *
                  t[:, None, :] + torch.stack([
                      cams_ori, cams_ori.roll(1, 0)
                  ], dim=-1)).mean((0, 2))
    elif center_est_method == 'point':
        # first estimation scene center as the average of all camera positions
        # later we'll use the center of all points bounded by the cameras as the
        # final scene center
        center = poses[..., :3, 3].mean(0)
    else:
        raise NotImplementedError(
            f'Unknown center estimation method: {center_est_method}')

    if up_est_method == 'ground':
        # estimate up direction as the normal of the estimated ground plane
        # use RANSAC to estimate the ground plane in the point cloud
        import pyransac3d as pyrsc # pylint: disable=C0415

        # Fix the seed every time we the load the dataset.
        random.seed(0)
        ground = pyrsc.Plane()

        plane_eq = ground.fit(pts.numpy(), thresh=0.01)[0]
        # A, B, C, D in Ax + By + Cz + D = 0
        plane_eq = torch.as_tensor(plane_eq)
        # plane normal as up direction
        z = F.normalize(plane_eq[:3], dim=-1)
        signed_distance = (
            torch.cat([pts, torch.ones_like(pts[..., 0:1])], dim=-1) * plane_eq
        ).sum(-1)
        if signed_distance.mean() < 0:
            z = -z  # flip the direction if points lie under the plane
    elif up_est_method == 'camera':
        # estimate up direction as the average of all camera up directions
        z = F.normalize((poses[..., 3] - center).mean(0), dim=0)
    else:
        raise NotImplementedError(
            f'Unknown up estimation method: {up_est_method}')

    # new axis
    y = torch.as_tensor([z[1], -z[0], 0.])
    x = F.normalize(y.cross(z), dim=0)
    y = z.cross(x)

    if center_est_method == 'point':
        # rotation
        Rc = torch.stack([x, y, z], dim=1)
        R = Rc.T
        inv_trans = torch.cat([
            torch.cat([R, torch.as_tensor([[0., 0., 0.]]).T], dim=1),
            torch.as_tensor([[0., 0., 0., 1.]])
        ], dim=0)
        poses_norm = (inv_trans @ poses)[:, :3]
        pts = (inv_trans @ torch.cat(
            [pts, torch.ones_like(pts[:, 0:1])], dim=-1
        )[..., None])[:, :3, 0]

        # translation and scaling
        poses_min, poses_max = poses_norm[..., 3].min(
            0)[0], poses_norm[..., 3].max(0)[0]
        pts_fg = pts[
            (poses_min[0] < pts[:, 0]) & (pts[:, 0] < poses_max[0]) &
            (poses_min[1] < pts[:, 1]) & (pts[:, 1] < poses_max[1])]
        center = get_center(pts_fg)
        tc = center.reshape(3, 1)
        t = -tc
        inv_trans = torch.cat([
            torch.cat([torch.eye(3), t], dim=1),
            torch.as_tensor([[0., 0., 0., 1.]])
        ], dim=0)
        poses_norm = (inv_trans @ poses)
    else:
        # rotation and translation
        Rc = torch.stack([x, y, z], dim=1)
        tc = center.reshape(3, 1)
        R, t = Rc.T, -Rc.T @ tc
        inv_trans = torch.cat([
            torch.cat([R, t], dim=1), torch.as_tensor([[0., 0., 0., 1.]])
        ], dim=0)

        poses_norm = (inv_trans @ poses)  # (N_images, 4, 4)
        pts = (R @ pts.T + t).T

    return poses_norm, pts, R, t


def create_spheric_poses(cameras, n_steps=120):
    center = torch.as_tensor([0., 0., 0.], dtype=cameras.dtype)
    mean_d = (cameras - center[None, :]).norm(p=2, dim=-1).mean()
    mean_h = cameras[:, 2].mean()
    r = (mean_d ** 2 - mean_h ** 2).sqrt()
    up = torch.as_tensor([0., 0., -1.], dtype=center.dtype)

    camtoworlds = []
    for theta in torch.linspace(0, 2 * math.pi, n_steps):
        cam_pos = torch.stack([r * theta.cos(), r * theta.sin(), mean_h])
        l = F.normalize(center - cam_pos, p=2, dim=0)
        s = F.normalize(l.cross(up), p=2, dim=0)
        u = F.normalize(s.cross(l), p=2, dim=0)
        c2w = torch.cat([torch.stack([s, u, l], dim=1),
                        cam_pos[:, None]], axis=1)
        c2w = torch.cat(
            [c2w, torch.tensor([[0, 0, 0, 1]], dtype=c2w.dtype)], axis=0)

        camtoworlds.append(c2w)

    camtoworlds = torch.stack(camtoworlds, dim=0)

    return camtoworlds
