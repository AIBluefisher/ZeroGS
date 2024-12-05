import os
import argparse

import torch
import numpy as np
import open3d as o3d

from conerf.datasets.realworld import similarity_from_cameras, normalize_poses
from conerf.datasets.utils import compute_bounding_box3D, points_in_bbox3D
from conerf.pycolmap.pycolmap.scene_manager import SceneManager
from conerf.visualization.scene_visualizer import visualize_single_scene


def config_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument("--colmap_dir",
                        type=str,
                        default="",
                        help="absolute path of config file")
    parser.add_argument("--output_dir",
                        type=str,
                        default="",
                        help="absolute path of config file")

    args = parser.parse_args()

    return args


if __name__ == '__main__':
    args = config_parser()
    rotate = False

    if not os.path.exists(args.output_dir):
        os.makedirs(args.output_dir)

    # (1) Loading camera poses and 3D points.
    manager = SceneManager(args.colmap_dir, load_points=False)
    manager.load()

    ply_path = os.path.join(args.colmap_dir, "points3D.ply")
    pcd = o3d.io.read_point_cloud(ply_path)
    points = np.asarray(pcd.points)
    colors = np.asarray(pcd.colors)
    num_points = np.asarray(pcd.points).shape[0]
    print(f'num points: {num_points}')

    colmap_image_data = manager.images
    colmap_camera_data = manager.cameras

    w2c_mats = []
    bottom = np.array([0, 0, 0, 1]).reshape(1, 4)
    for k in colmap_image_data:
        im_data = colmap_image_data[k]
        w2c = np.concatenate([
            np.concatenate(
                [im_data.R(), im_data.tvec.reshape(3, 1)], 1), bottom
        ], axis=0)
        w2c_mats.append(w2c)
    w2c_mats = np.stack(w2c_mats, axis=0)
    cam_to_world = np.linalg.inv(w2c_mats)

    # (2) Normalize the scene.
    T, scale = similarity_from_cameras(
        cam_to_world, strict_scaling=False
    )
    cam_to_world = np.einsum("nij, ki -> nkj", cam_to_world, T)
    cam_to_world[:, :3, 3:4] *= scale

    points = scale * (T[:3, :3] @ points.T + T[:3, 3][..., None]).T  # [Np, 3]

    # (3) Rotate the scene to align with ground plane.
    if rotate:
        down_pcd = pcd.voxel_down_sample(voxel_size=0.1)
        points_for_est_normal = np.asarray(down_pcd.points)
        print(
            f'num points for estimating normal: {points_for_est_normal.shape}')
        cam_to_world, _, R, t = normalize_poses(
            torch.from_numpy(cam_to_world).float(),  # pylint: disable=E1101
            torch.from_numpy(points_for_est_normal).float(
            ),      # pylint: disable=E1101
            up_est_method="ground",
            center_est_method="lookat",
        )
        cam_to_world = cam_to_world.numpy()
        points[:, :] = (R @ points.T + t).T

    # (4) Compute bounding box to exclude points outside the bounding box.
    aabb = compute_bounding_box3D(
        torch.from_numpy(cam_to_world[..., :, -1]),  # pylint: disable=E1101
        scale_factor=[7, 7, 7],  # [4.0,4.0,4.0]
    ).numpy()
    valid_point_indices = points_in_bbox3D(points, aabb).reshape(-1)
    points = points[valid_point_indices]
    colors = colors[valid_point_indices]
    colors = np.clip(colors, 0, 1)
    print(f'num points: {points.shape[0]}')

    pcd.points = o3d.utility.Vector3dVector(points)
    pcd.colors = o3d.utility.Vector3dVector(colors)

    # (5) Downsample points if there are too many.
    if num_points > 2000000:
        down_pcd = pcd.voxel_down_sample(voxel_size=0.005)
        points = np.asarray(down_pcd.points)
        colors = np.asarray(down_pcd.colors)
        print(f'points shape: {points.shape}')

        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
    else:
        colors = np.asarray(pcd.colors)
        pcd.colors = o3d.utility.Vector3dVector(colors)

    visualize_single_scene(
        pcd,
        cam_to_world,
        size=0.05,
        rainbow_color=True,
        output_directory=args.output_dir
    )

    video_filename = os.path.join(args.output_dir, "zero_gs_scene.mp4")
    os.system(f"ffmpeg -framerate 10 -i {args.output_dir}/screenshot_%05d.png -c:v libx264 " +
              f"-pix_fmt yuv420p {video_filename}"
              )
