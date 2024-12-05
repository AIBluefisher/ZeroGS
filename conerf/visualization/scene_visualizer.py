# pylint: disable=E1101

import os
import math
import time
import threading

from typing import List
from plyfile import PlyData

import numpy as np
import open3d as o3d


image_counter = -1      # Counter for saved images
saving_images = False   # Flag to control image saving
save_thread = None      # Thread for saving images
output_dir = ""
MAX_NUM_IMAGES = 60

available_colors = [
    [1, 0, 0], [0, 1, 0], [0, 0, 1],
    [1, 1, 0], [1, 0, 1], [0, 1, 1]
]


# Function to convert HSL to RGB
def hsl_to_rgb(h: float, s: float, l: float):
    c = (1 - abs(2 * l - 1)) * s
    x = c * (1 - abs(math.fmod(h / 60.0, 2) - 1))
    m = l - c / 2;
    r, g, b = 0, 0, 0

    if (0 <= h and h < 60):
        r, g, b = c, x, 0
    elif (60 <= h and h < 120):
        r, g, b = x, c, 0
    elif (120 <= h and h < 180):
        r, g, b = 0, c, x
    elif (180 <= h and h < 240):
        r, g, b = 0, x, c
    elif (240 <= h and h < 300):
        r, g, b = x, 0, c
    else:
        r, g, b = c, 0, x

    return [r + m, g + m, b + m]


def compute_rainbow_color(index: int, total_num: int):
    hue = (360.0 / total_num) * index  # Distribute hue across 360 degrees
    saturation = 1.0  # full saturation
    lightness = 0.5   #  medium lightness

    color = hsl_to_rgb(hue, saturation, lightness)

    return color


def fetch_ply(path: str):
    ply_data = PlyData.read(path)
    vertices = ply_data['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T
    colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0

    return positions, colors


def increase_point_size_callback(vis):
    vis.get_render_option().point_size += 1


def decrease_point_size_callback(vis):
    vis.get_render_option().point_size -= 1


def rotate_view(vis):
    # if START_ROTATE:
    ctr = vis.get_view_control()
    ctr.rotate(10.0, 0.0)

    # vis.update_geometry()
    vis.poll_events()
    vis.update_renderer()

    return False


def save_image(vis):
    global image_counter

    if image_counter < MAX_NUM_IMAGES:
        image_counter += 1  # Increment the counter
        image_name = os.path.join(output_dir, f"screenshot_{image_counter:05d}.png")

        # Update the visualizer before capturing the image
        vis.poll_events()
        vis.update_renderer()

        vis.capture_screen_image(image_name, True)
        print(f"Image saved as {image_name}")


def save_images_periodically(vis):
    global saving_images
    while saving_images:
        save_image(vis)
        time.sleep(1)  # Wait for 1 second before saving the next image

def toggle_image_saving(vis):
    global saving_images, save_thread
    saving_images = not saving_images  # Toggle saving state
    if saving_images:
        save_thread = threading.Thread(target=save_images_periodically, args=(vis,))
        save_thread.start()  # Start the saving thread
        print("Started saving images every 1 second.")
    else:
        if save_thread is not None:
            save_thread.join()  # Wait for the thread to finish
            print("Stopped saving images.")


def visualize_open3d(poses, labels: np.ndarray = None, size=0.1, rainbow_color=False):
    """
    Visualize camera poses in the axis-aligned bounding box, which
    can be utilized to tune the aabb size.
    """
    # poses: [B, 4, 4]

    points, lines, colors = [], [], []
    for i, pose in enumerate(poses):
        label = labels[i] if labels is not None else 0

        # a camera is visualized with 8 line segments.
        pos = pose[:3, 3]
        a = pos + size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        b = pos - size * pose[:3, 0] + size * pose[:3, 1] + size * pose[:3, 2]
        c = pos - size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]
        d = pos + size * pose[:3, 0] - size * pose[:3, 1] + size * pose[:3, 2]

        points.append(pos)
        points.append(a)
        points.append(b)
        points.append(c)
        points.append(d)

        lines.append([i * 5 + 0, i * 5 + 1])
        lines.append([i * 5 + 0, i * 5 + 2])
        lines.append([i * 5 + 0, i * 5 + 3])
        lines.append([i * 5 + 0, i * 5 + 4])
        lines.append([i * 5 + 1, i * 5 + 2])
        lines.append([i * 5 + 2, i * 5 + 3])
        lines.append([i * 5 + 3, i * 5 + 4])
        lines.append([i * 5 + 4, i * 5 + 1])

        for k in range(8): # pylint: disable=W0612
            if rainbow_color:
                colors.append(compute_rainbow_color(i, len(poses)))
            else:
                colors.append(available_colors[label])

    return points, lines, colors


def visualize_scene(
    ply_files: List[str],
    camera_poses: np.ndarray,
    camera_pose_labels: np.ndarray = None,
    size: float = 0.1,
    overwrite_ply_colors: bool = False,
):
    vis = o3d.visualization.VisualizerWithKeyCallback()
    # vis = o3d.visualization.Visualizer()
    vis.create_window()

    point_clouds = []
    for i, ply_file in enumerate(ply_files):
        points, colors = fetch_ply(ply_file)
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if overwrite_ply_colors:
            color = np.array(available_colors[i])
            color = np.tile(color, (points.shape[0], 1))
            pcd.colors = o3d.utility.Vector3dVector(color)
        else:
            pcd.colors = o3d.utility.Vector3dVector(colors)
        point_clouds.append(pcd)

    # vis.add_geometry(pcd)
    points, lines, colors = visualize_open3d(camera_poses, camera_pose_labels, size)
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    for pcd in point_clouds:
        vis.add_geometry(pcd)
    vis.add_geometry(line_set)

    # Create axes
    axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    vis.add_geometry(axes)
    # vis.get_render_option().show_coordinate_frame = True
    vis.get_render_option().line_width = 5
    vis.get_render_option().point_size = 1

    vis.poll_events()
    vis.update_renderer()

    vis.register_animation_callback(rotate_view)
    vis.register_key_callback(ord("+"), increase_point_size_callback)
    vis.register_key_callback(ord("-"), decrease_point_size_callback)

    vis.run()
    vis.destroy_window()


def visualize_single_scene(
    point_cloud: List[str],
    camera_poses: np.ndarray,
    camera_pose_labels: np.ndarray = None,
    size: float = 0.1,
    rainbow_color: bool = False,
    output_directory: str = None,
):
    """
        # pcd = o3d.geometry.PointCloud()
        # pcd.points = o3d.utility.Vector3dVector(points)
        # pcd.colors = o3d.utility.Vector3dVector(colors)
        # visualize_single_scene(pcd, cam_to_world, size=0.02, rainbow_color=True)
    """
    global output_dir
    output_dir = output_directory

    vis = o3d.visualization.VisualizerWithKeyCallback()
    # vis = o3d.visualization.Visualizer()
    vis.create_window()

    # vis.add_geometry(pcd)
    points, lines, colors = visualize_open3d(
        camera_poses, camera_pose_labels, size, rainbow_color
    )
    line_set = o3d.geometry.LineSet()
    line_set.points = o3d.utility.Vector3dVector(points)
    line_set.lines = o3d.utility.Vector2iVector(lines)
    line_set.colors = o3d.utility.Vector3dVector(colors)

    vis.add_geometry(point_cloud)
    vis.add_geometry(line_set)

    # Create axes
    # axes = o3d.geometry.TriangleMesh.create_coordinate_frame(size=1.0, origin=[0, 0, 0])
    # vis.add_geometry(axes)
    # vis.get_render_option().show_coordinate_frame = True
    vis.get_render_option().line_width = 5
    vis.get_render_option().point_size = 1

    vis.poll_events()
    vis.update_renderer()

    # Get the view control
    view_control = vis.get_view_control()
    # Flip the Y-axis by setting the up vector
    view_control.set_up([0, -1, 0])

    vis.register_animation_callback(rotate_view)
    vis.register_key_callback(ord("+"), increase_point_size_callback)
    vis.register_key_callback(ord("-"), decrease_point_size_callback)
    vis.register_key_callback(ord('S'), toggle_image_saving)

    vis.run()
    vis.destroy_window()
