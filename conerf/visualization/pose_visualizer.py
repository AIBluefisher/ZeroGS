# pylint: disable=E1101

from typing import List

import torch
import matplotlib.pyplot as plt
import numpy as np
from easydict import EasyDict as edict


def to_hom(X):
    # get homogeneous coordinates of the input
    X_hom = torch.cat([X, torch.ones_like(X[..., :1])], dim=-1)

    return X_hom


def get_camera_mesh(pose, depth=1):
    vertices = torch.tensor([[-0.5, -0.5, 1],
                             [0.5, -0.5, 1],
                             [0.5,  0.5, 1],
                             [-0.5,  0.5, 1],
                             [0,    0, 0]]) * depth

    faces = torch.tensor([[0, 1, 2],
                          [0, 2, 3],
                          [0, 1, 4],
                          [1, 2, 4],
                          [2, 3, 4],
                          [3, 0, 4]])

    # vertices = camera.cam2world(vertices[None], pose)
    vertices = to_hom(vertices[None]) @ pose.transpose(-1, -2)

    wire_frame = vertices[:, [0, 1, 2, 3, 0, 4, 1, 2, 4, 3]]

    return vertices, faces, wire_frame


def merge_wire_frames(wire_frame):
    wire_frame_merged = [[], [], []]
    for w in wire_frame:
        wire_frame_merged[0] += [float(n) for n in w[:, 0]] + [None]
        wire_frame_merged[1] += [float(n) for n in w[:, 1]] + [None]
        wire_frame_merged[2] += [float(n) for n in w[:, 2]] + [None]

    return wire_frame_merged


def merge_meshes(vertices, faces):
    mesh_N, vertex_N = vertices.shape[:2]
    faces_merged = torch.cat([faces+i*vertex_N for i in range(mesh_N)], dim=0)
    vertices_merged = vertices.view(-1, vertices.shape[-1])

    return vertices_merged, faces_merged


def merge_centers(centers):
    center_merged = [[], [], []]

    for c1, c2 in zip(*centers):
        center_merged[0] += [float(c1[0]), float(c2[0]), None]
        center_merged[1] += [float(c1[1]), float(c2[1]), None]
        center_merged[2] += [float(c1[2]), float(c2[2]), None]

    return center_merged


@torch.no_grad()
def visualize_cameras(
    vis,
    step: int = 0,
    poses: List = [],
    cam_depth: float = 0.5,
    colors: List = ["blue", "magenta"],
    plot_dist: bool = True
):
    win_name = "gt_pred"
    data = []

    # set up plots
    centers = []
    for pose, color in zip(poses, colors):
        pose = pose.detach().cpu()
        vertices, faces, wire_frame = get_camera_mesh(pose, depth=cam_depth)
        center = vertices[:, -1]
        centers.append(center)

        # camera centers
        data.append(dict(
            type="scatter3d",
            x=[float(n) for n in center[:, 0]],
            y=[float(n) for n in center[:, 1]],
            z=[float(n) for n in center[:, 2]],
            mode="markers",
            marker=dict(color=color, size=3),
        ))

        # colored camera mesh
        vertices_merged, faces_merged = merge_meshes(vertices, faces)

        data.append(dict(
            type="mesh3d",
            x=[float(n) for n in vertices_merged[:, 0]],
            y=[float(n) for n in vertices_merged[:, 1]],
            z=[float(n) for n in vertices_merged[:, 2]],
            i=[int(n) for n in faces_merged[:, 0]],
            j=[int(n) for n in faces_merged[:, 1]],
            k=[int(n) for n in faces_merged[:, 2]],
            flatshading=True,
            color=color,
            opacity=0.05,
        ))

        # camera wire_frame
        wire_frame_merged = merge_wire_frames(wire_frame)
        data.append(dict(
            type="scatter3d",
            x=wire_frame_merged[0],
            y=wire_frame_merged[1],
            z=wire_frame_merged[2],
            mode="lines",
            line=dict(color=color,),
            opacity=0.3,
        ))

    if plot_dist:
        # distance between two poses (camera centers)
        center_merged = merge_centers(centers[:2])
        data.append(dict(
            type="scatter3d",
            x=center_merged[0],
            y=center_merged[1],
            z=center_merged[2],
            mode="lines",
            line=dict(color="red", width=4,),
        ))

        if len(centers) == 4:
            center_merged = merge_centers(centers[2:4])
            data.append(dict(
                type="scatter3d",
                x=center_merged[0],
                y=center_merged[1],
                z=center_merged[2],
                mode="lines",
                line=dict(color="red", width=4,),
            ))

    # send data to visdom
    vis._send(dict(
        data=data,
        win="poses",
        eid=win_name,
        layout=dict(
            title=f"({step})",
            autosize=True,
            margin=dict(l=30, r=30, b=30, t=30,),
            showlegend=False,
            yaxis=dict(
                scaleanchor="x",
                scaleratio=1,
            )
        ),
        opts=dict(title=f"{win_name} poses ({step})",),
    ))


def plot_save_poses(
    cam_depth: float,
    fig,
    pose: torch.Tensor,
    pose_ref: torch.Tensor = None,
    path: str = None,
    ep=None,
    axis_len: float = 1.0,
):
    # get the camera meshes
    _, _, cam = get_camera_mesh(pose, depth=cam_depth)
    cam = cam.numpy()

    if pose_ref is not None:
        _, _, cam_ref = get_camera_mesh(pose_ref, depth=cam_depth)
        cam_ref = cam_ref.numpy()

    # set up plot window(s)
    plt.title(f"epoch {ep}")
    ax1 = fig.add_subplot(121, projection="3d")
    ax2 = fig.add_subplot(122, projection="3d")
    setup_3D_plot(
        ax1, elev=-90, azim=-90,
        lim=edict(x=(-axis_len, axis_len), y=(-axis_len,
                  axis_len), z=(-axis_len, axis_len))
    )
    setup_3D_plot(
        ax2, elev=0, azim=-90,
        lim=edict(x=(-axis_len, axis_len), y=(-axis_len,
                  axis_len), z=(-axis_len, axis_len))
    )
    ax1.set_title("forward-facing view", pad=0)
    ax2.set_title("top-down view", pad=0)
    plt.subplots_adjust(left=0, right=1, bottom=0,
                        top=0.95, wspace=0, hspace=0)
    plt.margins(tight=True, x=0, y=0)

    # plot the cameras
    N = len(cam)
    color = plt.get_cmap("gist_rainbow")
    for i in range(N):
        if pose_ref is not None:
            ax1.plot(cam_ref[i, :, 0], cam_ref[i, :, 1],
                     cam_ref[i, :, 2], color=(0.1, 0.1, 0.1), linewidth=1)
            ax2.plot(cam_ref[i, :, 0], cam_ref[i, :, 1],
                     cam_ref[i, :, 2], color=(0.1, 0.1, 0.1), linewidth=1)
            ax1.scatter(cam_ref[i, 5, 0], cam_ref[i, 5, 1],
                        cam_ref[i, 5, 2], color=(0.1, 0.1, 0.1), s=40)
            ax2.scatter(cam_ref[i, 5, 0], cam_ref[i, 5, 1],
                        cam_ref[i, 5, 2], color=(0.1, 0.1, 0.1), s=40)
        c = np.array(color(float(i) / N)) * 0.8
        ax1.plot(cam[i, :, 0], cam[i, :, 1], cam[i, :, 2], color=c)
        ax2.plot(cam[i, :, 0], cam[i, :, 1], cam[i, :, 2], color=c)
        ax1.scatter(cam[i, 5, 0], cam[i, 5, 1], cam[i, 5, 2], color=c, s=40)
        ax2.scatter(cam[i, 5, 0], cam[i, 5, 1], cam[i, 5, 2], color=c, s=40)

    png_fname = f"{path}/{ep}.png"
    plt.savefig(png_fname, dpi=75)
    # clean up
    plt.clf()


def setup_3D_plot(ax, elev, azim, lim=None):
    ax.xaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.yaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.zaxis.set_pane_color((1.0, 1.0, 1.0, 0.0))
    ax.xaxis._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 1)
    ax.yaxis._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 1)
    ax.zaxis._axinfo["grid"]["color"] = (0.9, 0.9, 0.9, 1)
    ax.xaxis.set_tick_params(labelsize=8)
    ax.yaxis.set_tick_params(labelsize=8)
    ax.zaxis.set_tick_params(labelsize=8)
    ax.set_xlabel("X", fontsize=16)
    ax.set_ylabel("Y", fontsize=16)
    ax.set_zlabel("Z", fontsize=16)
    ax.set_xlim(lim.x[0], lim.x[1])
    ax.set_ylim(lim.y[0], lim.y[1])
    ax.set_zlim(lim.z[0], lim.z[1])
    ax.view_init(elev=elev, azim=azim)
