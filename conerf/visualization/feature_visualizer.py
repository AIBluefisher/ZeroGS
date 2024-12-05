# pylint: disable=E1101

import math

import torch
import torchvision
import torchvision.transforms as transforms

import numpy as np
import cv2


def plot_feature_map(writer, global_step, ray_sampler, feat_maps, prefix=''):
    coarse_feat_map = ray_sampler.target_feat_map[0].transpose(0, 1)
    feat_map_grid = torchvision.utils.make_grid(
        coarse_feat_map, normalize=True, scale_each=True, nrow=8)
    writer.add_image(prefix + 'target_feat_map', feat_map_grid, global_step)

    num_nearby_views = feat_maps[0].shape[0]
    for i in range(num_nearby_views):
        feat_map = feat_maps[0][i].unsqueeze(0).transpose(0, 1)
        # print(f'[DEBUG] feat_map shape: {feat_map}')
        feat_map_grid = torchvision.utils.make_grid(
            feat_map, normalize=True, scale_each=True, nrow=8)
        writer.add_image(
            prefix + f'nearby_feat_map-{i}', feat_map_grid, global_step)


def feature_map_to_heatmap(feat_maps):
    '''
    feat_maps: [C, H, W]
    '''
    # Define a transform to convert the image to tensor
    transform = transforms.ToTensor()

    num_channels = feat_maps.shape[0]
    heat_maps = []

    for i in range(num_channels):
        feat_map = np.asarray(feat_maps)[i]
        # print('feat_map.shape:', feat_map.shape) # [H, W]
        # print('feat_map type:', feat_map.dtype) # float32

        feat_map = np.asarray(feat_map * 255, dtype=np.uint8)  # [0,255]
        # print('feat_map type:', feat_map.dtype) # uint8

        # https://www.sohu.com/a/343215045_120197868
        feat_map = cv2.applyColorMap(feat_map, cv2.COLORMAP_RAINBOW)
        feat_map = transform(feat_map)

        heat_maps.append(feat_map)

    heat_maps = torch.stack(heat_maps, dim=0)  # [C, 3, 25, 25]
    return heat_maps


def feature_maps_to_heatmap(feat_maps):
    '''
    Args:
        feat_maps: [C, H, W]
    Return:
        A composed heat map with shape [H, W]
    '''
    # Define a transform to convert the image to tensor
    transform = transforms.ToTensor()

    # print(f'[DEBUG] feat_maps shape: {feat_maps.shape}')
    [c, h, w] = feat_maps.shape

    heatmap = torch.zeros((h, w))
    weight = []
    feat_maps = np.asarray(feat_maps)

    for i in range(c):
        feat_map = feat_maps[i]
        weight = np.mean(feat_map)
        heatmap[:, :] += weight * feat_map

    heatmap = (heatmap - heatmap.min()) / heatmap.max()  # normalization

    heatmap = np.asarray(heatmap * 255, dtype=np.uint8)
    heatmap = cv2.applyColorMap(heatmap, cv2.COLORMAP_RAINBOW)
    heatmap = transform(heatmap)

    return heatmap


def plot_sampled_feature_map(writer, global_step, target_rgb_feat, rgb_feats, N_rand, prefix='train/'):
    width = int(math.sqrt(N_rand))
    target_rgb_feat = target_rgb_feat.detach().cpu()
    rgb_feats = rgb_feats.detach().cpu()

    target_rgb_feat = target_rgb_feat.permute(
        3, 2, 0, 1).reshape(35, -1, width, width)
    rgb_feats = rgb_feats.permute(3, 2, 0, 1).reshape(35, -1, width, width)
    res_rgb_feats = torch.abs(target_rgb_feat - rgb_feats)
    target_rgb_feat = target_rgb_feat[:, 0, ...]

    # target_feat_map = feature_map_to_heatmap(target_rgb_feat[3:])
    # feat_map_grid = torchvision.utils.make_grid(target_feat_map, normalize=True, scale_each=True, nrow=8)
    # writer.add_image(prefix + f'target_feat_map', feat_map_grid, global_step) # feature map
    target_feat_map = feature_maps_to_heatmap(target_rgb_feat[3:])
    writer.add_image(prefix + f'target_feat_map',
                     target_feat_map, global_step)  # feature map
    writer.add_image(prefix + f'target_rgb_map',
                     target_rgb_feat[0:3], global_step)

    num_nearby_views = rgb_feats.shape[1]
    nearby_feat_maps, nearby_rgb_maps = [], []
    nearby_res_feat_maps, nearby_res_rgb_maps = [], []
    for i in range(num_nearby_views):
        rgb_feat_map = rgb_feats[:, i, ...]
        feat_map = feature_maps_to_heatmap(rgb_feat_map[3:])
        nearby_feat_maps.append(feat_map)
        nearby_rgb_maps.append(rgb_feat_map[0:3])

        res_rgb_feat_map = res_rgb_feats[:, i, ...]
        res_feat_map = feature_maps_to_heatmap(res_rgb_feat_map[3:])
        nearby_res_feat_maps.append(res_feat_map)
        nearby_res_rgb_maps.append(res_rgb_feat_map[0:3])

    nearby_feat_maps = torch.stack(nearby_feat_maps, dim=0)
    nearby_feat_grid = torchvision.utils.make_grid(
        nearby_feat_maps, normalize=True, scale_each=True, nrow=5)
    writer.add_image(prefix + f'nearby_feat_maps',
                     nearby_feat_grid, global_step)

    nearby_rgb_maps = torch.stack(nearby_rgb_maps, dim=0)  # [n_views, 3, h, w]
    nearby_rgb_grid = torchvision.utils.make_grid(
        nearby_rgb_maps, normalize=True, scale_each=True, nrow=5)
    writer.add_image(prefix + f'nearby_rgb_maps', nearby_rgb_grid, global_step)

    nearby_res_feat_maps = torch.stack(nearby_res_feat_maps, dim=0)
    nearby_res_feat_grid = torchvision.utils.make_grid(
        nearby_res_feat_maps, normalize=True, scale_each=True, nrow=5)
    writer.add_image(prefix + f'nearby_res_feat_maps',
                     nearby_res_feat_grid, global_step)

    nearby_res_rgb_maps = torch.stack(
        nearby_res_rgb_maps, dim=0)  # [n_views, 3, h, w]
    nearby_res_rgb_grid = torchvision.utils.make_grid(
        nearby_res_rgb_maps, normalize=True, scale_each=True, nrow=5)
    writer.add_image(prefix + f'nearby_res_rgb_maps',
                     nearby_res_rgb_grid, global_step)
