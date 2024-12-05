# Code is adapted from: https://github.com/DonGovi/pyramid-detection-3D

import torch
import torch.nn as nn
import torch.nn.functional as F

from conerf.model.backbone.resnet3d import *


def init_conv_weights(layer, weights_std=0.01, bias=0):
    '''
    RetinaNet's layer initialization
    :layer
    :
    '''
    nn.init.xavier_normal_(layer.weight)
    nn.init.constant_(layer.bias.data, val=bias)
    return layer


def conv1x1x1(in_channels, out_channels, **kwargs):
    '''Return a 1x1 convolutional layer with RetinaNet's weight and bias initialization'''

    layer = nn.Conv3d(in_channels, out_channels, kernel_size=1, **kwargs)
    layer = init_conv_weights(layer)

    return layer


def conv3x3x3(in_channels, out_channels, **kwargs):
    '''Return a 3x3 convolutional layer with RetinaNet's weight and bias initialization'''

    layer = nn.Conv3d(in_channels, out_channels, kernel_size=3, **kwargs)
    layer = init_conv_weights(layer)

    return layer


class FeaturePyramid_v1(nn.Module):
    def __init__(self, resnet):
        super(FeaturePyramid_v1, self).__init__()

        self.resnet = resnet

        # applied in a pyramid
        self.pyramid_transformation_1 = conv3x3x3(64, 256, padding=1)
        self.pyramid_transformation_2 = conv1x1x1(256, 256)
        self.pyramid_transformation_3 = conv1x1x1(512, 256)
        self.pyramid_transformation_4 = conv1x1x1(1024, 256)
        self.pyramid_transformation_5 = conv1x1x1(2048, 256)

        # applied after upsampling
        self.upsample_transform_1 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_3 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_4 = conv3x3x3(256, 256, padding=1)

    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        # is this correct? You do lose information on the upscale...
        depth, height, width = scaled_feature.size()[2:]
        return F.interpolate(original_feature, scale_factor=scale_factor)[:, :, :depth, :height, :width]

    def forward(self, x):
        resnet_feature_1, resnet_feature_2, resnet_feature_3, \
            resnet_feature_4, resnet_feature_5 = self.resnet(x)

        # Transform c5 from 2048d to 256d
        pyramid_feature_5 = self.pyramid_transformation_5(resnet_feature_5)
        # Transform c4 from 1024d to 256d
        pyramid_feature_4 = self.pyramid_transformation_4(resnet_feature_4)
        # De-convolution c5 to c4.size
        upsampled_feature_5 = self._upsample(pyramid_feature_5, pyramid_feature_4)
        # Add up-c5 and c4, and conv
        pyramid_feature_4 = self.upsample_transform_4(
            torch.add(upsampled_feature_5, pyramid_feature_4)
        )

        # Transform c3 from 512d to 256d
        pyramid_feature_3 = self.pyramid_transformation_3(resnet_feature_3)
        # De-convolution c4 to c3.size
        upsampled_feature_4 = self._upsample(pyramid_feature_4, pyramid_feature_3)
        # Add up-c4 and c3, and conv
        pyramid_feature_3 = self.upsample_transform_3(
            torch.add(upsampled_feature_4, pyramid_feature_3)
        )
        
        # c2 is 256d, so no need to transform
        pyramid_feature_2 = self.pyramid_transformation_2(resnet_feature_2)
        # De-convolution c3 to c2.size                         
        upsampled_feature_3 = self._upsample(pyramid_feature_3, pyramid_feature_2)
        # Add up-c3 and c2, and conv
        pyramid_feature_2 = self.upsample_transform_2(
            torch.add(upsampled_feature_3, pyramid_feature_2)
        )
        
        # use conv3x3x3 up c1 from 64d to 256d
        pyramid_feature_1 = self.pyramid_transformation_1(resnet_feature_1)
        # De-convolution c2 to c1.size
        upsampled_feature_2 = self._upsample(pyramid_feature_2, pyramid_feature_1)
        # Add up-c2 and c1, and conv
        pyramid_feature_1 = self.upsample_transform_1(
            torch.add(upsampled_feature_2, pyramid_feature_1)
        )

        return pyramid_feature_1                # 8
        # return (pyramid_feature_1,             # 8
        #         pyramid_feature_2,             # 16
        #         pyramid_feature_3)             # 32


class FeaturePyramid_v3(nn.Module):
    def __init__(self, resnet):
        super(FeaturePyramid_v3, self).__init__()

        self.resnet = resnet

        # applied in a pyramid
        self.pyramid_transformation_1 = conv3x3x3(64, 256, padding=1)
        self.pyramid_transformation_2 = conv3x3x3(64, 256, padding=1)
        self.pyramid_transformation_3 = conv3x3x3(128, 256, padding=1)
        self.pyramid_transformation_4 = conv1x1x1(256, 256)
        self.pyramid_transformation_5 = conv1x1x1(512, 256)

        # applied after upsampling
        self.upsample_transform_1 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_2 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_3 = conv3x3x3(256, 256, padding=1)
        self.upsample_transform_4 = conv3x3x3(256, 256, padding=1)

    def _upsample(self, original_feature, scaled_feature, scale_factor=2):
        # is this correct? You do lose information on the upscale...
        depth, height, width = scaled_feature.size()[2:]
        return F.interpolate(original_feature, scale_factor=scale_factor)[:, :, :depth, :height, :width]

    def forward(self, x):
        resnet_feature_1, resnet_feature_2, resnet_feature_3, resnet_feature_4, resnet_feature_5 = self.resnet(x)

        # Transform c5 from 2048d to 256d
        pyramid_feature_5 = self.pyramid_transformation_5(resnet_feature_5)
        # Transform c4 from 1024d to 256d
        pyramid_feature_4 = self.pyramid_transformation_4(resnet_feature_4)
        # De-convolution c5 to c4.size
        upsampled_feature_5 = self._upsample(pyramid_feature_5, pyramid_feature_4)
        # Add up-c5 and c4, and conv
        pyramid_feature_4 = self.upsample_transform_4(
            torch.add(upsampled_feature_5, pyramid_feature_4)
        )

        # Transform c3 from 512d to 256d
        pyramid_feature_3 = self.pyramid_transformation_3(resnet_feature_3)
        # De-convolution c4 to c3.size
        upsampled_feature_4 = self._upsample(pyramid_feature_4, pyramid_feature_3)
        # Add up-c4 and c3, and conv
        pyramid_feature_3 = self.upsample_transform_3(
            torch.add(upsampled_feature_4, pyramid_feature_3)
        )
        
        # c2 is 256d, so no need to transform
        pyramid_feature_2 = self.pyramid_transformation_2(resnet_feature_2)
        # De-convolution c3 to c2.size
        upsampled_feature_3 = self._upsample(pyramid_feature_3, pyramid_feature_2)
        # # Add up-c3 and c2, and conv
        pyramid_feature_2 = self.upsample_transform_2(
            torch.add(upsampled_feature_3, pyramid_feature_2)
        )
        
        # use conv3x3x3 up c1 from 64d to 256d
        pyramid_feature_1 = self.pyramid_transformation_1(resnet_feature_1)
        # De-convolution c2 to c1.size
        upsampled_feature_2 = self._upsample(pyramid_feature_2, pyramid_feature_1)
        # Add up-c2 and c1, and conv
        pyramid_feature_1 = self.upsample_transform_1(
            torch.add(upsampled_feature_2, pyramid_feature_1)
        )

        return pyramid_feature_1                 # 8       
        # return (pyramid_feature_1,             # 8
        #         pyramid_feature_2,             # 16
        #         pyramid_feature_3)             # 32


class FeaturePyramidNet3D(nn.Module):
    backbones = {
        'resnet18': resnet18,
        'resnet34': resnet34,
        'resnet50': resnet50,
        'resnet101': resnet101,
        'resnet152': resnet152
    }

    def __init__(self, in_channels=3, backbone='resnet50', pretrained=False):
        super(FeaturePyramidNet3D, self).__init__()
        
        self.backbone_net = FeaturePyramidNet3D.backbones[backbone](
            in_channels=in_channels,
            pretrained=pretrained
        )	
        
        if backbone == 'resnet50' or backbone == 'resnet101' or backbone == 'resnet152':
            self.feature_pyramid = FeaturePyramid_v1(self.backbone_net)
        else:
            self.feature_pyramid = FeaturePyramid_v3(self.backbone_net)

    def forward(self, x):
        pyramid_features = self.feature_pyramid(x)
        
        return pyramid_features


if __name__ == '__main__':
    # from torch.autograd import Variable
    # x = Variable(torch.rand(1, 3, 64, 64, 64).cuda())

    net = FeaturePyramidNet3D(in_channels=7).cuda()    
    x = torch.load(
        '/home/chenyu/Datasets/nerf_synthetic/eval/chair/block_0/grid_features.pt',
        map_location=torch.device('cuda:0')
    ) # [x_res, y_res, z_res, FEATURE_DIM]
    x = x.permute(3, 2, 0, 1).unsqueeze(dim=0) # [1, FEATURE_DIM, z_res, x_res, y_res]

    for l in net(x):
        print(l.size(), type(l))
