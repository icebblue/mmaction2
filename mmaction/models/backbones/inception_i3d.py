from typing import Dict, List, Optional, Sequence, Tuple, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from mmengine.model import BaseModule
from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import load_checkpoint
from mmengine.model.weight_init import constant_init, xavier_init

from mmaction.registry import MODELS


class MaxPool3dSamePadding(nn.MaxPool3d):
    def compute_pad(self, dim, s):
        if s % self.stride[dim] == 0:
            return max(self.kernel_size[dim] - self.stride[dim], 0)
        else:
            return max(self.kernel_size[dim] - (s % self.stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        batch, channel, t, h, w = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = [pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b]
        x = F.pad(x, pad)
        return super(MaxPool3dSamePadding, self).forward(x)


class Unit3D(nn.Module):
    def __init__(self, in_channels,
                 output_channels,
                 kernel_shape=(1, 1, 1),
                 stride=(1, 1, 1),
                 padding=0,
                 activation_fn=F.relu,
                 use_batch_norm=True,
                 use_bias=False,
                 padding_valid_spatial=False,
                 name='unit_3d'):
        """Initializes Unit3D module."""
        super(Unit3D, self).__init__()

        self._output_channels = output_channels
        self._kernel_shape = kernel_shape
        self._stride = stride
        self._use_batch_norm = use_batch_norm
        self._activation_fn = activation_fn
        self._use_bias = use_bias
        self.name = name
        self.padding = padding
        self.padding_valid_spatial = padding_valid_spatial

        self.conv3d = nn.Conv3d(in_channels=in_channels,
                                out_channels=self._output_channels,
                                kernel_size=self._kernel_shape,
                                stride=self._stride,
                                padding=0,
                                # we always want padding to be 0 here.
                                # We will dynamically pad based on input size in forward function
                                bias=self._use_bias)

        if self._use_batch_norm:
            self.bn = nn.BatchNorm3d(
                self._output_channels, eps=0.001, momentum=0.01)

    def compute_pad(self, dim, s):
        if s % self._stride[dim] == 0:
            return max(self._kernel_shape[dim] - self._stride[dim], 0)
        else:
            return max(self._kernel_shape[dim] - (s % self._stride[dim]), 0)

    def forward(self, x):
        # compute 'same' padding
        (batch, channel, t, h, w) = x.size()
        pad_t = self.compute_pad(0, t)
        pad_h = self.compute_pad(1, h)
        pad_w = self.compute_pad(2, w)

        pad_t_f = pad_t // 2
        pad_t_b = pad_t - pad_t_f
        pad_h_f = pad_h // 2
        pad_h_b = pad_h - pad_h_f
        pad_w_f = pad_w // 2
        pad_w_b = pad_w - pad_w_f

        pad = [pad_w_f, pad_w_b, pad_h_f, pad_h_b, pad_t_f, pad_t_b]
        if self.padding_valid_spatial:
            pad = [0, 0, 0, 0, pad_t_f, pad_t_b]

        if self.padding == -1:
            pad = [0, 0, 0, 0, 0, 0]

        x = F.pad(x, pad)

        x = self.conv3d(x)
        if self._use_batch_norm:
            x = self.bn(x)
        if self._activation_fn is not None:
            x = self._activation_fn(x)
        return x


class InceptionModule(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0],
                         kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1],
                          kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2],
                          kernel_shape=[3, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3],
                          kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4],
                          kernel_shape=[3, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[3, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5],
                          kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)


class InceptionModule_2D(nn.Module):
    def __init__(self, in_channels, out_channels, name):
        super(InceptionModule_2D, self).__init__()

        self.b0 = Unit3D(in_channels=in_channels, output_channels=out_channels[0],
                         kernel_shape=[1, 1, 1], padding=0,
                         name=name + '/Branch_0/Conv3d_0a_1x1')
        self.b1a = Unit3D(in_channels=in_channels, output_channels=out_channels[1],
                          kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_1/Conv3d_0a_1x1')
        self.b1b = Unit3D(in_channels=out_channels[1], output_channels=out_channels[2],
                          kernel_shape=[1, 3, 3],
                          name=name + '/Branch_1/Conv3d_0b_3x3')
        self.b2a = Unit3D(in_channels=in_channels, output_channels=out_channels[3],
                          kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_2/Conv3d_0a_1x1')
        self.b2b = Unit3D(in_channels=out_channels[3], output_channels=out_channels[4],
                          kernel_shape=[1, 3, 3],
                          name=name + '/Branch_2/Conv3d_0b_3x3')
        self.b3a = MaxPool3dSamePadding(kernel_size=[1, 3, 3],
                                        stride=(1, 1, 1), padding=0)
        self.b3b = Unit3D(in_channels=in_channels, output_channels=out_channels[5],
                          kernel_shape=[1, 1, 1], padding=0,
                          name=name + '/Branch_3/Conv3d_0b_1x1')
        self.name = name

    def forward(self, x):
        b0 = self.b0(x)
        b1 = self.b1b(self.b1a(x))
        b2 = self.b2b(self.b2a(x))
        b3 = self.b3b(self.b3a(x))
        return torch.cat([b0, b1, b2, b3], dim=1)
    

@MODELS.register_module()
class TopI3D(BaseModule):

    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
    )

    def __init__(self, 
                 name: str = 'inception_i3d', 
                 in_channels: int = 3,
                 freeze_bn: bool = False, 
                 freeze_bn_affine: bool = False,
                 out_layers: Sequence[str] = ('Mixed_5c', ),
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None, 
                 **kwargs) -> None:
        
        for out_layer in out_layers:
            if out_layer not in self.VALID_ENDPOINTS:
                raise ValueError('Unknown layer %s' % out_layer)
        
        super().__init__(init_cfg=init_cfg)

        self._out_layers = out_layers
        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'
        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64,
                                            kernel_shape=[1, 7, 7],
                                            stride=(2, 2, 2), padding=[3, 3, 3],
                                            name=name + end_point)

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64,
                                            kernel_shape=[1, 1, 1], padding=0,
                                            name=name + end_point)

        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192,
                                            kernel_shape=[1, 3, 3], padding=1,
                                            name=name + end_point)

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule_2D(192, [64, 96, 128, 16, 32, 32],
                                                        name + end_point)

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule_2D(256, [128, 128, 192, 32, 96, 64],
                                                        name + end_point)

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                          padding=0)

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule_2D(128 + 192 + 96 + 64,
                                                        [192, 96, 208, 16, 48, 64], name + end_point)

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule_2D(192 + 208 + 48 + 64,
                                                        [160, 112, 224, 24, 64, 64], name + end_point)

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64,
                                                     [128, 128, 256, 24, 64, 64], name + end_point)

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64,
                                                     [112, 144, 288, 32, 64, 64], name + end_point)

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64,
                                                     [256, 160, 320, 32, 128, 128],
                                                     name + end_point)

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                          padding=0)

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128,
                                                     [256, 160, 320, 32, 128, 128],
                                                     name + end_point)

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128,
                                                     [384, 192, 384, 48, 128, 128],
                                                     name + end_point)
        
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])


    def init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) :
                xavier_init(m, distribution='uniform')
            if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                constant_init(m, 1)
    
    def forward(self, x: torch.Tensor):
        outs = []
        for end_point in self.VALID_ENDPOINTS:
            x = self._modules[end_point](x)
            if end_point in self._out_layers:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)
    
    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if self._freeze_bn and mode:
            for _, m in self.named_modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)


@MODELS.register_module()
class I3D(BaseModule):
    
    VALID_ENDPOINTS = (
        'Conv3d_1a_7x7',
        'MaxPool3d_2a_3x3',
        'Conv3d_2b_1x1',
        'Conv3d_2c_3x3',
        'MaxPool3d_3a_3x3',
        'Mixed_3b',
        'Mixed_3c',
        'MaxPool3d_4a_3x3',
        'Mixed_4b',
        'Mixed_4c',
        'Mixed_4d',
        'Mixed_4e',
        'Mixed_4f',
        'MaxPool3d_5a_2x2',
        'Mixed_5b',
        'Mixed_5c',
    )

    def __init__(self, 
                 name: str = 'inception_i3d', 
                 in_channels: int = 3,
                 freeze_bn: bool = False, 
                 freeze_bn_affine: bool = False,
                 pretrained: Optional[str] = None,
                 out_layers: Sequence[str] = ('Mixed_5c', ),
                 init_cfg: Optional[Union[Dict, List[Dict]]] = None, 
                 **kwargs) -> None:
        
        for out_layer in out_layers:
            if out_layer not in self.VALID_ENDPOINTS:
                raise ValueError('Unknown layer %s' % out_layer)
        
        super().__init__(init_cfg=init_cfg)

        self._freeze_bn = freeze_bn
        self._freeze_bn_affine = freeze_bn_affine
        self.pretrained = pretrained
        self._out_layers = out_layers

        self.end_points = {}
        end_point = 'Conv3d_1a_7x7'

        self.end_points[end_point] = Unit3D(in_channels=in_channels, output_channels=64,
                                            kernel_shape=[7, 7, 7],
                                            stride=(2, 2, 2), padding=[3, 3, 3],
                                            name=name + end_point)

        end_point = 'MaxPool3d_2a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)

        end_point = 'Conv3d_2b_1x1'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=64,
                                            kernel_shape=[1, 1, 1], padding=0,
                                            name=name + end_point)
        
        end_point = 'Conv3d_2c_3x3'
        self.end_points[end_point] = Unit3D(in_channels=64, output_channels=192,
                                            kernel_shape=[3, 3, 3], padding=1,
                                            name=name + end_point)

        end_point = 'MaxPool3d_3a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[1, 3, 3], stride=(1, 2, 2),
                                                          padding=0)

        end_point = 'Mixed_3b'
        self.end_points[end_point] = InceptionModule(192, [64, 96, 128, 16, 32, 32],
                                                     name + end_point)

        end_point = 'Mixed_3c'
        self.end_points[end_point] = InceptionModule(256, [128, 128, 192, 32, 96, 64],
                                                     name + end_point)

        end_point = 'MaxPool3d_4a_3x3'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[3, 3, 3], stride=(2, 2, 2),
                                                          padding=0)

        end_point = 'Mixed_4b'
        self.end_points[end_point] = InceptionModule(128 + 192 + 96 + 64,
                                                     [192, 96, 208, 16, 48, 64], name + end_point)

        end_point = 'Mixed_4c'
        self.end_points[end_point] = InceptionModule(192 + 208 + 48 + 64,
                                                     [160, 112, 224, 24, 64, 64], name + end_point)

        end_point = 'Mixed_4d'
        self.end_points[end_point] = InceptionModule(160 + 224 + 64 + 64,
                                                     [128, 128, 256, 24, 64, 64], name + end_point)

        end_point = 'Mixed_4e'
        self.end_points[end_point] = InceptionModule(128 + 256 + 64 + 64,
                                                     [112, 144, 288, 32, 64, 64], name + end_point)

        end_point = 'Mixed_4f'
        self.end_points[end_point] = InceptionModule(112 + 288 + 64 + 64,
                                                     [256, 160, 320, 32, 128, 128],
                                                     name + end_point)

        end_point = 'MaxPool3d_5a_2x2'
        self.end_points[end_point] = MaxPool3dSamePadding(kernel_size=[2, 2, 2], stride=(2, 2, 2),
                                                          padding=0)

        end_point = 'Mixed_5b'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128,
                                                     [256, 160, 320, 32, 128, 128],
                                                     name + end_point)

        end_point = 'Mixed_5c'
        self.end_points[end_point] = InceptionModule(256 + 320 + 128 + 128,
                                                     [384, 192, 384, 48, 128, 128],
                                                     name + end_point)
        
        for k in self.end_points.keys():
            self.add_module(k, self.end_points[k])
        

    def init_weights(self, pretrained: Optional[str] = None) -> None:
        """Initiate the parameters either from existing checkpoint or from
        scratch.

        Args:
            pretrained (str | None): The path of the pretrained weight. Will
                override the original `pretrained` if set. The arg is added to
                be compatible with mmdet. Defaults to None.
        """
        if pretrained:
            self.pretrained = pretrained
        if isinstance(self.pretrained, str):
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')

            load_checkpoint(self, self.pretrained, strict=False, logger=logger)

        elif self.pretrained is None:
            for m in self.modules():
                if isinstance(m, nn.Conv3d) or isinstance(m, nn.Conv2d) :
                    xavier_init(m, distribution='uniform')
                if isinstance(m, nn.BatchNorm3d) or isinstance(m, nn.BatchNorm2d):
                    constant_init(m, 1)
        else:
            raise TypeError('pretrained must be a str or None')
    
    def forward(self, x: torch.Tensor):
        outs = []
        for end_point in self.VALID_ENDPOINTS:
            x = self._modules[end_point](x)
            if end_point in self._out_layers:
                outs.append(x)
        if len(outs) == 1:
            return outs[0]

        return tuple(outs)
    
    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if self._freeze_bn and mode:
            for _, m in self.named_modules():
                if isinstance(m, nn.BatchNorm3d):
                    m.eval()
                    if self._freeze_bn_affine:
                        m.weight.requires_grad_(False)
                        m.bias.requires_grad_(False)