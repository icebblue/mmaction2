# Copyright (c) OpenMMLab. All rights reserved.
from typing import Dict, Optional, Tuple, Union
from mmengine.model.weight_init import kaiming_init, constant_init,  normal_init
from torch import Tensor, nn

from mmengine.logging import MMLogger
from mmengine.runner.checkpoint import load_checkpoint
from mmaction.registry import MODELS
from mmaction.utils import ConfigType
from .base import BaseHead
from functools import partial
from einops import rearrange


class SwinFeatureProjector(nn.Module):
    def __init__(self) -> None:
        super().__init__()

    def forward(self, feature: Tensor) -> Tensor:
        return rearrange(feature, 'b c d h w -> b d h w c').contiguous()


def build_transfer(type: str,
                   s_channels: int, 
                   t_channels: int, 
                   factor: int = 2):

    def conv1x1(in_channels, out_channels, stride=1):
            return nn.Conv3d(in_channels, out_channels, kernel_size=1, padding=0, stride=stride, bias=False)
    def conv3x3(in_channels, out_channels, stride=1, groups=1):
            return nn.Conv3d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride, bias=False, groups=groups)
    
    transfer = nn.Sequential(conv1x1(s_channels, t_channels//factor),
        nn.BatchNorm3d(t_channels//factor),
        nn.ReLU(inplace=True),
        conv3x3(t_channels//factor, t_channels//factor),
        # depthwise convolution
        # conv3x3(t_n//factor, t_n//factor, groups=t_n//factor),
        nn.BatchNorm3d(t_channels//factor),
        nn.ReLU(inplace=True),
        conv1x1(t_channels//factor, t_channels),
        nn.BatchNorm3d(t_channels),
        nn.ReLU(inplace=True),)
    
    if type == 'swin':
        additional = nn.Sequential(
            conv1x1(t_channels, t_channels),
            SwinFeatureProjector(),
            nn.LayerNorm(t_channels),
            nn.GELU(),)
        transfer = nn.Sequential(*(list(transfer.children()) + list(additional.children())))

    if type == 'swin_proj':
        transfer = nn.Sequential(
            SwinFeatureProjector(),
            nn.utils.parametrizations.orthogonal(nn.Linear(s_channels, t_channels, bias=False)),
            conv1x1(t_channels, t_channels),
            SwinFeatureProjector(),
            nn.LayerNorm(t_channels),
            nn.GELU(),)

        
    return transfer


@MODELS.register_module()
class I3DHeadWithTransfer(BaseHead):

    def __init__(self,
                 num_classes: int,
                 in_channels: int,
                 loss_cls: ConfigType = dict(type='CrossEntropyLoss'),
                 spatial_type: str = 'avg',
                 dropout_ratio: float = 0.5,
                 init_std: float = 0.01,
                 pretrained: Optional[str] = None,
                 freeze_fc: bool = False,
                 transfer_config: Dict = dict(type='cnn', s_channels=1024, t_channels=1024, factor=2),
                 **kwargs) -> None:
        super().__init__(num_classes=num_classes, in_channels=in_channels, loss_cls=loss_cls, **kwargs)

        self.spatial_type = spatial_type
        self.dropout_ratio = dropout_ratio
        self.init_std = init_std
        if self.dropout_ratio != 0:
            self.dropout = nn.Dropout(p=self.dropout_ratio)
        else:
            self.dropout = None
        self.fc_cls = nn.Linear(self.in_channels, self.num_classes)

        if self.spatial_type == 'avg':
            # use `nn.AdaptiveAvgPool3d` to adaptively match the in_channels.
            self.avg_pool = nn.AdaptiveAvgPool3d((1, 1, 1))
        else:
            self.avg_pool = None

        self.pretrained = pretrained
        self.freeze_fc = freeze_fc
        self.transfer_config = transfer_config
        self.transfer = build_transfer(**transfer_config)


    def init_weights(self) -> None:
        # init fc_cls
        if self.pretrained:
            logger = MMLogger.get_current_instance()
            logger.info(f'load model from: {self.pretrained}')
            load_checkpoint(self, self.pretrained, strict=False, logger=logger, revise_keys=[(r'^cls_head\.', '')])
        else:
            normal_init(self.fc_cls, std=self.init_std)
        
                # init transfer
        for m in self.transfer.modules():
            if isinstance(m, nn.Conv3d):
                kaiming_init(m)
            elif isinstance(m, nn.BatchNorm3d):
                constant_init(m, 1)


    def forward(self, x: Tensor, **kwargs) -> Tensor:
        if self.transfer is not None:
            x = self.transfer(x)

        if self.transfer_config["type"] == 'swin' or self.transfer_config["type"]== 'swin_proj':
            x = rearrange(x, 'b d h w c -> b c d h w').contiguous()

        # [N, in_channels, 4, 7, 7]
        if self.avg_pool is not None:
            x = self.avg_pool(x)
        # [N, in_channels, 1, 1, 1]
        if self.dropout is not None:
            x = self.dropout(x)
        # [N, in_channels, 1, 1, 1]
        x = x.view(x.shape[0], -1)
        # [N, in_channels]
        cls_score = self.fc_cls(x)
        # [N, num_classes]
        return cls_score

    def train(self, mode: bool = True) -> None:
        super().train(mode)
        if self.freeze_fc:
            self.fc_cls.eval()
            for param in self.fc_cls.parameters():
                param.requires_grad = False