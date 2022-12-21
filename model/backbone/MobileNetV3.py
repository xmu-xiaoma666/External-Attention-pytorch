""" MobileNet V3
A PyTorch impl of MobileNet-V3, compatible with TF weights from official impl.
Paper: Searching for MobileNetV3 - https://arxiv.org/abs/1905.02244
Hacked together by / Copyright 2019, Ross Wightman
"""
from functools import partial
from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD, IMAGENET_INCEPTION_MEAN, IMAGENET_INCEPTION_STD
from timm.layers import SelectAdaptivePool2d, Linear, create_conv2d, get_norm_act_layer
from ._builder import build_model_with_cfg, pretrained_cfg_for_features
from ._efficientnet_blocks import SqueezeExcite
from ._efficientnet_builder import EfficientNetBuilder, decode_arch_def, efficientnet_init_weights, \
    round_channels, resolve_bn_args, resolve_act_layer, BN_EPS_TF_DEFAULT
from ._features import FeatureInfo, FeatureHooks
from ._manipulate import checkpoint_seq
from ._registry import register_model

__all__ = ['MobileNetV3', 'MobileNetV3Features']


def _cfg(url='', **kwargs):
    return {
        'url': url, 'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': (7, 7),
        'crop_pct': 0.875, 'interpolation': 'bilinear',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'conv_stem', 'classifier': 'classifier',
        **kwargs
    }


default_cfgs = {
    'mobilenetv3_large_075': _cfg(url=''),
    'mobilenetv3_large_100': _cfg(
        interpolation='bicubic',
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_large_100_ra-f55367f5.pth'),
    'mobilenetv3_large_100_miil': _cfg(
        interpolation='bilinear', mean=(0., 0., 0.), std=(1., 1., 1.),
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mobilenetv3_large_100_1k_miil_78_0-66471c13.pth'),
    'mobilenetv3_large_100_miil_in21k': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-tresnet/mobilenetv3_large_100_in21k_miil-d71cc17b.pth',
        interpolation='bilinear', mean=(0., 0., 0.), std=(1., 1., 1.), num_classes=11221),

    'mobilenetv3_small_050': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_050_lambc-4b7bbe87.pth',
        interpolation='bicubic'),
    'mobilenetv3_small_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_075_lambc-384766db.pth',
        interpolation='bicubic'),
    'mobilenetv3_small_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_small_100_lamb-266a294c.pth',
        interpolation='bicubic'),

    'mobilenetv3_rw': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/mobilenetv3_100-35495452.pth',
        interpolation='bicubic'),

    'tf_mobilenetv3_large_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_075-150ee8b0.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_large_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_100-427764d5.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_large_minimal_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_large_minimal_100-8596ae28.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_075': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_075-da427f52.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_100': _cfg(
        url= 'https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_100-37f49e2b.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),
    'tf_mobilenetv3_small_minimal_100': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/tf_mobilenetv3_small_minimal_100-922a7843.pth',
        mean=IMAGENET_INCEPTION_MEAN, std=IMAGENET_INCEPTION_STD),

    'fbnetv3_b': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_b_224-ead5d2a1.pth',
        test_input_size=(3, 256, 256), crop_pct=0.95),
    'fbnetv3_d': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_d_224-c98bce42.pth',
        test_input_size=(3, 256, 256), crop_pct=0.95),
    'fbnetv3_g': _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/fbnetv3_g_240-0b1df83b.pth',
        input_size=(3, 240, 240), test_input_size=(3, 288, 288), crop_pct=0.95, pool_size=(8, 8)),

    "lcnet_035": _cfg(),
    "lcnet_050": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_050-f447553b.pth',
        interpolation='bicubic',
    ),
    "lcnet_075": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_075-318cad2c.pth',
        interpolation='bicubic',
    ),
    "lcnet_100": _cfg(
        url='https://github.com/rwightman/pytorch-image-models/releases/download/v0.1-weights/lcnet_100-a929038c.pth',
        interpolation='bicubic',
    ),
    "lcnet_150": _cfg(),
}

class MobileNetV3(nn.Module):
    """ MobiletNet-V3
    Based on my EfficientNet implementation and building blocks, this model utilizes the MobileNet-v3 specific
    'efficient head', where global pooling is done before the head convolution without a final batch-norm
    layer before the classifier.
    Paper: `Searching for MobileNetV3` - https://arxiv.org/abs/1905.02244
    Other architectures utilizing MobileNet-V3 efficient head that are supported by this impl include:
      * HardCoRe-NAS - https://arxiv.org/abs/2102.11646 (defn in hardcorenas.py uses this class)
      * FBNet-V3 - https://arxiv.org/abs/2006.02049
      * LCNet - https://arxiv.org/abs/2109.15099
    """
    
    def __init__(
            self, block_args, num_classes=1000, in_chans=3, stem_size=16, fix_stem=False, num_features=1280,
            head_bias=True, pad_type='', act_layer=None, norm_layer=None, se_layer=None, se_from_exp=True,
            round_chs_fn=round_channels, drop_rate=0., drop_path_rate=0., global_pool='avg'):
        super(MobileNetV3, self).__init__()
        act_layer = act_layer or nn.ReLU
        norm_layer = norm_layer or nn.BatchNorm2d
        norm_act_layer = get_norm_act_layer(norm_layer, act_layer)
        se_layer = se_layer or SqueezeExcite
        self.num_classes = num_classes
        self.num_features = num_features
        self.drop_rate = drop_rate
        self.grad_checkpointing = False

        # Stem
        if not fix_stem:
            stem_size = round_chs_fn(stem_size)
        self.conv_stem = create_conv2d(in_chans, stem_size, 3, stride=2, padding=pad_type)
        self.bn1 = norm_act_layer(stem_size, inplace=True)

        # Middle stages (IR/ER/DS Blocks)
        builder = EfficientNetBuilder(
            output_stride=32, pad_type=pad_type, round_chs_fn=round_chs_fn, se_from_exp=se_from_exp,
            act_layer=act_layer, norm_layer=norm_layer, se_layer=se_layer, drop_path_rate=drop_path_rate)
        self.blocks = nn.Sequential(*builder(stem_size, block_args))
        self.feature_info = builder.features
        head_chs = builder.in_chs

        # Head + Pooling
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        num_pooled_chs = head_chs * self.global_pool.feat_mult()
        self.conv_head = create_conv2d(num_pooled_chs, self.num_features, 1, padding=pad_type, bias=head_bias)
        self.act2 = act_layer(inplace=True)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()

        efficientnet_init_weights(self)
        
    def as_sequential(self):
        layers = [self.conv_stem, self.bn1]
        layers.extend(self.blocks)
        layers.extend([self.global_pool, self.conv_head, self.act2])
        layers.extend([nn.Flatten(), nn.Dropout(self.drop_rate), self.classifier])
        return nn.Sequential(*layers)
    
    @torch.jit.ignore
    def group_matcher(self, coarse=False):
        return dict(
            stem=r'^conv_stem|bn1',
            blocks=r'^blocks\.(\d+)' if coarse else r'^blocks\.(\d+)\.(\d+)'
        )
    
    @torch.jit.ignore
    def set_grad_checkpointing(self, enable=True):
        self.grad_checkpointing = enable
        
    @torch.jit.ignore
    def get_classifier(self):
        return self.classifier
    
    def reset_classifier(self, num_classes, global_pool='avg'):
        self.num_classes = num_classes
        # cannot meaningfully change pooling of efficient head after creation
        self.global_pool = SelectAdaptivePool2d(pool_type=global_pool)
        self.flatten = nn.Flatten(1) if global_pool else nn.Identity()  # don't flatten if pooling disabled
        self.classifier = Linear(self.num_features, num_classes) if num_classes > 0 else nn.Identity()
        
    def forward_features(self, x):
        x = self.conv_stem(x)
        x = self.bn1(x)
        if self.grad_checkpointing and not torch.jit.is_scripting():
            x = checkpoint_seq(self.blocks, x, flatten=True)
        else:
            x = self.blocks(x)
        return x
    
    def forward_head(self, x, pre_logits: bool = False):
        x = self.global_pool(x)
        x = self.conv_head(x)
        x = self.act2(x)
        if pre_logits:
            return x.flatten(1)
        else:
            x = self.flatten(x)
            if self.drop_rate > 0.:
                x = F.dropout(x, p=self.drop_rate, training=self.training)
            return self.classifier(x)
        
    def forward(self, x):
        x = self.forward_features(x)
        x = self.forward_head(x)
        return x
    
class MobileNetV3Features(nn.Module):
    """ MobileNetV3 Feature Extractor
    A work-in-progress feature extraction module for MobileNet-V3 to use as a backbone for segmentation
    and object detection models.
    """
