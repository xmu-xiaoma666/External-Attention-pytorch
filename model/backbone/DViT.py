""" 
Code for DeepViT. The implementation has heavy reference to timm.
"""
import torch
import torch.nn as nn
from functools import partial
import pickle
from torch.nn.parameter import Parameter

from timm.data import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
from timm.models.helpers import load_pretrained
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from timm.models.resnet import resnet26d, resnet50d
from timm.models.registry import register_model

import torch.nn.init as init
import torch.nn.functional as F

from torch.nn import functional as F

import numpy as np

class Mlp(nn.Module):
    def __init__(self, in_features, hidden_features=None, out_features=None, act_layer=nn.GELU, drop=0., expansion_ratio=3):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features

        self.fc1 = nn.Linear(in_features, hidden_features)
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.act = act_layer()
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., expansion_ratio=3):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.expansion = expansion_ratio
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * self.expansion, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, atten=None):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)

        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn
class ReAttention(nn.Module):
    """
    It is observed that similarity along same batch of data is extremely large. 
    Thus can reduce the bs dimension when calculating the attention map.
    """
    def __init__(self, dim, num_heads=8, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0.,expansion_ratio = 3, apply_transform=True, transform_scale=False):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.apply_transform = apply_transform
        
        # NOTE scale factor was wrong in my original version, can set manually to be compat with prev weights
        self.scale = qk_scale or head_dim ** -0.5
        if apply_transform:
            self.reatten_matrix = nn.Conv2d(self.num_heads,self.num_heads, 1, 1)
            self.var_norm = nn.BatchNorm2d(self.num_heads)
            self.qkv = nn.Linear(dim, dim * expansion_ratio, bias=qkv_bias)
            self.reatten_scale = self.scale if transform_scale else 1.0
        else:
            self.qkv = nn.Linear(dim, dim * expansion_ratio, bias=qkv_bias)
        
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
    def forward(self, x, atten=None):
        B, N, C = x.shape
        # x = self.fc(x)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]   # make torchscript happy (cannot use tensor as tuple)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)
        if self.apply_transform:
            attn = self.var_norm(self.reatten_matrix(attn)) * self.reatten_scale
        attn_next = attn
        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x, attn_next
class Block(nn.Module):

    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop=0., attn_drop=0.,
                 drop_path=0., act_layer=nn.GELU, norm_layer=nn.LayerNorm, expansion=3, 
                 group = False, share = False, re_atten=False, bs=False, apply_transform=False,
                 scale_adjustment=1.0, transform_scale=False):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.re_atten = re_atten

        self.adjust_ratio = scale_adjustment
        self.dim = dim
        if  self.re_atten:
            self.attn = ReAttention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
                expansion_ratio = expansion, apply_transform=apply_transform, transform_scale=transform_scale)
        else:
            self.attn = Attention(
                dim, num_heads=num_heads, qkv_bias=qkv_bias, qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, 
                expansion_ratio = expansion)
        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)
    def forward(self, x, atten=None):
        if self.re_atten:
            x_new, atten = self.attn(self.norm1(x * self.adjust_ratio), atten)
            x = x + self.drop_path(x_new/self.adjust_ratio)
            x = x + self.drop_path(self.mlp(self.norm2(x * self.adjust_ratio))) / self.adjust_ratio
            return x, atten
        else:
            x_new, atten = self.attn(self.norm1(x), atten)
            x= x + self.drop_path(x_new)
            x = x + self.drop_path(self.mlp(self.norm2(x)))
            return x, atten

class PatchEmbed_CNN(nn.Module):
    """ 
        Following T2T, we use 3 layers of CNN for comparison with other methods.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768,spp=32):
        super().__init__()

        new_patch_size = to_2tuple(patch_size // 2)

        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.conv1 = nn.Conv2d(in_chans, 64, kernel_size=7, stride=2, padding=3, bias=False)  # 112x112
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1, bias=False)  # 112x112
        self.bn2 = nn.BatchNorm2d(64)

        self.proj = nn.Conv2d(64, embed_dim, kernel_size=new_patch_size, stride=new_patch_size)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)

        x = self.proj(x).flatten(2).transpose(1, 2)  # [B, C, W, H]

        return x
class PatchEmbed(nn.Module):
    """ 
        Same embedding as timm lib.
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        img_size = to_2tuple(img_size)
        patch_size = to_2tuple(patch_size)
        num_patches = (img_size[1] // patch_size[1]) * (img_size[0] // patch_size[0])
        self.img_size = img_size
        self.patch_size = patch_size
        self.num_patches = num_patches

        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        B, C, H, W = x.shape
        # FIXME look at relaxing size constraints
        assert H == self.img_size[0] and W == self.img_size[1], \
            f"Input image size ({H}*{W}) doesn't match model ({self.img_size[0]}*{self.img_size[1]})."
        x = self.proj(x).flatten(2).transpose(1, 2)
        return x


class HybridEmbed(nn.Module):
    """ 
        Same embedding as timm lib.
    """
    def __init__(self, backbone, img_size=224, feature_size=None, in_chans=3, embed_dim=768):
        super().__init__()
        assert isinstance(backbone, nn.Module)
        img_size = to_2tuple(img_size)
        self.img_size = img_size
        self.backbone = backbone
        if feature_size is None:
            with torch.no_grad():
                training = backbone.training
                if training:
                    backbone.eval()
                o = self.backbone(torch.zeros(1, in_chans, img_size[0], img_size[1]))[-1]
                feature_size = o.shape[-2:]
                feature_dim = o.shape[1]
                backbone.train(training)
        else:
            feature_size = to_2tuple(feature_size)
            feature_dim = self.backbone.feature_info.channels()[-1]
        self.num_patches = feature_size[0] * feature_size[1]
        self.proj = nn.Linear(feature_dim, embed_dim)

    def forward(self, x):
        x = self.backbone(x)[-1]
        x = x.flatten(2).transpose(1, 2)
        x = self.proj(x)
        return x

def _cfg(url='', **kwargs):
    return {
        'url': url,
        'num_classes': 1000, 'input_size': (3, 224, 224), 'pool_size': None,
        'crop_pct': .9, 'interpolation': 'bicubic',
        'mean': IMAGENET_DEFAULT_MEAN, 'std': IMAGENET_DEFAULT_STD,
        'first_conv': 'patch_embed.proj', 'classifier': 'head',
        **kwargs
    }


default_cfgs = {
    # patch models
    'Deepvit_base_patch16_224_16B': _cfg(
        url='',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'Deepvit_base_patch16_224_24B': _cfg(
        url='',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'Deepvit_base_patch16_224_32B': _cfg(
        url='',
        mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5),
    ),
    'Deepvit_L_384': _cfg(
        url='',
        input_size=(3, 384, 384), mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5), crop_pct=1.0),
}



class DeepVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=1000, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., hybrid_backbone=None, norm_layer=nn.LayerNorm, group = False, re_atten=True, cos_reg = False,
                 use_cnn_embed=False, apply_transform=None, transform_scale=False, scale_adjustment=1.):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        # use cosine similarity as a regularization term
        self.cos_reg = cos_reg

        if hybrid_backbone is not None:
            self.patch_embed = HybridEmbed(
                hybrid_backbone, img_size=img_size, in_chans=in_chans, embed_dim=embed_dim)
        else:
            if use_cnn_embed:
                self.patch_embed = PatchEmbed_CNN(img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
            else:
                self.patch_embed = PatchEmbed(
                    img_size=img_size, patch_size=patch_size, in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.num_patches

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)
        d = depth if isinstance(depth, int) else len(depth)
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, d)]  # stochastic depth decay rule

        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, share=depth[i], num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer, group = group, 
                re_atten=re_atten, apply_transform=apply_transform[i], transform_scale=transform_scale, scale_adjustment=scale_adjustment)
            for i in range(len(depth))])
        self.norm = norm_layer(embed_dim)

        # Classifier head
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        trunc_normal_(self.pos_embed, std=.02)
        trunc_normal_(self.cls_token, std=.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x):
        if self.cos_reg:
            atten_list = []
        B = x.shape[0]
        x = self.patch_embed(x)

        cls_tokens = self.cls_token.expand(B, -1, -1)  # stole cls_tokens impl from Phil Wang, thanks
        x = torch.cat((cls_tokens, x), dim=1)
        x = x + self.pos_embed
        x = self.pos_drop(x)
        attn = None
        for blk in self.blocks:
            x, attn = blk(x, attn)
            if self.cos_reg:
                atten_list.append(attn)

        x = self.norm(x)
        if self.cos_reg and self.training:
            return x[:, 0], atten_list
        else:
            return x[:, 0]

    def forward(self, x):
        if self.cos_reg and self.training:
            x, atten = self.forward_features(x)
            x = self.head(x)
            return x, atten
        else:
            x = self.forward_features(x)
            x = self.head(x)
            return x


@register_model
def deepvit_patch16_224_re_attn_16b(pretrained=False, **kwargs):
    apply_transform = [False] * 0 + [True] * 16
    model = DeepVisionTransformer(
        patch_size=16, embed_dim=384, depth=[False] * 16, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_base_patch16_224_16B']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

@register_model
def deepvit_patch16_224_re_attn_24b(pretrained=False, **kwargs):
    apply_transform = [False] * 0 + [True] * 24
    model = DeepVisionTransformer(
        patch_size=16, embed_dim=384, depth=[False] * 24, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_base_patch16_224_24B']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model
 
@register_model
def deepvit_patch16_224_re_attn_32b(pretrained=False, **kwargs):
    apply_transform = [False] * 0 + [True] * 32
    model = DeepVisionTransformer(
        patch_size=16, embed_dim=384, depth=[False] * 32, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_base_patch16_224_32B']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model
@register_model
def deepvit_S(pretrained=False, **kwargs):
    apply_transform = [False] * 11 + [True] * 5
    model = DeepVisionTransformer(
        patch_size=16, embed_dim=396, depth=[False] * 16, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),  transform_scale=True, use_cnn_embed = True, scale_adjustment=0.5, **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_base_patch16_224_32B']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model
@register_model
def deepvit_L(pretrained=False, **kwargs):
    apply_transform = [False] * 20 + [True] * 12
    model = DeepVisionTransformer(
        patch_size=16, embed_dim=420, depth=[False] * 32, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), use_cnn_embed = True, scale_adjustment=0.5, **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_base_patch16_224_32B']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

@register_model
def deepvit_L_384(pretrained=False, **kwargs):
    apply_transform = [False] * 20 + [True] * 12
    model = DeepVisionTransformer(
        img_size=384, patch_size=16, embed_dim=420, depth=[False] * 32, apply_transform=apply_transform, num_heads=12, mlp_ratio=3, qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), use_cnn_embed = True, scale_adjustment=0.5, **kwargs)
    # We following the same settings for original ViT
    model.default_cfg = default_cfgs['Deepvit_L_384']
    if pretrained:
        load_pretrained(
            model, num_classes=model.num_classes, in_chans=kwargs.get('in_chans', 3), filter_fn=_conv_filter)
    return model

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = DeepVisionTransformer(
        patch_size=16, embed_dim=384, 
        depth=[False] * 16, 
        apply_transform=[False] * 0 + [True] * 32, 
        num_heads=12, 
        mlp_ratio=3, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
    output=model(input)
    print(output.shape)