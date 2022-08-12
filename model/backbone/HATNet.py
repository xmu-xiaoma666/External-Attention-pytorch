from pyexpat import model
import torch
import torch.nn as nn
import torch.nn.functional as F

from timm.models.layers import DropPath, trunc_normal_
from timm.models.registry import register_model
from timm.models.vision_transformer import _cfg


class InvertedResidual(nn.Module):
    def __init__(self, in_dim, hidden_dim=None, out_dim=None, kernel_size=3,
                 drop=0., act_layer=nn.SiLU):
        super().__init__()
        hidden_dim = hidden_dim or in_dim
        out_dim = out_dim or in_dim
        pad = (kernel_size - 1) // 2
        self.conv1 = nn.Sequential(
            nn.GroupNorm(1, in_dim, eps=1e-6),
            nn.Conv2d(in_dim, hidden_dim, 1, bias=False),
            act_layer(inplace=True)
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(hidden_dim, hidden_dim, kernel_size, padding=pad, groups=hidden_dim, bias=False),
            act_layer(inplace=True)
        )
        self.conv3 = nn.Sequential(
            nn.Conv2d(hidden_dim, out_dim, 1, bias=False),
            nn.GroupNorm(1, out_dim, eps=1e-6)
        )
        self.drop = nn.Dropout2d(drop, inplace=True)

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)
        x = self.drop(x)
        x = self.conv3(x)
        x = self.drop(x)

        return x


class Attention(nn.Module):
    def __init__(self, dim, head_dim, grid_size=1, ds_ratio=1, drop=0.):
        super().__init__()
        assert dim % head_dim == 0
        self.num_heads = dim // head_dim
        self.head_dim = head_dim
        self.scale = self.head_dim ** -0.5
        self.grid_size = grid_size

        self.norm = nn.GroupNorm(1, dim, eps=1e-6)
        self.qkv = nn.Conv2d(dim, dim * 3, 1)
        self.proj = nn.Conv2d(dim, dim, 1)
        self.proj_norm = nn.GroupNorm(1, dim, eps=1e-6)
        self.drop = nn.Dropout2d(drop, inplace=True)

        if grid_size > 1:
            self.grid_norm = nn.GroupNorm(1, dim, eps=1e-6)
            self.avg_pool = nn.AvgPool2d(ds_ratio, stride=ds_ratio)
            self.ds_norm = nn.GroupNorm(1, dim, eps=1e-6)
            self.q = nn.Conv2d(dim, dim, 1)
            self.kv = nn.Conv2d(dim, dim * 2, 1)

    def forward(self, x):
        B, C, H, W = x.shape
        qkv = self.qkv(self.norm(x))

        if self.grid_size > 1:
            grid_h, grid_w = H // self.grid_size, W // self.grid_size
            qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, grid_h,
                self.grid_size, grid_w, self.grid_size)
            qkv = qkv.permute(1, 0, 2, 4, 6, 5, 7, 3)
            qkv = qkv.reshape(3, -1, self.grid_size * self.grid_size, self.head_dim)
            q, k, v = qkv[0], qkv[1], qkv[2]

            attn = (q * self.scale) @ k.transpose(-2, -1)
            attn = attn.softmax(dim=-1)
            grid_x = (attn @ v).reshape(B, self.num_heads, grid_h, grid_w,
                self.grid_size, self.grid_size, self.head_dim)
            grid_x = grid_x.permute(0, 1, 6, 2, 4, 3, 5).reshape(B, C, H, W)
            grid_x = self.grid_norm(x + grid_x)

            q = self.q(grid_x).reshape(B, self.num_heads, self.head_dim, -1)
            q = q.transpose(-2, -1)
            kv = self.kv(self.ds_norm(self.avg_pool(grid_x)))
            kv = kv.reshape(B, 2, self.num_heads, self.head_dim, -1)
            kv = kv.permute(1, 0, 2, 4, 3)
            k, v = kv[0], kv[1]
        else:
            qkv = qkv.reshape(B, 3, self.num_heads, self.head_dim, -1)
            qkv = qkv.permute(1, 0, 2, 4, 3)
            q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q * self.scale) @ k.transpose(-2, -1)
        attn = attn.softmax(dim=-1)
        global_x = (attn @ v).transpose(-2, -1).reshape(B, C, H, W)
        if self.grid_size > 1:
            global_x = global_x + grid_x
        x = self.drop(self.proj(global_x))

        return x


class Block(nn.Module):
    def __init__(self, dim, head_dim, grid_size=1, ds_ratio=1, expansion=4,
                 drop=0., drop_path=0., kernel_size=3, act_layer=nn.SiLU):
        super().__init__()
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.attn = Attention(dim, head_dim, grid_size=grid_size, ds_ratio=ds_ratio, drop=drop)
        self.conv = InvertedResidual(dim, hidden_dim=dim * expansion, out_dim=dim,
            kernel_size=kernel_size, drop=drop, act_layer=act_layer)

    def forward(self, x):
        x = x + self.drop_path(self.attn(x))
        x = x + self.drop_path(self.conv(x))
        return x


class Downsample(nn.Module):
    def __init__(self, in_dim, out_dim, kernel_size=3):
        super().__init__()
        self.conv = nn.Conv2d(in_dim, out_dim, kernel_size, padding=1, stride=2)
        self.norm = nn.GroupNorm(1, out_dim, eps=1e-6)

    def forward(self, x):
        x = self.norm(self.conv(x))
        return x


class HATNet(nn.Module):
    def __init__(self, img_size=224, in_chans=3, num_classes=1000, dims=[64, 128, 256, 512],
                 head_dim=64, expansions=[4, 4, 6, 6], grid_sizes=[1, 1, 1, 1],
                 ds_ratios=[8, 4, 2, 1], depths=[3, 4, 8, 3], drop_rate=0.,
                 drop_path_rate=0., act_layer=nn.SiLU, kernel_sizes=[3, 3, 3, 3]):
        super().__init__()
        self.depths = depths
        self.patch_embed = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1, stride=2),
            nn.GroupNorm(1, 16, eps=1e-6),
            act_layer(inplace=True),
            nn.Conv2d(16, dims[0], 3, padding=1, stride=2),
        )

        self.blocks = []
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        for stage in range(len(dims)):
            self.blocks.append(nn.ModuleList([Block(
                dims[stage], head_dim, grid_size=grid_sizes[stage], ds_ratio=ds_ratios[stage],
                expansion=expansions[stage], drop=drop_rate, drop_path=dpr[sum(depths[:stage]) + i],
                kernel_size=kernel_sizes[stage], act_layer=act_layer)
                for i in range(depths[stage])]))
        self.blocks = nn.ModuleList(self.blocks)

        self.ds2 = Downsample(dims[0], dims[1])
        self.ds3 = Downsample(dims[1], dims[2])
        self.ds4 = Downsample(dims[2], dims[3])
        self.classifier = nn.Sequential(
            nn.Dropout(0.2, inplace=True),
            nn.Linear(dims[-1], num_classes),
        )

        # init weights
        self.apply(self._init_weights)

    def reset_drop_path(self, drop_path_rate):
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, sum(self.depths))]
        cur = 0
        for stage in range(len(self.blocks)):
            for idx in range(self.depths[stage]):
                self.blocks[stage][idx].drop_path.drop_prob = dpr[cur + idx]
            cur += self.depths[stage]

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, (nn.LayerNorm, nn.BatchNorm2d, nn.GroupNorm)):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward(self, x):
        x = self.patch_embed(x)
        for block in self.blocks[0]:
            x = block(x)
        x = self.ds2(x)
        for block in self.blocks[1]:
            x = block(x)
        x = self.ds3(x)
        for block in self.blocks[2]:
            x = block(x)
        x = self.ds4(x)
        for block in self.blocks[3]:
            x = block(x)
        x = F.adaptive_avg_pool2d(x, (1, 1)).flatten(1)
        x = self.classifier(x)

        return x

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    hat = HATNet(dims=[48, 96, 240, 384], head_dim=48, expansions=[8, 8, 4, 4],
        grid_sizes=[8, 7, 7, 1], ds_ratios=[8, 4, 2, 1], depths=[2, 2, 6, 3])
    output=hat(input)
    print(output.shape)