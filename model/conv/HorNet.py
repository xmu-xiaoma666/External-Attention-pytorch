from functools import partial
import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models.layers import trunc_normal_, DropPath
from timm.models.registry import register_model
import torch.fft

def get_dwconv(dim, kernel, bias):
    return nn.Conv2d(dim, dim, kernel_size=kernel, padding=(kernel-1)//2 ,bias=bias, groups=dim)

class GlobalLocalFilter(nn.Module):
    # https://arxiv.org/abs/2207.14284
    def __init__(self, dim, h=14, w=8):
        super().__init__()
        self.dw = nn.Conv2d(dim // 2, dim // 2, kernel_size=3, padding=1, bias=False, groups=dim // 2)
        self.complex_weight = nn.Parameter(torch.randn(dim // 2, h, w, 2, dtype=torch.float32) * 0.02)
        trunc_normal_(self.complex_weight, std=.02)
        self.pre_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.post_norm = LayerNorm(dim, eps=1e-6, data_format='channels_first')

    def forward(self, x):
        x = self.pre_norm(x)
        x1, x2 = torch.chunk(x, 2, dim=1)
        x1 = self.dw(x1)

        x2 = x2.to(torch.float32)
        B, C, a, b = x2.shape
        x2 = torch.fft.rfft2(x2, dim=(2, 3), norm='ortho')

        weight = self.complex_weight
        if not weight.shape[1:3] == x2.shape[2:4]:
            weight = F.interpolate(weight.permute(3,0,1,2), size=x2.shape[2:4], mode='bilinear', align_corners=True).permute(1,2,3,0)

        weight = torch.view_as_complex(weight.contiguous())

        x2 = x2 * weight
        x2 = torch.fft.irfft2(x2, s=(a, b), dim=(2, 3), norm='ortho')

        x = torch.cat([x1.unsqueeze(2), x2.unsqueeze(2)], dim=2).reshape(B, 2 * C, a, b)
        x = self.post_norm(x)
        return x


class gnconv(nn.Module):
    def __init__(self, dim, order=5, gflayer=None, h=14, w=8, s=1.0):
        super().__init__()
        self.order = order
        self.dims = [dim // 2 ** i for i in range(order)]
        self.dims.reverse()
        self.proj_in = nn.Conv2d(dim, 2*dim, 1)

        if gflayer is None:
            self.dwconv = get_dwconv(sum(self.dims), 7, True)
        else:
            self.dwconv = gflayer(sum(self.dims), h=h, w=w)
        
        self.proj_out = nn.Conv2d(dim, dim, 1)

        self.pws = nn.ModuleList(
            [nn.Conv2d(self.dims[i], self.dims[i+1], 1) for i in range(order-1)]
        )

        self.scale = s

    def forward(self, x, mask=None, dummy=False):
        B, C, H, W = x.shape

        fused_x = self.proj_in(x)
        pwa, abc = torch.split(fused_x, (self.dims[0], sum(self.dims)), dim=1)

        dw_abc = self.dwconv(abc) * self.scale

        dw_list = torch.split(dw_abc, self.dims, dim=1)
        x = pwa * dw_list[0]

        for i in range(self.order -1):
            x = self.pws[i](x) * dw_list[i+1]

        x = self.proj_out(x)

        return x

class Block(nn.Module):
    r""" HorNet block
    """
    def __init__(self, dim, drop_path=0., layer_scale_init_value=1e-6, gnconv=gnconv):
        super().__init__()

        self.norm1 = LayerNorm(dim, eps=1e-6, data_format='channels_first')
        self.gnconv = gnconv(dim) # depthwise conv
        self.norm2 = LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim) # pointwise/1x1 convs, implemented with linear layers
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)

        self.gamma1 = nn.Parameter(layer_scale_init_value * torch.ones(dim), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None

        self.gamma2 = nn.Parameter(layer_scale_init_value * torch.ones((dim)), 
                                    requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()

    def forward(self, x):
        B, C, H, W  = x.shape
        if self.gamma1 is not None:
            gamma1 = self.gamma1.view(C, 1, 1)
        else:
            gamma1 = 1
        x = x + self.drop_path(gamma1 * self.gnconv(self.norm1(x)))

        input = x
        x = x.permute(0, 2, 3, 1) # (N, C, H, W) -> (N, H, W, C)
        x = self.norm2(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        if self.gamma2 is not None:
            x = self.gamma2 * x
        x = x.permute(0, 3, 1, 2) # (N, H, W, C) -> (N, C, H, W)

        x = input + self.drop_path(x)
        return x


class HorNet(nn.Module):
    def __init__(self, in_chans=3, num_classes=1000, 
                 depths=[3, 3, 9, 3], base_dim=96, drop_path_rate=0.,
                 layer_scale_init_value=1e-6, head_init_scale=1.,
                 gnconv=gnconv, block=Block, uniform_init=False, **kwargs
                 ):
        super().__init__()
        dims = [base_dim, base_dim*2, base_dim*4, base_dim*8]

        self.downsample_layers = nn.ModuleList() # stem and 3 intermediate downsampling conv layers
        stem = nn.Sequential(
            nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
            LayerNorm(dims[0], eps=1e-6, data_format="channels_first")
        )
        self.downsample_layers.append(stem)
        for i in range(3):
            downsample_layer = nn.Sequential(
                    LayerNorm(dims[i], eps=1e-6, data_format="channels_first"),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
            )
            self.downsample_layers.append(downsample_layer)

        self.stages = nn.ModuleList() # 4 feature resolution stages, each consisting of multiple residual blocks
        dp_rates=[x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))] 


        if not isinstance(gnconv, list):
            gnconv = [gnconv, gnconv, gnconv, gnconv]
        else:
            gnconv = gnconv
            assert len(gnconv) == 4

        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[block(dim=dims[i], drop_path=dp_rates[cur + j], 
                layer_scale_init_value=layer_scale_init_value, gnconv=gnconv[i]) for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6) # final norm layer
        self.head = nn.Linear(dims[-1], num_classes)

        self.uniform_init = uniform_init

        self.apply(self._init_weights)
        self.head.weight.data.mul_(head_init_scale)
        self.head.bias.data.mul_(head_init_scale)

    def _init_weights(self, m):
        if not self.uniform_init:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                trunc_normal_(m.weight, std=.02)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)
        else:
            if isinstance(m, (nn.Conv2d, nn.Linear)):
                nn.init.xavier_uniform_(m.weight)
                if hasattr(m, 'bias') and m.bias is not None:
                    nn.init.constant_(m.bias, 0)            

    def forward_features(self, x):
        for i in range(4):
            x = self.downsample_layers[i](x)
            for j, blk in enumerate(self.stages[i]):
                    x = blk(x)
        return self.norm(x.mean([-2, -1])) # global average pooling, (N, C, H, W) -> (N, C)

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

class LayerNorm(nn.Module):
    r""" LayerNorm that supports two data formats: channels_last (default) or channels_first. 
    The ordering of the dimensions in the inputs. channels_last corresponds to inputs with 
    shape (batch_size, height, width, channels) while channels_first corresponds to inputs 
    with shape (batch_size, channels, height, width).
    """
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.bias = nn.Parameter(torch.zeros(normalized_shape))
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise NotImplementedError 
        self.normalized_shape = (normalized_shape, )
    
    def forward(self, x):
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            u = x.mean(1, keepdim=True)
            s = (x - u).pow(2).mean(1, keepdim=True)
            x = (x - u) / torch.sqrt(s + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x

@register_model
def hornet_tiny_7x7(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=64, block=Block,
    gnconv=[
        partial(gnconv, order=2, s=s),
        partial(gnconv, order=3, s=s),
        partial(gnconv, order=4, s=s),
        partial(gnconv, order=5, s=s),
    ],
    **kwargs
    )
    return model

@register_model
def hornet_tiny_gf(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=64, block=Block,
    gnconv=[
        partial(gnconv, order=2, s=s),
        partial(gnconv, order=3, s=s),
        partial(gnconv, order=4, s=s, h=14, w=8, gflayer=GlobalLocalFilter),
        partial(gnconv, order=5, s=s, h=7, w=4, gflayer=GlobalLocalFilter),
    ],
    **kwargs
    )
    return model

@register_model
def hornet_small_7x7(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=96, block=Block,
    gnconv=[
        partial(gnconv, order=2, s=s),
        partial(gnconv, order=3, s=s),
        partial(gnconv, order=4, s=s),
        partial(gnconv, order=5, s=s),
    ],
    **kwargs
    )
    return model

@register_model
def hornet_small_gf(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=96, block=Block,
    gnconv=[
        partial(gnconv, order=2, s=s),
        partial(gnconv, order=3, s=s),
        partial(gnconv, order=4, s=s, h=14, w=8, gflayer=GlobalLocalFilter),
        partial(gnconv, order=5, s=s, h=7, w=4, gflayer=GlobalLocalFilter),
    ],
    **kwargs
    )
    return model

@register_model
def hornet_base_7x7(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=128, block=Block,
    gnconv=[
        partial(gnconv, order=2, s=s),
        partial(gnconv, order=3, s=s),
        partial(gnconv, order=4, s=s),
        partial(gnconv, order=5, s=s),
    ],
    **kwargs
    )
    return model

@register_model
def hornet_base_gf(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=128, block=Block,
    gnconv=[
        partial(gnconv, order=2, s=s),
        partial(gnconv, order=3, s=s),
        partial(gnconv, order=4, s=s, h=14, w=8, gflayer=GlobalLocalFilter),
        partial(gnconv, order=5, s=s, h=7, w=4, gflayer=GlobalLocalFilter),
    ],
    **kwargs
    )
    return model

@register_model
def hornet_base_gf_img384(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=128, block=Block,
    gnconv=[
        partial(gnconv, order=2, s=s),
        partial(gnconv, order=3, s=s),
        partial(gnconv, order=4, s=s, h=24, w=13, gflayer=GlobalLocalFilter),
        partial(gnconv, order=5, s=s, h=12, w=7, gflayer=GlobalLocalFilter),
    ],
    **kwargs
    )
    return model

@register_model
def hornet_large_7x7(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=192, block=Block,
    gnconv=[
        partial(gnconv, order=2, s=s),
        partial(gnconv, order=3, s=s),
        partial(gnconv, order=4, s=s),
        partial(gnconv, order=5, s=s),
    ],
    **kwargs
    )
    return model

@register_model
def hornet_large_gf(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=192, block=Block,
    gnconv=[
        partial(gnconv, order=2, s=s),
        partial(gnconv, order=3, s=s),
        partial(gnconv, order=4, s=s, h=14, w=8, gflayer=GlobalLocalFilter),
        partial(gnconv, order=5, s=s, h=7, w=4, gflayer=GlobalLocalFilter),
    ],
    **kwargs
    )
    return model

@register_model
def hornet_large_gf_img384(pretrained=False,in_22k=False, **kwargs):
    s = 1.0/3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=192, block=Block,
    gnconv=[
        partial(gnconv, order=2, s=s),
        partial(gnconv, order=3, s=s),
        partial(gnconv, order=4, s=s, h=24, w=13, gflayer=GlobalLocalFilter),
        partial(gnconv, order=5, s=s, h=12, w=7, gflayer=GlobalLocalFilter),
    ],
    **kwargs
    )
    return model

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    s = 1.0/3.0
    model = HorNet(depths=[2, 3, 18, 2], base_dim=64, block=Block,
    gnconv=[
        partial(gnconv, order=2, s=s),
        partial(gnconv, order=3, s=s),
        partial(gnconv, order=4, s=s),
        partial(gnconv, order=5, s=s),
    ],
    )
    outputs = model(input)
    print(outputs.shape)