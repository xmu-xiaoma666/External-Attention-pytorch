import torch.nn as nn
from torch.nn.modules.activation import GELU
import torch
from torch.nn.modules.pooling import AdaptiveAvgPool2d

class Residual(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn=fn
    def forward(self,x):
        return x+self.fn(x)

def ConvMixer(dim,depth,kernel_size=9,patch_size=7,num_classes=1000):
    return nn.Sequential(
        nn.Conv2d(3,dim,kernel_size=patch_size,stride=patch_size),
        nn.GELU(),
        nn.BatchNorm2d(dim),
        *[nn.Sequential(
            Residual(nn.Sequential(
                nn.Conv2d(dim,dim,kernel_size=kernel_size,groups=dim,padding=kernel_size//2),
                nn.GELU(),
                nn.BatchNorm2d(dim)
            )),
            nn.Conv2d(dim,dim,kernel_size=1),
            nn.GELU(),
            nn.BatchNorm2d(dim)
        ) for _ in range(depth)],
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Linear(dim,num_classes)
    )

if __name__ == '__main__':
    x=torch.randn(1,3,224,224)
    convmixer=ConvMixer(dim=512,depth=12)
    out=convmixer(x)
    print(out.shape)  #[1, 1000]

    
