import torch
from torch import nn

class DepthwiseSeparableConvolution(nn.Module):
    def __init__(self,in_ch,out_ch,kernel_size=3,stride=1,padding=1):
        super().__init__()
        self.depthwise_conv=nn.Conv2d(
            in_channels=in_ch,
            out_channels=in_ch,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            groups=in_ch
        )
        self.pointwise_conv=nn.Conv2d(
            in_channels=in_ch,
            out_channels=out_ch,
            kernel_size=1,
            stride=1,
            padding=0,
            groups=1
        )
        
    def forward(self, x):
        out=self.depthwise_conv(x)
        out=self.pointwise_conv(out)
        return out

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    dsconv=DepthwiseSeparableConvolution(3,64)
    out=dsconv(input)
    print(out.shape)
    