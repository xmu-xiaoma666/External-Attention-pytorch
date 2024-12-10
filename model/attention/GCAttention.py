import torch
from torch import nn


class GCModule(nn.Module):
    def __init__(self, channel, reduction=16):
        super().__init__()
        self.conv = nn.Conv2d(channel, 1, kernel_size=1)
        self.softmax = nn.Softmax(dim=2)
        self.transform = nn.Sequential(
            nn.Conv2d(channel, channel // reduction, kernel_size=1),
            nn.LayerNorm([channel // reduction, 1, 1]),
            nn.ReLU(inplace=True),
            nn.Conv2d(channel // reduction, channel, kernel_size=1)
        )
    
    def context_modeling(self, x):
        b, c, h, w = x.shape
        input_x = x
        input_x = input_x.reshape(b, c, h * w)
        context = self.conv(x)
        context = context.reshape(b, 1, h * w).transpose(1, 2)
        out = torch.matmul(input_x, context)
        out = out.reshape(b, c, 1, 1)
        return out
    
    def forward(self, x):
        context = self.context_modeling(x)
        y = self.transform(context)
        return x + y
    
if __name__ == "__main__":
    input = torch.randn(16, 64, 32, 32)
    gc_layer = GCModule(64)
    output = gc_layer(input)
    print(output.shape)