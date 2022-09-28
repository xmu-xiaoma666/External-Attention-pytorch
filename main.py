from model.attention.MobileViTv2Attention import *
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(50,49,512)
    sa = MobileViTv2Attention(d_model=512)
    output=sa(input)
    print(output.shape)
 