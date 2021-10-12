from model.conv.CondConv import *
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(2,32,64,64)
    m=CondConv(in_planes=32,out_planes=64,kernel_size=3,stride=1,padding=1,bias=False)
    out=m(input)
    print(out.shape)