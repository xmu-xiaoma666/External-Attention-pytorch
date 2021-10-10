from fightingcv.attention.MobileViTAttention import MobileViTAttention
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    m=MobileViTAttention()
    input=torch.randn(1,3,49,49)
    output=m(input)
    print(output.shape)
    

    





