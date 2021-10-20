from model.attention.ParNetAttention import *
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    pna = ParNetAttention(channel=512)
    output=pna(input)
    print(output.shape) #50,512,7,7