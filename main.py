from attention.S2Attention import S2Attention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
s2att = S2Attention(channels=512)
output=s2att(input)
print(output.shape)
