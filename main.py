from conv.Involution import Involution
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,4,64,64)
involution=Involution(kernel_size=3,in_channel=4,stride=2)
out=involution(input)
print(out.shape)