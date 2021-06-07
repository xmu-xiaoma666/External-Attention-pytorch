from attention.EMSA import EMSA
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(50,64,512)
    emsa = EMSA(d_model=512, d_k=512, d_v=512, h=8,H=8,W=8,ratio=2,apply_transform=True)
    output=emsa(input,input,input)
    print(output.shape)
    
    