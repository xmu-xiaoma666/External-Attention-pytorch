from fightingcv.mlp.sMLP_block import sMLPBlock
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(50,3,224,224)
    smlp=sMLPBlock(h=224,w=224)
    out=smlp(input)
    print(out.shape)





