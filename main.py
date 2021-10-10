from model.backbone.ConvMixer import *
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    x=torch.randn(1,3,224,224)
    convmixer=ConvMixer(dim=512,depth=12)
    out=convmixer(x)
    print(out.shape)  #[1, 1000]
