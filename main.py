from fightingcv.backbone.MobileViT import *
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)

    ### mobilevit_xxs
    mvit_xxs=mobilevit_xxs()
    out=mvit_xxs(input)
    print(out.shape)

    ### mobilevit_xs
    mvit_xs=mobilevit_xs()
    out=mvit_xs(input)
    print(out.shape)


    ### mobilevit_s
    mvit_s=mobilevit_s()
    out=mvit_s(input)
    print(out.shape)

    

    





