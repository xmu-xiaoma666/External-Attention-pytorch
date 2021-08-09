# from attention.S2Attention import S2Attention
# import torch
# from torch import nn
# from torch.nn import functional as F

# input=torch.randn(50,512,7,7)
# s2att = S2Attention(channels=512)
# output=s2att(input)
# print(output.shape)


from backbone_cnn.resnext import ResNeXt50,ResNeXt101,ResNeXt152
import torch

if __name__ == '__main__':
    input=torch.randn(50,3,224,224)
    resnext50=ResNeXt50(1000)
    # resnext101=ResNeXt101(1000)
    # resnext152=ResNeXt152(1000)
    out=resnext50(input)
    print(out.shape)

