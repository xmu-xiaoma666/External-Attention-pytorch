# from attention.S2Attention import S2Attention
# import torch
# from torch import nn
# from torch.nn import functional as F

# input=torch.randn(50,512,7,7)
# s2att = S2Attention(channels=512)
# output=s2att(input)
# print(output.shape)


from backbone_cnn.resnet import ResNet50,ResNet101,ResNet152
import torch
if __name__ == '__main__':
    input=torch.randn(50,3,224,224)
    resnet50=ResNet50(1000)
    # resnet101=ResNet101(1000)
    # resnet152=ResNet152(1000)
    out=resnet50(input)
    print(out.shape)