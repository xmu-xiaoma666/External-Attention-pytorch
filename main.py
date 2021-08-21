# from attention.gfnet import GFNet
# import torch
# from torch import nn
# from torch.nn import functional as F

# x = torch.randn(1, 3, 224, 224)
# gfnet = GFNet(embed_dim=384, img_size=224, patch_size=16, num_classes=1000)
# out = gfnet(x)
# print(out.shape)


# from backbone_cnn.resnext import ResNeXt50,ResNeXt101,ResNeXt152
# import torch

# if __name__ == '__main__':
#     input=torch.randn(50,3,224,224)
#     resnext50=ResNeXt50(1000)
#     # resnext101=ResNeXt101(1000)
#     # resnext152=ResNeXt152(1000)
#     out=resnext50(input)
#     print(out.shape)


