from torch import conv2d, nn
import torch
from torch.nn import functional as F

def transI_conv_bn(conv, bn):

    std = (bn.running_var + bn.eps).sqrt()
    gamma=bn.weight

    weight=conv.weight*((gamma/std).reshape(-1, 1, 1, 1))
    if(conv.bias is not None):
        bias=gamma/std*conv.bias-gamma/std*bn.running_mean+bn.bias
    else:
        bias=bn.bias-gamma/std*bn.running_mean
    return weight,bias

def transII_conv_branch(conv1, conv2):
    weight=conv1.weight.data+conv2.weight.data
    bias=conv1.bias.data+conv2.bias.data
    return weight,bias


def transIII_conv_sequential(conv1, conv2):
    weight=F.conv2d(conv2.weight.data,conv1.weight.data.permute(1,0,2,3))
    # bias=((conv2.weight.data*(conv1.bias.data.reshape(1,-1,1,1))).sum(-1).sum(-1).sum(-1))+conv2.bias.data
    return weight#,bias

def transIV_conv_concat(conv1, conv2):
    print(conv1.bias.data.shape)
    print(conv2.bias.data.shape)
    weight=torch.cat([conv1.weight.data,conv2.weight.data],0)
    bias=torch.cat([conv1.bias.data,conv2.bias.data],0)
    return weight,bias

def transV_avg(channel,kernel):
    conv=nn.Conv2d(channel,channel,kernel,bias=False)
    conv.weight.data[:]=0
    for i in range(channel):
        conv.weight.data[i,i,:,:]=1/(kernel*kernel)
    return conv

def transVI_conv_scale(conv1, conv2, conv3):
    weight=F.pad(conv1.weight.data,(1,1,1,1))+F.pad(conv2.weight.data,(0,0,1,1))+F.pad(conv3.weight.data,(1,1,0,0))
    bias=conv1.bias.data+conv2.bias.data+conv3.bias.data
    return weight,bias

if __name__ == '__main__':
    input=torch.randn(1,64,7,7)

    #conv+conv
    conv1x1=nn.Conv2d(64,64,1)
    conv1x3=nn.Conv2d(64,64,(1,3),padding=(0,1))
    conv3x1=nn.Conv2d(64,64,(3,1),padding=(1,0))
    out1=conv1x1(input)+conv1x3(input)+conv3x1(input)

    #conv_fuse
    conv_fuse=nn.Conv2d(64,64,3,padding=1)
    conv_fuse.weight.data,conv_fuse.bias.data=transVI_conv_scale(conv1x1,conv1x3,conv3x1)
    out2=conv_fuse(input)

    print("difference:",((out2-out1)**2).sum().item())
    