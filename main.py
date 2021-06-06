from rep.acnet import ACNet
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(50,512,49,49)
    acnet=ACNet(512,512)
    acnet.eval()
    out=acnet(input)
    acnet._switch_to_deploy()
    out2=acnet(input)
    print('difference:')
    print(((out2-out)**2).sum())
    
    