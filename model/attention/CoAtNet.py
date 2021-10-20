from torch import nn, sqrt
import torch
import sys
from math import sqrt
sys.path.append('.')
from model.conv.MBConv import MBConvBlock
from model.attention.SelfAttention import ScaledDotProductAttention

class CoAtNet(nn.Module):
    def __init__(self,in_ch,image_size,out_chs=[64,96,192,384,768]):
        super().__init__()
        self.out_chs=out_chs
        self.maxpool2d=nn.MaxPool2d(kernel_size=2,stride=2)
        self.maxpool1d = nn.MaxPool1d(kernel_size=2, stride=2)

        self.s0=nn.Sequential(
            nn.Conv2d(in_ch,in_ch,kernel_size=3,padding=1),
            nn.ReLU(),
            nn.Conv2d(in_ch,in_ch,kernel_size=3,padding=1)
        )
        self.mlp0=nn.Sequential(
            nn.Conv2d(in_ch,out_chs[0],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[0],out_chs[0],kernel_size=1)
        )
        
        self.s1=MBConvBlock(ksize=3,input_filters=out_chs[0],output_filters=out_chs[0],image_size=image_size//2)
        self.mlp1=nn.Sequential(
            nn.Conv2d(out_chs[0],out_chs[1],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[1],out_chs[1],kernel_size=1)
        )

        self.s2=MBConvBlock(ksize=3,input_filters=out_chs[1],output_filters=out_chs[1],image_size=image_size//4)
        self.mlp2=nn.Sequential(
            nn.Conv2d(out_chs[1],out_chs[2],kernel_size=1),
            nn.ReLU(),
            nn.Conv2d(out_chs[2],out_chs[2],kernel_size=1)
        )

        self.s3=ScaledDotProductAttention(out_chs[2],out_chs[2]//8,out_chs[2]//8,8)
        self.mlp3=nn.Sequential(
            nn.Linear(out_chs[2],out_chs[3]),
            nn.ReLU(),
            nn.Linear(out_chs[3],out_chs[3])
        )

        self.s4=ScaledDotProductAttention(out_chs[3],out_chs[3]//8,out_chs[3]//8,8)
        self.mlp4=nn.Sequential(
            nn.Linear(out_chs[3],out_chs[4]),
            nn.ReLU(),
            nn.Linear(out_chs[4],out_chs[4])
        )


    def forward(self, x) :
        B,C,H,W=x.shape
        #stage0
        y=self.mlp0(self.s0(x))
        y=self.maxpool2d(y)
        #stage1
        y=self.mlp1(self.s1(y))
        y=self.maxpool2d(y)
        #stage2
        y=self.mlp2(self.s2(y))
        y=self.maxpool2d(y)
        #stage3
        y=y.reshape(B,self.out_chs[2],-1).permute(0,2,1) #B,N,C
        y=self.mlp3(self.s3(y,y,y))
        y=self.maxpool1d(y.permute(0,2,1)).permute(0,2,1)
        #stage4
        y=self.mlp4(self.s4(y,y,y))
        y=self.maxpool1d(y.permute(0,2,1))
        N=y.shape[-1]
        y=y.reshape(B,self.out_chs[4],int(sqrt(N)),int(sqrt(N)))

        return y

if __name__ == '__main__':
    x=torch.randn(1,3,224,224)
    coatnet=CoAtNet(3,224)
    y=coatnet(x)
    print(y.shape)
    