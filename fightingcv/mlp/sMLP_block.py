import torch
from torch import nn







class sMLPBlock(nn.Module):
    def __init__(self,h=224,w=224,c=3):
        super().__init__()
        self.proj_h=nn.Linear(h,h)
        self.proj_w=nn.Linear(w,w)
        self.fuse=nn.Linear(3*c,c)
    
    def forward(self,x):
        x_h=self.proj_h(x.permute(0,1,3,2)).permute(0,1,3,2)
        x_w=self.proj_w(x)
        x_id=x
        x_fuse=torch.cat([x_h,x_w,x_id],dim=1)
        out=self.fuse(x_fuse.permute(0,2,3,1)).permute(0,3,1,2)
        return out


if __name__ == '__main__':
    input=torch.randn(50,3,224,224)
    smlp=sMLPBlock(h=224,w=224)
    out=smlp(input)
    print(out.shape)