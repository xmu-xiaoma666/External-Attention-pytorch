import torch
from torch import nn


class Rearange(nn.Module):
    def __init__(self,image_size=14,patch_size=7) :
        self.h=patch_size
        self.w=patch_size
        self.nw=image_size // patch_size
        self.nh=image_size // patch_size

        num_patches = (image_size // patch_size) ** 2
        super().__init__()

    def forward(self,x):
        ### bs,c,H,W
        bs,c,H,W=x.shape

        y=x.reshape(bs,c,self.h,self.nh,self.w,self.nw)
        y=y.permute(0,3,5,2,4,1) #bs,nh,nw,h,w,c
        y=y.contiguous().view(bs,self.nh*self.nw,-1) #bs,nh*nw,h*w*c
        return y

class Affine(nn.Module):
    def __init__(self, channel):
        super().__init__()
        self.g = nn.Parameter(torch.ones(1, 1, channel))
        self.b = nn.Parameter(torch.zeros(1, 1, channel))

    def forward(self, x):
        return x * self.g + self.b

class PreAffinePostLayerScale(nn.Module): # https://arxiv.org/abs/2103.17239
    def __init__(self, dim, depth, fn):
        super().__init__()
        if depth <= 18:
            init_eps = 0.1
        elif depth > 18 and depth <= 24:
            init_eps = 1e-5
        else:
            init_eps = 1e-6

        scale = torch.zeros(1, 1, dim).fill_(init_eps)
        self.scale = nn.Parameter(scale)
        self.affine = Affine(dim)
        self.fn = fn

    def forward(self, x):
        return self.fn(self.affine(x)) * self.scale + x


class ResMLP(nn.Module):
    def __init__(self,dim=128,image_size=14,patch_size=7,expansion_factor=4,depth=4,class_num=1000):
        super().__init__()
        self.flatten=Rearange(image_size,patch_size)
        num_patches = (image_size // patch_size) ** 2
        wrapper = lambda i, fn: PreAffinePostLayerScale(dim, i + 1, fn)
        self.embedding=nn.Linear((patch_size ** 2) * 3, dim)
        self.mlp=nn.Sequential()

        for i in range(depth):
            self.mlp.add_module('fc1_%d'%i,wrapper(i, nn.Conv1d(patch_size ** 2, patch_size ** 2, 1)))
            self.mlp.add_module('fc1_%d'%i,wrapper(i, nn.Sequential(
                    nn.Linear(dim, dim * expansion_factor),
                    nn.GELU(),
                    nn.Linear(dim * expansion_factor, dim)
                )))

        self.aff=Affine(dim)

        self.classifier=nn.Linear(dim,class_num)
        self.softmax=nn.Softmax(1)
    
    def forward(self, x) :
        y=self.flatten(x)
        y=self.embedding(y)
        y=self.mlp(y)
        y=self.aff(y)
        y=torch.mean(y,dim=1) #bs,dim
        out=self.softmax(self.classifier(y))
        return out

if __name__ == '__main__':
    input=torch.randn(50,3,14,14)
    resmlp=ResMLP(dim=128,image_size=14,patch_size=7,class_num=1000)
    out=resmlp(input)
    print(out.shape)

    