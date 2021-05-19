from collections import OrderedDict
import torch
from torch import nn


def exist(x):
    return x is not None

class Residual(nn.Module):
    def __init__(self,fn):
        super().__init__()
        self.fn=fn
    
    def forward(self,x):
        return self.fn(x)+x

class SpatialGatingUnit(nn.Module):
    def __init__(self,dim,len_sen):
        super().__init__()
        self.ln=nn.LayerNorm(dim)
        self.proj=nn.Conv1d(len_sen,len_sen,1)

        nn.init.zeros_(self.proj.weight)
        nn.init.ones_(self.proj.bias)
    
    def forward(self,x):
        res,gate=torch.chunk(x,2,-1) #bs,n,d_ff
        ###Norm
        gate=self.ln(gate) #bs,n,d_ff
        ###Spatial Proj
        gate=self.proj(gate) #bs,n,d_ff

        return res*gate

class gMLP(nn.Module):
    def __init__(self,num_tokens=None,len_sen=49,dim=512,d_ff=1024,num_layers=6):
        super().__init__()
        self.num_layers=num_layers
        self.embedding=nn.Embedding(num_tokens,dim) if exist(num_tokens) else nn.Identity()

        self.gmlp=nn.ModuleList([Residual(nn.Sequential(OrderedDict([
            ('ln1_%d'%i,nn.LayerNorm(dim)),
            ('fc1_%d'%i,nn.Linear(dim,d_ff*2)),
            ('gelu_%d'%i,nn.GELU()),
            ('sgu_%d'%i,SpatialGatingUnit(d_ff,len_sen)),
            ('fc2_%d'%i,nn.Linear(d_ff,dim)),
        ])))  for i in range(num_layers)])



        self.to_logits=nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim,num_tokens),
            nn.Softmax(-1)
        )


    def forward(self,x):
        #embedding
        embeded=self.embedding(x)

        #gMLP
        y=nn.Sequential(*self.gmlp)(embeded)


        #to logits
        logits=self.to_logits(y)


        return logits


            


if __name__ == '__main__':

    num_tokens=10000
    bs=50
    len_sen=49
    num_layers=6
    input=torch.randint(num_tokens,(bs,len_sen)) #bs,len_sen
    gmlp = gMLP(num_tokens=num_tokens,len_sen=len_sen,dim=512,d_ff=1024)
    output=gmlp(input)
    print(output.shape)
        