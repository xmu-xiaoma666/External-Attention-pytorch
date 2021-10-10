import torch
from torch import nn


class MLP(nn.Module):
    def __init__(self,in_features,hidden_features,out_features,act_layer=nn.GELU,drop=0.1):
        super().__init__()
        self.fc1=nn.Linear(in_features,hidden_features)
        self.act=act_layer()
        self.fc2=nn.Linear(hidden_features,out_features)
        self.drop=nn.Dropout(drop)

    def forward(self, x) :
        return self.drop(self.fc2(self.drop(self.act(self.fc1(x)))))

class WeightedPermuteMLP(nn.Module):
    def __init__(self,dim,seg_dim=8, qkv_bias=False, proj_drop=0.):
        super().__init__()
        self.seg_dim=seg_dim

        self.mlp_c=nn.Linear(dim,dim,bias=qkv_bias)
        self.mlp_h=nn.Linear(dim,dim,bias=qkv_bias)
        self.mlp_w=nn.Linear(dim,dim,bias=qkv_bias)

        self.reweighting=MLP(dim,dim//4,dim*3)

        self.proj=nn.Linear(dim,dim)
        self.proj_drop=nn.Dropout(proj_drop)
    
    def forward(self,x) :
        B,H,W,C=x.shape

        c_embed=self.mlp_c(x)

        S=C//self.seg_dim
        h_embed=x.reshape(B,H,W,self.seg_dim,S).permute(0,3,2,1,4).reshape(B,self.seg_dim,W,H*S)
        h_embed=self.mlp_h(h_embed).reshape(B,self.seg_dim,W,H,S).permute(0,3,2,1,4).reshape(B,H,W,C)

        w_embed=x.reshape(B,H,W,self.seg_dim,S).permute(0,3,1,2,4).reshape(B,self.seg_dim,H,W*S)
        w_embed=self.mlp_w(w_embed).reshape(B,self.seg_dim,H,W,S).permute(0,2,3,1,4).reshape(B,H,W,C)

        weight=(c_embed+h_embed+w_embed).permute(0,3,1,2).flatten(2).mean(2)
        weight=self.reweighting(weight).reshape(B,C,3).permute(2,0,1).softmax(0).unsqueeze(2).unsqueeze(2)

        x=c_embed*weight[0]+w_embed*weight[1]+h_embed*weight[2]

        x=self.proj_drop(self.proj(x))

        return x



if __name__ == '__main__':
    input=torch.randn(64,8,8,512)
    seg_dim=8
    vip=WeightedPermuteMLP(512,seg_dim)
    out=vip(input)
    print(out.shape)
    