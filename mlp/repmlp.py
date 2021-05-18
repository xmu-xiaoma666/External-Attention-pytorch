import torch
from torch import nn
from collections import OrderedDict
from torch.nn import functional as F
import numpy as np
from numpy import random


def setup_seed(seed):
     torch.manual_seed(seed)
     torch.cuda.manual_seed_all(seed)
     np.random.seed(seed)
     random.seed(seed)
     torch.backends.cudnn.deterministic = True

class RepMLP(nn.Module):
    def __init__(self,C,O,H,W,h,w,fc1_fc2_reduction=1,fc3_groups=8,repconv_kernels=None,deploy=False):
        super().__init__()
        self.C=C
        self.O=O
        self.H=H
        self.W=W
        self.h=h
        self.w=w
        self.fc1_fc2_reduction=fc1_fc2_reduction
        self.repconv_kernels=repconv_kernels
        self.h_part=H//h
        self.w_part=W//w
        self.deploy=deploy
        self.fc3_groups=fc3_groups

        # make sure H,W can divided by h,w respectively
        assert H%h==0
        assert W%w==0

        self.is_global_perceptron= (H!=h) or (W!=w)
        ### global perceptron
        if(self.is_global_perceptron):
            if(not self.deploy):
                self.avg=nn.Sequential(OrderedDict([
                    ('avg',nn.AvgPool2d(kernel_size=(self.h,self.w))),
                    ('bn',nn.BatchNorm2d(num_features=C))
                ])
                )
            else:
                self.avg=nn.AvgPool2d(kernel_size=(self.h,self.w))
            hidden_dim=self.C//self.fc1_fc2_reduction
            self.fc1_fc2=nn.Sequential(OrderedDict([
                ('fc1',nn.Linear(C*self.h_part*self.w_part,hidden_dim)),
                ('relu',nn.ReLU()),
                ('fc2',nn.Linear(hidden_dim,C*self.h_part*self.w_part))
            ])
            )

        self.fc3=nn.Conv2d(self.C*self.h*self.w,self.O*self.h*self.w,kernel_size=1,groups=fc3_groups,bias=self.deploy)
        self.fc3_bn=nn.Identity() if self.deploy else nn.BatchNorm2d(self.O*self.h*self.w)
        
        if not self.deploy and self.repconv_kernels is not None:
            for k in self.repconv_kernels:
                repconv=nn.Sequential(OrderedDict([
                    ('conv',nn.Conv2d(self.C,self.O,kernel_size=k,padding=(k-1)//2, groups=fc3_groups,bias=False)),
                    ('bn',nn.BatchNorm2d(self.O))
                ])

                )
                self.__setattr__('repconv{}'.format(k),repconv)
                

    def switch_to_deploy(self):
        self.deploy=True
        fc1_weight,fc1_bias,fc3_weight,fc3_bias=self.get_equivalent_fc1_fc3_params()
        #del conv
        if(self.repconv_kernels is not None):
            for k in self.repconv_kernels:
                self.__delattr__('repconv{}'.format(k))
        #del fc3,bn
        self.__delattr__('fc3')
        self.__delattr__('fc3_bn')
        self.fc3 = nn.Conv2d(self.C * self.h * self.w, self.O * self.h * self.w, 1, 1, 0, bias=True, groups=self.fc3_groups)
        self.fc3_bn = nn.Identity()
        #   Remove the BN after AVG
        if self.is_global_perceptron:
            self.__delattr__('avg')
            self.avg = nn.AvgPool2d(kernel_size=(self.h, self.w))
        #   Set values
        if fc1_weight is not None:
            self.fc1_fc2.fc1.weight.data = fc1_weight
            self.fc1_fc2.fc1.bias.data = fc1_bias
        self.fc3.weight.data = fc3_weight
        self.fc3.bias.data = fc3_bias




    def get_equivalent_fc1_fc3_params(self):
        #training fc3+bn weight
        fc_weight,fc_bias=self._fuse_bn(self.fc3,self.fc3_bn)
        #training conv weight
        if(self.repconv_kernels is not None):
            max_kernel=max(self.repconv_kernels)
            max_branch=self.__getattr__('repconv{}'.format(max_kernel))
            conv_weight,conv_bias=self._fuse_bn(max_branch.conv,max_branch.bn)
            for k in self.repconv_kernels:
                if(k!=max_kernel):
                    tmp_branch=self.__getattr__('repconv{}'.format(k))
                    tmp_weight,tmp_bias=self._fuse_bn(tmp_branch.conv,tmp_branch.bn)
                    tmp_weight=F.pad(tmp_weight,[(max_kernel-k)//2]*4)
                    conv_weight+=tmp_weight
                    conv_bias+=tmp_bias
            repconv_weight,repconv_bias=self._conv_to_fc(conv_weight,conv_bias)
            final_fc3_weight=fc_weight+repconv_weight.reshape_as(fc_weight)
            final_fc3_bias=fc_bias+repconv_bias
        else:
            final_fc3_weight=fc_weight
            final_fc3_bias=fc_bias

        #fc1
        if(self.is_global_perceptron):
            #remove BN after avg
            avgbn = self.avg.bn
            std = (avgbn.running_var + avgbn.eps).sqrt()
            scale = avgbn.weight / std
            avgbias = avgbn.bias - avgbn.running_mean * scale
            fc1 = self.fc1_fc2.fc1
            replicate_times = fc1.in_features // len(avgbias)
            replicated_avgbias = avgbias.repeat_interleave(replicate_times).view(-1, 1)
            bias_diff = fc1.weight.matmul(replicated_avgbias).squeeze()
            final_fc1_bias = fc1.bias + bias_diff
            final_fc1_weight = fc1.weight * scale.repeat_interleave(replicate_times).view(1, -1)

        else:
            final_fc1_weight=None
            final_fc1_bias=None
        
        return final_fc1_weight,final_fc1_bias,final_fc3_weight,final_fc3_bias




    # def _conv_to_fc(self,weight,bias):
    #     i_maxtrix=torch.eye(self.C*self.h*self.w//self.fc3_groups).repeat(1,self.fc3_groups).reshape(self.C*self.h*self.w//self.fc3_groups,self.C,self.h,self.w)
    #     fc_weight=F.conv2d(i_maxtrix,weight=weight,bias=bias,padding=weight.shape[2]//2,groups=self.fc3_groups)
    #     fc_weight=fc_weight.reshape(self.C*self.h*self.w//self.fc3_groups,-1)
    #     fc_bias = bias.repeat_interleave(self.h * self.w)
    #     return fc_weight,fc_bias


    def _conv_to_fc(self,conv_kernel, conv_bias):
        I = torch.eye(self.C * self.h * self.w // self.fc3_groups).repeat(1, self.fc3_groups).reshape(self.C * self.h * self.w // self.fc3_groups, self.C, self.h, self.w).to(conv_kernel.device) 
        fc_k = F.conv2d(I, conv_kernel, padding=conv_kernel.size(2)//2, groups=self.fc3_groups)
        fc_k = fc_k.reshape(self.C * self.h * self.w // self.fc3_groups, self.O * self.h * self.w).t()
        fc_bias = conv_bias.repeat_interleave(self.h * self.w)
        return fc_k, fc_bias


    def _fuse_bn(self, conv_or_fc, bn):
        std = (bn.running_var + bn.eps).sqrt()
        t = bn.weight / std
        if conv_or_fc.weight.ndim == 4:
            t = t.reshape(-1, 1, 1, 1)
        else:
            t = t.reshape(-1, 1)
        return conv_or_fc.weight * t, bn.bias - bn.running_mean * bn.weight / std


    def forward(self,x) :
        ### global partition
        if(self.is_global_perceptron):
            input=x
            v=self.avg(x) #bs,C,h_part,w_part
            v=v.reshape(-1,self.C*self.h_part*self.w_part) #bs,C*h_part*w_part
            v=self.fc1_fc2(v) #bs,C*h_part*w_part
            v=v.reshape(-1,self.C,self.h_part,1,self.w_part,1) #bs,C,h_part,w_part
            input=input.reshape(-1,self.C,self.h_part,self.h,self.w_part,self.w) #bs,C,h_part,h,w_part,w
            input=v+input
        else:
            input=x.view(-1,self.C,self.h_part,self.h,self.w_part,self.w) #bs,C,h_part,h,w_part,w
        partition=input.permute(0,2,4,1,3,5) #bs,h_part,w_part,C,h,w

        ### partition partition
        fc3_out=partition.reshape(-1,self.C*self.h*self.w,1,1) #bs*h_part*w_part,C*h*w,1,1
        fc3_out=self.fc3_bn(self.fc3(fc3_out)) #bs*h_part*w_part,O*h*w,1,1
        fc3_out=fc3_out.reshape(-1,self.h_part,self.w_part,self.O,self.h,self.w) #bs,h_part,w_part,O,h,w

        ### local perceptron
        if(self.repconv_kernels is not None and not self.deploy):
            conv_input=partition.reshape(-1,self.C,self.h,self.w) #bs*h_part*w_part,C,h,w
            conv_out=0
            for k in self.repconv_kernels:
                repconv=self.__getattr__('repconv{}'.format(k))
                conv_out+=repconv(conv_input) ##bs*h_part*w_part,O,h,w
            conv_out=conv_out.view(-1,self.h_part,self.w_part,self.O,self.h,self.w) #bs,h_part,w_part,O,h,w
            fc3_out+=conv_out
        fc3_out=fc3_out.permute(0,3,1,4,2,5)#bs,O,h_part,h,w_part,w
        fc3_out=fc3_out.reshape(-1,self.C,self.H,self.W) #bs,O,H,W


        return fc3_out



if __name__ == '__main__':
    setup_seed(20)
    N=4 #batch size
    C=512 #input dim
    O=1024 #output dim
    H=14 #image height
    W=14 #image width
    h=7 #patch height
    w=7 #patch width
    fc1_fc2_reduction=1 #reduction ratio
    fc3_groups=8 # groups
    repconv_kernels=[1,3,5,7] #kernel list
    repmlp=RepMLP(C,O,H,W,h,w,fc1_fc2_reduction,fc3_groups,repconv_kernels=repconv_kernels)
    x=torch.randn(N,C,H,W)
    repmlp.eval()
    for module in repmlp.modules():
        if isinstance(module, nn.BatchNorm2d) or isinstance(module, nn.BatchNorm1d):
            nn.init.uniform_(module.running_mean, 0, 0.1)
            nn.init.uniform_(module.running_var, 0, 0.1)
            nn.init.uniform_(module.weight, 0, 0.1)
            nn.init.uniform_(module.bias, 0, 0.1)

    #training result
    out=repmlp(x)


    #inference result
    repmlp.switch_to_deploy()
    deployout = repmlp(x)

    print(((deployout-out)**2).sum())