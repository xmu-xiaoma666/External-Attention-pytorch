import torch
from torch import mean, nn
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

def _conv_bn(input_channel,output_channel,kernel_size=3,padding=1,stride=1,groups=1):
     res=nn.Sequential()
     res.add_module('conv',nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=kernel_size,padding=padding,padding_mode='zeros',stride=stride,groups=groups,bias=False))
     res.add_module('bn',nn.BatchNorm2d(output_channel))
     return res

class RepBlock(nn.Module):
     def __init__(self,input_channel,output_channel,kernel_size=3,groups=1,stride=1,deploy=False,use_se=False):
          super().__init__()
          self.use_se=use_se
          self.input_channel=input_channel
          self.output_channel=output_channel
          self.deploy=deploy
          self.kernel_size=kernel_size
          self.padding=kernel_size//2
          self.groups=groups
          self.activation=nn.ReLU()



          #make sure kernel_size=3 padding=1
          assert self.kernel_size==3
          assert self.padding==1
          if(not self.deploy):
               self.brb_3x3=_conv_bn(input_channel,output_channel,kernel_size=self.kernel_size,padding=self.padding,groups=groups)
               self.brb_1x1=_conv_bn(input_channel,output_channel,kernel_size=1,padding=0,groups=groups)
               self.brb_identity=nn.BatchNorm2d(self.input_channel) if self.input_channel == self.output_channel else None
          else:
               self.brb_rep=nn.Conv2d(in_channels=input_channel,out_channels=output_channel,kernel_size=self.kernel_size,padding=self.padding,padding_mode='zeros',stride=stride,bias=True)


     
     def forward(self, inputs):
          if(self.deploy):
               return self.activation(self.brb_rep(inputs))
          
          if(self.brb_identity==None):
               identity_out=0
          else:
               identity_out=self.brb_identity(inputs)
          
          return self.activation(self.brb_1x1(inputs)+self.brb_3x3(inputs)+identity_out)

     
     

     def _switch_to_deploy(self):
          self.deploy=True
          kernel,bias=self._get_equivalent_kernel_bias()
          self.brb_rep=nn.Conv2d(in_channels=self.brb_3x3.conv.in_channels,out_channels=self.brb_3x3.conv.out_channels,
                                   kernel_size=self.brb_3x3.conv.kernel_size,padding=self.brb_3x3.conv.padding,
                                   padding_mode=self.brb_3x3.conv.padding_mode,stride=self.brb_3x3.conv.stride,
                                   groups=self.brb_3x3.conv.groups,bias=True)
          self.brb_rep.weight.data=kernel
          self.brb_rep.bias.data=bias
          #消除梯度更新
          for para in self.parameters():
               para.detach_()
          #删除没用的分支
          self.__delattr__('brb_3x3')
          self.__delattr__('brb_1x1')
          self.__delattr__('brb_identity')


     #将1x1的卷积变成3x3的卷积参数
     def _pad_1x1_kernel(self,kernel):
          if(kernel is None):
               return 0
          else:
               return F.pad(kernel,[1]*4)


     #将identity，1x1,3x3的卷积融合到一起，变成一个3x3卷积的参数
     def _get_equivalent_kernel_bias(self):
          brb_3x3_weight,brb_3x3_bias=self._fuse_conv_bn(self.brb_3x3)
          brb_1x1_weight,brb_1x1_bias=self._fuse_conv_bn(self.brb_1x1)
          brb_id_weight,brb_id_bias=self._fuse_conv_bn(self.brb_identity)
          return brb_3x3_weight+self._pad_1x1_kernel(brb_1x1_weight)+brb_id_weight,brb_3x3_bias+brb_1x1_bias+brb_id_bias
     
     
     ### 将卷积和BN的参数融合到一起
     def _fuse_conv_bn(self,branch):
          if(branch is None):
               return 0,0
          elif(isinstance(branch,nn.Sequential)):
               kernel=branch.conv.weight
               running_mean=branch.bn.running_mean
               running_var=branch.bn.running_var
               gamma=branch.bn.weight
               beta=branch.bn.bias
               eps=branch.bn.eps
          else:
               assert isinstance(branch, nn.BatchNorm2d)
               if not hasattr(self, 'id_tensor'):
                    input_dim = self.input_channel // self.groups
                    kernel_value = np.zeros((self.input_channel, input_dim, 3, 3), dtype=np.float32)
                    for i in range(self.input_channel):
                         kernel_value[i, i % input_dim, 1, 1] = 1
                    self.id_tensor = torch.from_numpy(kernel_value).to(branch.weight.device)
               kernel = self.id_tensor
               running_mean = branch.running_mean
               running_var = branch.running_var
               gamma = branch.weight
               beta = branch.bias
               eps = branch.eps
          
          std=(running_var+eps).sqrt()
          t=gamma/std
          t=t.view(-1,1,1,1)
          return kernel*t,beta-running_mean*gamma/std
          


if __name__ == '__main__':
    input=torch.randn(50,512,49,49)
    repblock=RepBlock(512,512)
    repblock.eval()
    out=repblock(input)
    repblock._switch_to_deploy()
    out2=repblock(input)
    print('difference between vgg and repvgg')
    print(((out2-out)**2).sum())
    