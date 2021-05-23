from attention.DANet import DAModule
import torch

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    danet=DAModule(d_model=512,kernel_size=3,H=7,W=7)
    print(danet(input).shape)
