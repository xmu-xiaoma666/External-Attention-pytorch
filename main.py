from rep.repvgg import RepBlock
import torch


if __name__ == '__main__':
    input=torch.randn(50,512,49,49)
    repblock=RepBlock(512,512)
    repblock.eval()
    out=repblock(input)
    repblock._switch_to_deploy()
    out2=repblock(input)
    print('difference between vgg and repvgg')
    print(((out2-out)**2).sum())
    
