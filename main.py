from mlp.g_mlp import gMLP
import torch

if __name__ == '__main__':
    
    num_tokens=10000
    bs=50
    len_sen=49
    num_layers=6
    input=torch.randint(num_tokens,(bs,len_sen)) #bs,len_sen
    gmlp = gMLP(num_tokens=num_tokens,len_sen=len_sen,dim=512,d_ff=1024)
    output=gmlp(input)
    print(output.shape)