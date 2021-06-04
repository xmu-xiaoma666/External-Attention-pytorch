from attention.PSA import PSA
import torch

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    psa = PSA(channel=512,reduction=8)
    output=psa(input)
    print(output.shape)

