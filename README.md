# External-Attention-pytorch

Pytorch implementation of ["Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks"](https://arxiv.org/abs/2105.02358)
Pytorch implementation of ["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf)
Pytorch implementation of ["Squeeze-and-Excitation Networks"](https://arxiv.org/abs/1709.01507)



### Overview

![](./img/ExternalAttention.png)



### 1. External Attention Usage
#### 1.1. Paper
["Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks"](https://arxiv.org/abs/2105.02358)

#### 1.2. Overview
![](./img/External_Attention.png)

#### 1.3. Code
```python
from ExternalAttention import ExternalAttention
import torch

input=torch.randn(50,49,512)
ea = ExternalAttention(d_model=512,S=8)
output=ea(input)
print(output.shape)
```



### 2.Self Attention Usage
#### 2.1. Paper
["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf)

#### 1.2. Overview
![](./img/SA.png)

#### 1.3. Code
```python
from SelfAttention import ScaledDotProductAttention
import torch

input=torch.randn(50,49,512)
sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
output=sa(input,input,input)
print(output.shape)
```


### Simplified Self Attention Usage
#### 2.1. Paper
["None"]()

#### 1.2. Overview
![](./img/SSA.png)

#### 1.3. Code
```python
from SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
import torch

input=torch.randn(50,49,512)
ssa = SimplifiedScaledDotProductAttention(d_model=512, h=8)
output=ssa(input,input,input)
print(output.shape)

```



### 3. Squeeze-and-Excitation Attention Usage
#### 3.1. Paper
["Squeeze-and-Excitation Networks"](https://arxiv.org/abs/1709.01507)

#### 3.2. Overview
![](./img/SE.png)

#### 3.3. Code
```python
from SEAttention import SEAttention
import torch

input=torch.randn(50,512,7,7)
se = SEAttention(channel=512,reduction=8)
output=se(input)
print(output.shape)

```