## Content
  
- [1. External Attention](#1-external-attention)

- [2. Self Attention](#2-self-attention)

- [3. Squeeze-and-Excitation(SE) Attention](#3-squeeze-and-excitationse-attention)

- [4. Selective Kernel(SK) Attention](#4-selective-kernelsk-attention)

- [5. CBAM Attention](#5-cbam-attention)

- [6. BAM Attention](#6-bam-attention)

- [7. ECA Attention](#7-eca-attention)

- [8. DANet Attention](#8-danet-attention)

- [9. Pyramid Split Attention(PSA)](#9-pyramid-split-attentionpsa)

- [10. Efficient Multi-Head Self-Attention(EMSA)](#10-efficient-multi-head-self-attentionemsa)

- [Write at the end](#Write_at_the_end)



## 1. External Attention

### 1.1. Citation

Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks.---arXiv 2021.05.05

Address：[https://arxiv.org/abs/2105.02358](https://arxiv.org/abs/2105.02358)

### 1.2. Model Structure

![](./img/External_Attention.png)

### 1.3. Brief
This is an article on arXiv in May. It mainly solves two pain points of Self-Attention (SA): (1) O(n^2) computational complexity; (2) SA is in the same sample The above calculates Attention based on different positions, ignoring the relationship between different samples. Therefore, this paper uses two serial MLP structures as memory units, which reduces the computational complexity to O(n); in addition, these two memory units are learned based on all training data, so they also implicitly consider the differences. The connection between the samples.

### 1.4. Usage

```python
from attention.ExternalAttention import ExternalAttention
import torch


input=torch.randn(50,49,512)
ea = ExternalAttention(d_model=512,S=8)
output=ea(input)
print(output.shape)
```





## 2. Self Attention

### 2.1. Citation

Attention Is All You Need---NeurIPS2017

Address：[https://arxiv.org/abs/1706.03762](https://arxiv.org/abs/1706.03762)

### 2.2. Model Structure

![](./img/SA.png)

### 2.3. Brief
This is an article published by Google in NeurIPS2017. It has a great influence in various fields such as CV, NLP, and multi-modality. The current citation volume has been 2.2w+. The Self-Attention proposed in Transformer is a kind of Attention, which is used to calculate the weight between different positions in the feature, so as to achieve the effect of updating the feature. First, the input feature is mapped into three features of Q, K, and V through FC, and then Q and K are dot-multiplied to obtain the attention map, and the attention map and V are dot-multiplied to obtain the weighted feature. Finally, the feature is mapped through FC, and a new feature is obtained. (There are many very good explanations about Transformer and Self-Attention on the Internet, so I won’t give a detailed introduction here)

### 2.4. Usage

```python
from attention.SelfAttention import ScaledDotProductAttention
import torch

input=torch.randn(50,49,512)
sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
output=sa(input,input,input)
print(output.shape)
```





## 3. Squeeze-and-Excitation(SE) Attention

### 3.1. Citation

Squeeze-and-Excitation Networks---CVPR2018

Address：[https://arxiv.org/abs/1709.01507](https://arxiv.org/abs/1709.01507)

### 3.2. Model Structure

![](./img/SE.png)

### 3.3. Brief
This is an article of CVPR2018, which is also very influential. The current citation volume is 7k+. This article is for channel attention. Because of its simple structure and effectiveness, it has set off a wave of channel attention. From the avenue to the simple, the idea of this article can be said to be very simple. First, the spatial dimension is applied to AdaptiveAvgPool, and then the channel attention is learned through two FCs, and the Sigmoid is used for normalization to obtain the Channel Attention Map, and finally the Channel Attention Map is combined with the original Multiply the features to get the weighted features.

### 3.4. Usage

```python
from attention.SEAttention import SEAttention
import torch

input=torch.randn(50,512,7,7)
se = SEAttention(channel=512,reduction=8)
output=se(input)
print(output.shape)
```



 

## 4. Selective Kernel(SK) Attention

### 4.1. Citation

Selective Kernel Networks---CVPR2019

Address：[https://arxiv.org/pdf/1903.06586.pdf](https://arxiv.org/pdf/1903.06586.pdf)

### 4.2. Model Structure

![](./img/SK.png)

### 4.3. Brief
This is an article from CVPR2019, which pays tribute to SENet's thoughts. In traditional CNN, each convolutional layer uses the same size convolution kernel, which limits the expressive ability of the model; and the "wider" model structure of Inception is also verified, using multiple different convolution kernels. Learning can indeed improve the expressive ability of the model. The author draws on the idea of ​​SENet, obtains the weight of the channel by dynamically calculating each convolution kernel, and dynamically merges the results of each convolution kernel.

I personally think that the reason why this article can also be called lightweight is that when channel attention is performed on the features of different kernels, the parameters are shared (ie because before Attention, the features are first fused, so different The result of the convolution kernel shares a parameter of the SE module).

The method in this article is divided into three parts: Split, Fuse, and Select. Split is a multi-branch operation, convolution with different convolution kernels to get different features; the Fuse part is to use the SE structure to obtain the channel attention matrix (N convolution kernels can get N attention matrices , This step is shared with all the feature parameters), so that the features of different kernels after SE can be obtained; the Select operation is to add these features.

### 4.4. Usage

```python
from attention.SKAttention import SKAttention
import torch

input=torch.randn(50,512,7,7)
se = SKAttention(channel=512,reduction=8)
output=se(input)
print(output.shape)
```



 

## 5. CBAM Attention

### 5.1. Citation

CBAM: Convolutional Block Attention Module---ECCV2018

Address：[https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)

### 5.2. Model Structure

![](./img/CBAM1.png)

![](./img/CBAM2.png)

### 5.3. Brief
This is an ECCV2018 paper. This article uses Channel Attention and Spatial Attention at the same time and connects the two in series (the article also does ablation experiments in parallel and two series).

In terms of Channel Attention, the general structure is still similar to SE, but the author proposes that AvgPool and MaxPool have different representation effects, so the author performs AvgPool and MaxPool on the original features in the Spatial dimension, and then uses the SE structure to extract channel attention. Note here The parameters are shared, and then the two features are added and normalized to obtain the attention matrix.

Spatial Attention is similar to Channel Attention. After performing two pools in the channel dimension, the two features are spliced, and then a 7x7 convolution is used to extract the Spatial Attention (the reason for using 7x7 is because the spatial attention is extracted, so use The convolution kernel must be large enough). Then do a normalization to get the spatial attention matrix.

### 5.4. Usage

```python
from attention.CBAM import CBAMBlock
import torch

input=torch.randn(50,512,7,7)
kernel_size=input.shape[2]
cbam = CBAMBlock(channel=512,reduction=16,kernel_size=kernel_size)
output=cbam(input)
print(output.shape)
```



 

## 6. BAM Attention

### 6.1. Citation

BAM: Bottleneck Attention Module---BMCV2018

Address：[https://arxiv.org/pdf/1807.06514.pdf](https://arxiv.org/pdf/1807.06514.pdf)

### 6.2. Model Structure

![](./img/BAM.png)

### 6.3. Brief
This is the work of CBAM and the author at the same time. The work is very similar to CBAM, and it is also dual attention. The difference is that CBAM connects the results of two attention in series; while BAM directly adds two attention matrices.

In terms of Channel Attention, the structure is basically the same as SE. In terms of Spatial Attention, the pool is still performed in the channel dimension, and then a 3x3 hole convolution is used twice, and finally a 1x1 convolution will be used to obtain the Spatial Attention matrix.

Finally, the Channel Attention and Spatial Attention matrices are added (the broadcast mechanism is used here) and normalized. In this way, the attention matrix that combines space and channel is obtained.

### 6.4. Usage

```python
from attention.BAM import BAMBlock
import torch

input=torch.randn(50,512,7,7)
bam = BAMBlock(channel=512,reduction=16,dia_val=2)
output=bam(input)
print(output.shape)
```





## 7. ECA Attention

### 7.1. Citation

ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks---CVPR2020

Address：[https://arxiv.org/pdf/1910.03151.pdf](https://arxiv.org/pdf/1910.03151.pdf)

### 7.2. Model Structure

![](./img/ECA.png)

### 7.3. Brief
This is an article of CVPR2020.

As shown in the figure above, SE uses two fully connected layers to achieve channel attention, while ECA requires one convolution. The reason why the author did this is that it is not necessary to calculate the attention between all channels. On the other hand, the use of two fully connected layers does introduce too many parameters and calculations.

Therefore, after the author performed AvgPool, he only used a one-dimensional convolution with a receptive field of k (equivalent to only calculating the attention of the adjacent k channels), which greatly reduced the parameters and calculation amount. (i.e. is equivalent to SE being a global attention, while ECA is a local attention).
### 7.4. Usage

```python
from attention.ECAAttention import ECAAttention
import torch

input=torch.randn(50,512,7,7)
eca = ECAAttention(kernel_size=3)
output=eca(input)
print(output.shape)
```



 

## 8. DANet Attention

### 8.1. Citation

Dual Attention Network for Scene Segmentation---CVPR2019

Address：[https://arxiv.org/pdf/1809.02983.pdf](https://arxiv.org/pdf/1809.02983.pdf)

### 8.2. Model Structure

![](./img/danet.png)


### 8.3. Brief
This is an article by CVPR2019. The idea is very simple, that is, self-attention is used in the task of scene segmentation. The difference is that self-attention is to pay attention to the attention between each position, and this article will make a self-attention. To expand, we also made a branch of channel attention. The operation is the same as self-attention. The three Linears that generate Q, K, and V are removed from different channel attention. Finally, the features after the two attentions are summed element-wise.

### 8.4. Usage

```python
from attention.DANet import DAModule
import torch

input=torch.randn(50,512,7,7)
danet=DAModule(d_model=512,kernel_size=3,H=7,W=7)
print(danet(input).shape)
```



 

## 9. Pyramid Split Attention(PSA)

### 9.1. Citation

EPSANet: An Efficient Pyramid Split Attention Block on Convolutional Neural Network---arXiv 2021.05.30

Address：[https://arxiv.org/pdf/2105.14447.pdf](https://arxiv.org/pdf/2105.14447.pdf)

### 9.2. Model Structure

![](./img/psa.png)


### 9.3. Brief
This is an article uploaded by Shenzhen University on arXiv on May 30. The purpose of this article is how to obtain and explore spatial information of different scales to enrich the feature space. The network structure is relatively simple, mainly divided into four steps. In the first part, the original feature is divided into n groups according to the channel, and then the different groups are convolved with different scales to obtain the new feature W1; the second part is SE performs SE on the original features to obtain different Attention Map; the third part is to perform softmax on different groups; the fourth part is to multiply the obtained attention with the original feature W1.
### 9.4. Usage

```python
from attention.PSA import PSA
import torch

input=torch.randn(50,512,7,7)
psa = PSA(channel=512,reduction=8)
output=psa(input)
print(output.shape)
```



 

## 10. Efficient Multi-Head Self-Attention(EMSA)

### 10.1. Citation

ResT: An Efficient Transformer for Visual Recognition---arXiv 2021.05.28

Address：[https://arxiv.org/abs/2105.13677](https://arxiv.org/abs/2105.13677)

### 10.2. Model Structure

![](./img/EMSA.png)

### 10.3. Brief

This is an article uploaded by Nanjing University on arXiv on May 28. This article mainly solves two pain points of SA: (1) The computational complexity of Self-Attention is squared with n; (2) Each head has only partial information of q, k, v, if q, k, v If the dimension of is too small, continuous information will not be obtained, resulting in performance loss. The idea given in this article is also very simple. In SA, before FC, a convolution is used to reduce the spatial dimension, thereby obtaining smaller K and V in spatial dimension.

### 10.4. Usage

```python
from attention.EMSA import EMSA
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,64,512)
emsa = EMSA(d_model=512, d_k=512, d_v=512, h=8,H=8,W=8,ratio=2,apply_transform=True)
output=emsa(input,input,input)
print(output.shape)
```



 

## Write at the end
At present, the Attention work organized by this project is indeed not comprehensive enough. As the amount of reading increases, we will continue to improve this project. Welcome everyone star to support. If there are incorrect statements or incorrect code implementations in the article, you are welcome to point out~