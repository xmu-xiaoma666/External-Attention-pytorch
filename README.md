
<img src="./FightingCVimg/LOGO.gif" height="200" width="400"/>

简体中文 | [English](./README_EN.md)

# FightingCV 代码库， 包含 [***Attention***](#attention-series),[***Backbone***](#backbone-series), [***MLP***](#mlp-series), [***Re-parameter***](#re-parameter-series), [**Convolution**](#convolution-series)

![](https://img.shields.io/badge/fightingcv-v0.0.1-brightgreen)
![](https://img.shields.io/badge/python->=v3.0-blue)
![](https://img.shields.io/badge/pytorch->=v1.4-red)

<!--
-------
*If this project is helpful to you, welcome to give a ***star***.* 

*Don't forget to ***follow*** me to learn about project updates.*

-->




Hello，大家好，我是小马🚀🚀🚀

***For 小白（Like Me）：***
最近在读论文的时候会发现一个问题，有时候论文核心思想非常简单，核心代码可能也就十几行。但是打开作者release的源码时，却发现提出的模块嵌入到分类、检测、分割等任务框架中，导致代码比较冗余，对于特定任务框架不熟悉的我，**很难找到核心代码**，导致在论文和网络思想的理解上会有一定困难。

***For 进阶者（Like You）：***
如果把Conv、FC、RNN这些基本单元看做小的Lego积木，把Transformer、ResNet这些结构看成已经搭好的Lego城堡。那么本项目提供的模块就是一个个具有完整语义信息的Lego组件。**让科研工作者们避免反复造轮子**，只需思考如何利用这些“Lego组件”，搭建出更多绚烂多彩的作品。

***For 大神（May Be Like You）：***
能力有限，**不喜轻喷**！！！

***For All：***
本项目致力于实现一个既能**让深度学习小白也能搞懂**，又能**服务科研和工业社区**的代码库。

<!--
作为[**FightingCV公众号**](https://mp.weixin.qq.com/s/m9RiivbbDPdjABsTd6q8FA)和 **[FightingCV-Paper-Reading](https://github.com/xmu-xiaoma666/FightingCV-Paper-Reading)** 的补充，本项目的宗旨是从代码角度，实现🚀**让世界上没有难读的论文**🚀。
-->

（同时也非常欢迎各位科研工作者将自己的工作的核心代码整理到本项目中，推动科研社区的发展，会在readme中注明代码的作者~）



<!--

## 技术交流 <img title="" src="https://user-images.githubusercontent.com/48054808/157800467-2a9946ad-30d1-49a9-b9db-ba33413d9c90.png" alt="" width="20">

欢迎大家关注公众号：**FightingCV**



| FightingCV公众号 | 小助手微信 （备注【**公司/学校+方向+ID**】）|
:-------------------------:|:-------------------------:
<img src='./FightingCVimg/FightingCV.jpg' width='200px'>  |  <img src='./FightingCVimg/xiaozhushou.jpg' width='200px'> 

- 公众号**每天**都会进行**论文、算法和代码的干货分享**哦~

- **交流群每天分享一些最新的论文和解析**，欢迎大家一起**学习交流**哈~~~
（加不进去可以加微信：**775629340**，记得备注【**公司/学校+方向+ID**】）


- 强烈推荐大家关注[**知乎**](https://www.zhihu.com/people/jason-14-58-38/posts)账号和[**FightingCV公众号**](https://mp.weixin.qq.com/s/m9RiivbbDPdjABsTd6q8FA)，可以快速了解到最新优质的干货资源。
-->

-------

## 新增

- 支持通过 pip 方式使用该代码库

## 使用

### 安装

 直接通过 pip 安装

  ```shell
  pip install fightingcv-attention
  ```


或克隆该仓库

  ```shell
  git clone https://github.com/xmu-xiaoma666/External-Attention-pytorch.git

  cd External-Attention-pytorch
  ```

### 演示

#### 使用 pip 方式
```python
import torch
from torch import nn
from torch.nn import functional as F

# 使用 pip 方式

from fightingcv_attention.attention.MobileViTv2Attention import *

if __name__ == '__main__':
    input=torch.randn(50,49,512)
    sa = MobileViTv2Attention(d_model=512)
    output=sa(input)
    print(output.shape)
```

 - pip包 内置模块使用参考: [fightingcv-attention 说明文档](./README_pip.md)

#### 使用 git 方式
```python
import torch
from torch import nn
from torch.nn import functional as F

# 与 pip方式 区别在于 将 `fightingcv_attention` 替换 `model`

from model.attention.MobileViTv2Attention import *

if __name__ == '__main__':
    input=torch.randn(50,49,512)
    sa = MobileViTv2Attention(d_model=512)
    output=sa(input)
    print(output.shape)
```

-------



# 目录

- [Attention Series](#attention-series)
    - [1. External Attention Usage](#1-external-attention-usage)

    - [2. Self Attention Usage](#2-self-attention-usage)

    - [3. Simplified Self Attention Usage](#3-simplified-self-attention-usage)

    - [4. Squeeze-and-Excitation Attention Usage](#4-squeeze-and-excitation-attention-usage)

    - [5. SK Attention Usage](#5-sk-attention-usage)

    - [6. CBAM Attention Usage](#6-cbam-attention-usage)

    - [7. BAM Attention Usage](#7-bam-attention-usage)
    
    - [8. ECA Attention Usage](#8-eca-attention-usage)

    - [9. DANet Attention Usage](#9-danet-attention-usage)

    - [10. Pyramid Split Attention (PSA) Usage](#10-Pyramid-Split-Attention-Usage)

    - [11. Efficient Multi-Head Self-Attention(EMSA) Usage](#11-Efficient-Multi-Head-Self-Attention-Usage)

    - [12. Shuffle Attention Usage](#12-Shuffle-Attention-Usage)
    
    - [13. MUSE Attention Usage](#13-MUSE-Attention-Usage)
  
    - [14. SGE Attention Usage](#14-SGE-Attention-Usage)

    - [15. A2 Attention Usage](#15-A2-Attention-Usage)

    - [16. AFT Attention Usage](#16-AFT-Attention-Usage)

    - [17. Outlook Attention Usage](#17-Outlook-Attention-Usage)

    - [18. ViP Attention Usage](#18-ViP-Attention-Usage)

    - [19. CoAtNet Attention Usage](#19-CoAtNet-Attention-Usage)

    - [20. HaloNet Attention Usage](#20-HaloNet-Attention-Usage)

    - [21. Polarized Self-Attention Usage](#21-Polarized-Self-Attention-Usage)

    - [22. CoTAttention Usage](#22-CoTAttention-Usage)

    - [23. Residual Attention Usage](#23-Residual-Attention-Usage)
  
    - [24. S2 Attention Usage](#24-S2-Attention-Usage)

    - [25. GFNet Attention Usage](#25-GFNet-Attention-Usage)

    - [26. Triplet Attention Usage](#26-TripletAttention-Usage)

    - [27. Coordinate Attention Usage](#27-Coordinate-Attention-Usage)

    - [28. MobileViT Attention Usage](#28-MobileViT-Attention-Usage)

    - [29. ParNet Attention Usage](#29-ParNet-Attention-Usage)

    - [30. UFO Attention Usage](#30-UFO-Attention-Usage)

    - [31. ACmix Attention Usage](#31-Acmix-Attention-Usage)
  
    - [32. MobileViTv2 Attention Usage](#32-MobileViTv2-Attention-Usage)

    - [33. DAT Attention Usage](#33-DAT-Attention-Usage)

    - [34. CrossFormer Attention Usage](#34-CrossFormer-Attention-Usage)

    - [35. MOATransformer Attention Usage](#35-MOATransformer-Attention-Usage)

    - [36. CrissCrossAttention Attention Usage](#36-CrissCrossAttention-Attention-Usage)

    - [37. Axial_attention Attention Usage](#37-Axial_attention-Attention-Usage)

- [Backbone Series](#Backbone-series)

    - [1. ResNet Usage](#1-ResNet-Usage)

    - [2. ResNeXt Usage](#2-ResNeXt-Usage)

    - [3. MobileViT Usage](#3-MobileViT-Usage)

    - [4. ConvMixer Usage](#4-ConvMixer-Usage)

    - [5. ShuffleTransformer Usage](#5-ShuffleTransformer-Usage)

    - [6. ConTNet Usage](#6-ConTNet-Usage)

    - [7. HATNet Usage](#7-HATNet-Usage)

    - [8. CoaT Usage](#8-CoaT-Usage)

    - [9. PVT Usage](#9-PVT-Usage)

    - [10. CPVT Usage](#10-CPVT-Usage)

    - [11. PIT Usage](#11-PIT-Usage)

    - [12. CrossViT Usage](#12-CrossViT-Usage)

    - [13. TnT Usage](#13-TnT-Usage)

    - [14. DViT Usage](#14-DViT-Usage)

    - [15. CeiT Usage](#15-CeiT-Usage)

    - [16. ConViT Usage](#16-ConViT-Usage)

    - [17. CaiT Usage](#17-CaiT-Usage)

    - [18. PatchConvnet Usage](#18-PatchConvnet-Usage)

    - [19. DeiT Usage](#19-DeiT-Usage)

    - [20. LeViT Usage](#20-LeViT-Usage)

    - [21. VOLO Usage](#21-VOLO-Usage)
    
    - [22. Container Usage](#22-Container-Usage)

    - [23. CMT Usage](#23-CMT-Usage)

    - [24. EfficientFormer Usage](#24-EfficientFormer-Usage)

    - [25. ConvNeXtV2 Usage](#25-ConvNeXtV2-Usage)



- [MLP Series](#mlp-series)

    - [1. RepMLP Usage](#1-RepMLP-Usage)

    - [2. MLP-Mixer Usage](#2-MLP-Mixer-Usage)

    - [3. ResMLP Usage](#3-ResMLP-Usage)

    - [4. gMLP Usage](#4-gMLP-Usage)

    - [5. sMLP Usage](#5-sMLP-Usage)

    - [6. vip-mlp Usage](#6-vip-mlp-Usage)

- [Re-Parameter(ReP) Series](#Re-Parameter-series)

    - [1. RepVGG Usage](#1-RepVGG-Usage)

    - [2. ACNet Usage](#2-ACNet-Usage)

    - [3. Diverse Branch Block(DDB) Usage](#3-Diverse-Branch-Block-Usage)

- [Convolution Series](#Convolution-series)

    - [1. Depthwise Separable Convolution Usage](#1-Depthwise-Separable-Convolution-Usage)

    - [2. MBConv Usage](#2-MBConv-Usage)

    - [3. Involution Usage](#3-Involution-Usage)

    - [4. DynamicConv Usage](#4-DynamicConv-Usage)

    - [5. CondConv Usage](#5-CondConv-Usage)

***



# Attention Series

- Pytorch implementation of ["Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks---arXiv 2021.05.05"](https://arxiv.org/abs/2105.02358)

- Pytorch implementation of ["Attention Is All You Need---NIPS2017"](https://arxiv.org/pdf/1706.03762.pdf)

- Pytorch implementation of ["Squeeze-and-Excitation Networks---CVPR2018"](https://arxiv.org/abs/1709.01507)

- Pytorch implementation of ["Selective Kernel Networks---CVPR2019"](https://arxiv.org/pdf/1903.06586.pdf)

- Pytorch implementation of ["CBAM: Convolutional Block Attention Module---ECCV2018"](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)

- Pytorch implementation of ["BAM: Bottleneck Attention Module---BMCV2018"](https://arxiv.org/pdf/1807.06514.pdf)

- Pytorch implementation of ["ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks---CVPR2020"](https://arxiv.org/pdf/1910.03151.pdf)

- Pytorch implementation of ["Dual Attention Network for Scene Segmentation---CVPR2019"](https://arxiv.org/pdf/1809.02983.pdf)

- Pytorch implementation of ["EPSANet: An Efficient Pyramid Split Attention Block on Convolutional Neural Network---arXiv 2021.05.30"](https://arxiv.org/pdf/2105.14447.pdf)

- Pytorch implementation of ["ResT: An Efficient Transformer for Visual Recognition---arXiv 2021.05.28"](https://arxiv.org/abs/2105.13677)

- Pytorch implementation of ["SA-NET: SHUFFLE ATTENTION FOR DEEP CONVOLUTIONAL NEURAL NETWORKS---ICASSP 2021"](https://arxiv.org/pdf/2102.00240.pdf)

- Pytorch implementation of ["MUSE: Parallel Multi-Scale Attention for Sequence to Sequence Learning---arXiv 2019.11.17"](https://arxiv.org/abs/1911.09483)

- Pytorch implementation of ["Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks---arXiv 2019.05.23"](https://arxiv.org/pdf/1905.09646.pdf)

- Pytorch implementation of ["A2-Nets: Double Attention Networks---NIPS2018"](https://arxiv.org/pdf/1810.11579.pdf)


- Pytorch implementation of ["An Attention Free Transformer---ICLR2021 (Apple New Work)"](https://arxiv.org/pdf/2105.14103v1.pdf)


- Pytorch implementation of [VOLO: Vision Outlooker for Visual Recognition---arXiv 2021.06.24"](https://arxiv.org/abs/2106.13112) 
  [【论文解析】](https://zhuanlan.zhihu.com/p/385561050)


- Pytorch implementation of [Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition---arXiv 2021.06.23](https://arxiv.org/abs/2106.12368) 
  [【论文解析】](https://mp.weixin.qq.com/s/5gonUQgBho_m2O54jyXF_Q)


- Pytorch implementation of [CoAtNet: Marrying Convolution and Attention for All Data Sizes---arXiv 2021.06.09](https://arxiv.org/abs/2106.04803) 
  [【论文解析】](https://zhuanlan.zhihu.com/p/385578588)


- Pytorch implementation of [Scaling Local Self-Attention for Parameter Efficient Visual Backbones---CVPR2021 Oral](https://arxiv.org/pdf/2103.12731.pdf)  [【论文解析】](https://zhuanlan.zhihu.com/p/388598744)



- Pytorch implementation of [Polarized Self-Attention: Towards High-quality Pixel-wise Regression---arXiv 2021.07.02](https://arxiv.org/abs/2107.00782)  [【论文解析】](https://zhuanlan.zhihu.com/p/389770482) 


- Pytorch implementation of [Contextual Transformer Networks for Visual Recognition---arXiv 2021.07.26](https://arxiv.org/abs/2107.12292)  [【论文解析】](https://zhuanlan.zhihu.com/p/394795481) 


- Pytorch implementation of [Residual Attention: A Simple but Effective Method for Multi-Label Recognition---ICCV2021](https://arxiv.org/abs/2108.02456) 


- Pytorch implementation of [S²-MLPv2: Improved Spatial-Shift MLP Architecture for Vision---arXiv 2021.08.02](https://arxiv.org/abs/2108.01072) [【论文解析】](https://zhuanlan.zhihu.com/p/397003638) 

- Pytorch implementation of [Global Filter Networks for Image Classification---arXiv 2021.07.01](https://arxiv.org/abs/2107.00645) 

- Pytorch implementation of [Rotate to Attend: Convolutional Triplet Attention Module---WACV 2021](https://arxiv.org/abs/2010.03045) 

- Pytorch implementation of [Coordinate Attention for Efficient Mobile Network Design ---CVPR 2021](https://arxiv.org/abs/2103.02907)

- Pytorch implementation of [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer---ArXiv 2021.10.05](https://arxiv.org/abs/2110.02178)

- Pytorch implementation of [Non-deep Networks---ArXiv 2021.10.20](https://arxiv.org/abs/2110.07641)

- Pytorch implementation of [UFO-ViT: High Performance Linear Vision Transformer without Softmax---ArXiv 2021.09.29](https://arxiv.org/abs/2109.14382)

- Pytorch implementation of [Separable Self-attention for Mobile Vision Transformers---ArXiv 2022.06.06](https://arxiv.org/abs/2206.02680)

- Pytorch implementation of [On the Integration of Self-Attention and Convolution---ArXiv 2022.03.14](https://arxiv.org/pdf/2111.14556.pdf)

- Pytorch implementation of [CROSSFORMER: A VERSATILE VISION TRANSFORMER HINGING ON CROSS-SCALE ATTENTION---ICLR 2022](https://arxiv.org/pdf/2108.00154.pdf)

- Pytorch implementation of [Aggregating Global Features into Local Vision Transformer](https://arxiv.org/abs/2201.12903)

- Pytorch implementation of [CCNet: Criss-Cross Attention for Semantic Segmentation](https://arxiv.org/abs/1811.11721)

- Pytorch implementation of [Axial Attention in Multidimensional Transformers](https://arxiv.org/abs/1912.12180)
***


### 1. External Attention Usage
#### 1.1. Paper
["Beyond Self-attention: External Attention using Two Linear Layers for Visual Tasks"](https://arxiv.org/abs/2105.02358)

#### 1.2. Overview
![](./model/img/External_Attention.png)

#### 1.3. Usage Code
```python
from model.attention.ExternalAttention import ExternalAttention
import torch

input=torch.randn(50,49,512)
ea = ExternalAttention(d_model=512,S=8)
output=ea(input)
print(output.shape)
```

***


### 2. Self Attention Usage
#### 2.1. Paper
["Attention Is All You Need"](https://arxiv.org/pdf/1706.03762.pdf)

#### 1.2. Overview
![](./model/img/SA.png)

#### 1.3. Usage Code
```python
from model.attention.SelfAttention import ScaledDotProductAttention
import torch

input=torch.randn(50,49,512)
sa = ScaledDotProductAttention(d_model=512, d_k=512, d_v=512, h=8)
output=sa(input,input,input)
print(output.shape)
```

***

### 3. Simplified Self Attention Usage
#### 3.1. Paper
[None]()

#### 3.2. Overview
![](./model/img/SSA.png)

#### 3.3. Usage Code
```python
from model.attention.SimplifiedSelfAttention import SimplifiedScaledDotProductAttention
import torch

input=torch.randn(50,49,512)
ssa = SimplifiedScaledDotProductAttention(d_model=512, h=8)
output=ssa(input,input,input)
print(output.shape)

```

***

### 4. Squeeze-and-Excitation Attention Usage
#### 4.1. Paper
["Squeeze-and-Excitation Networks"](https://arxiv.org/abs/1709.01507)

#### 4.2. Overview
![](./model/img/SE.png)

#### 4.3. Usage Code
```python
from model.attention.SEAttention import SEAttention
import torch

input=torch.randn(50,512,7,7)
se = SEAttention(channel=512,reduction=8)
output=se(input)
print(output.shape)

```

***

### 5. SK Attention Usage
#### 5.1. Paper
["Selective Kernel Networks"](https://arxiv.org/pdf/1903.06586.pdf)

#### 5.2. Overview
![](./model/img/SK.png)

#### 5.3. Usage Code
```python
from model.attention.SKAttention import SKAttention
import torch

input=torch.randn(50,512,7,7)
se = SKAttention(channel=512,reduction=8)
output=se(input)
print(output.shape)

```
***

### 6. CBAM Attention Usage
#### 6.1. Paper
["CBAM: Convolutional Block Attention Module"](https://openaccess.thecvf.com/content_ECCV_2018/papers/Sanghyun_Woo_Convolutional_Block_Attention_ECCV_2018_paper.pdf)

#### 6.2. Overview
![](./model/img/CBAM1.png)

![](./model/img/CBAM2.png)

#### 6.3. Usage Code
```python
from model.attention.CBAM import CBAMBlock
import torch

input=torch.randn(50,512,7,7)
kernel_size=input.shape[2]
cbam = CBAMBlock(channel=512,reduction=16,kernel_size=kernel_size)
output=cbam(input)
print(output.shape)

```

***

### 7. BAM Attention Usage
#### 7.1. Paper
["BAM: Bottleneck Attention Module"](https://arxiv.org/pdf/1807.06514.pdf)

#### 7.2. Overview
![](./model/img/BAM.png)

#### 7.3. Usage Code
```python
from model.attention.BAM import BAMBlock
import torch

input=torch.randn(50,512,7,7)
bam = BAMBlock(channel=512,reduction=16,dia_val=2)
output=bam(input)
print(output.shape)

```

***

### 8. ECA Attention Usage
#### 8.1. Paper
["ECA-Net: Efficient Channel Attention for Deep Convolutional Neural Networks"](https://arxiv.org/pdf/1910.03151.pdf)

#### 8.2. Overview
![](./model/img/ECA.png)

#### 8.3. Usage Code
```python
from model.attention.ECAAttention import ECAAttention
import torch

input=torch.randn(50,512,7,7)
eca = ECAAttention(kernel_size=3)
output=eca(input)
print(output.shape)

```

***

### 9. DANet Attention Usage
#### 9.1. Paper
["Dual Attention Network for Scene Segmentation"](https://arxiv.org/pdf/1809.02983.pdf)

#### 9.2. Overview
![](./model/img/danet.png)

#### 9.3. Usage Code
```python
from model.attention.DANet import DAModule
import torch

input=torch.randn(50,512,7,7)
danet=DAModule(d_model=512,kernel_size=3,H=7,W=7)
print(danet(input).shape)

```

***

### 10. Pyramid Split Attention Usage

#### 10.1. Paper
["EPSANet: An Efficient Pyramid Split Attention Block on Convolutional Neural Network"](https://arxiv.org/pdf/2105.14447.pdf)

#### 10.2. Overview
![](./model/img/psa.png)

#### 10.3. Usage Code
```python
from model.attention.PSA import PSA
import torch

input=torch.randn(50,512,7,7)
psa = PSA(channel=512,reduction=8)
output=psa(input)
print(output.shape)

```

***


### 11. Efficient Multi-Head Self-Attention Usage

#### 11.1. Paper
["ResT: An Efficient Transformer for Visual Recognition"](https://arxiv.org/abs/2105.13677)

#### 11.2. Overview
![](./model/img/EMSA.png)

#### 11.3. Usage Code
```python

from model.attention.EMSA import EMSA
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,64,512)
emsa = EMSA(d_model=512, d_k=512, d_v=512, h=8,H=8,W=8,ratio=2,apply_transform=True)
output=emsa(input,input,input)
print(output.shape)
    
```

***


### 12. Shuffle Attention Usage

#### 12.1. Paper
["SA-NET: SHUFFLE ATTENTION FOR DEEP CONVOLUTIONAL NEURAL NETWORKS"](https://arxiv.org/pdf/2102.00240.pdf)

#### 12.2. Overview
![](./model/img/ShuffleAttention.jpg)

#### 12.3. Usage Code
```python

from model.attention.ShuffleAttention import ShuffleAttention
import torch
from torch import nn
from torch.nn import functional as F


input=torch.randn(50,512,7,7)
se = ShuffleAttention(channel=512,G=8)
output=se(input)
print(output.shape)

    
```


***


### 13. MUSE Attention Usage

#### 13.1. Paper
["MUSE: Parallel Multi-Scale Attention for Sequence to Sequence Learning"](https://arxiv.org/abs/1911.09483)

#### 13.2. Overview
![](./model/img/MUSE.png)

#### 13.3. Usage Code
```python
from model.attention.MUSEAttention import MUSEAttention
import torch
from torch import nn
from torch.nn import functional as F


input=torch.randn(50,49,512)
sa = MUSEAttention(d_model=512, d_k=512, d_v=512, h=8)
output=sa(input,input,input)
print(output.shape)

```

***


### 14. SGE Attention Usage

#### 14.1. Paper
[Spatial Group-wise Enhance: Improving Semantic Feature Learning in Convolutional Networks](https://arxiv.org/pdf/1905.09646.pdf)

#### 14.2. Overview
![](./model/img/SGE.png)

#### 14.3. Usage Code
```python
from model.attention.SGE import SpatialGroupEnhance
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
sge = SpatialGroupEnhance(groups=8)
output=sge(input)
print(output.shape)

```

***


### 15. A2 Attention Usage

#### 15.1. Paper
[A2-Nets: Double Attention Networks](https://arxiv.org/pdf/1810.11579.pdf)

#### 15.2. Overview
![](./model/img/A2.png)

#### 15.3. Usage Code
```python
from model.attention.A2Atttention import DoubleAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
a2 = DoubleAttention(512,128,128,True)
output=a2(input)
print(output.shape)

```



### 16. AFT Attention Usage

#### 16.1. Paper
[An Attention Free Transformer](https://arxiv.org/pdf/2105.14103v1.pdf)

#### 16.2. Overview
![](./model/img/AFT.jpg)

#### 16.3. Usage Code
```python
from model.attention.AFT import AFT_FULL
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,49,512)
aft_full = AFT_FULL(d_model=512, n=49)
output=aft_full(input)
print(output.shape)

```






### 17. Outlook Attention Usage

#### 17.1. Paper


[VOLO: Vision Outlooker for Visual Recognition"](https://arxiv.org/abs/2106.13112)


#### 17.2. Overview
![](./model/img/OutlookAttention.png)

#### 17.3. Usage Code
```python
from model.attention.OutlookAttention import OutlookAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,28,28,512)
outlook = OutlookAttention(dim=512)
output=outlook(input)
print(output.shape)

```


***






### 18. ViP Attention Usage

#### 18.1. Paper


[Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition"](https://arxiv.org/abs/2106.12368)


#### 18.2. Overview
![](./model/img/ViP.png)

#### 18.3. Usage Code
```python

from model.attention.ViP import WeightedPermuteMLP
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(64,8,8,512)
seg_dim=8
vip=WeightedPermuteMLP(512,seg_dim)
out=vip(input)
print(out.shape)

```


***





### 19. CoAtNet Attention Usage

#### 19.1. Paper


[CoAtNet: Marrying Convolution and Attention for All Data Sizes"](https://arxiv.org/abs/2106.04803) 


#### 19.2. Overview
None


#### 19.3. Usage Code
```python

from model.attention.CoAtNet import CoAtNet
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,3,224,224)
mbconv=CoAtNet(in_ch=3,image_size=224)
out=mbconv(input)
print(out.shape)

```


***






### 20. HaloNet Attention Usage

#### 20.1. Paper


[Scaling Local Self-Attention for Parameter Efficient Visual Backbones"](https://arxiv.org/pdf/2103.12731.pdf) 


#### 20.2. Overview

![](./model/img/HaloNet.png)

#### 20.3. Usage Code
```python

from model.attention.HaloAttention import HaloAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,512,8,8)
halo = HaloAttention(dim=512,
    block_size=2,
    halo_size=1,)
output=halo(input)
print(output.shape)

```


***

### 21. Polarized Self-Attention Usage

#### 21.1. Paper

[Polarized Self-Attention: Towards High-quality Pixel-wise Regression"](https://arxiv.org/abs/2107.00782)  


#### 21.2. Overview

![](./model/img/PoSA.png)

#### 21.3. Usage Code
```python

from model.attention.PolarizedSelfAttention import ParallelPolarizedSelfAttention,SequentialPolarizedSelfAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,512,7,7)
psa = SequentialPolarizedSelfAttention(channel=512)
output=psa(input)
print(output.shape)


```


***


### 22. CoTAttention Usage

#### 22.1. Paper

[Contextual Transformer Networks for Visual Recognition---arXiv 2021.07.26](https://arxiv.org/abs/2107.12292) 


#### 22.2. Overview

![](./model/img/CoT.png)

#### 22.3. Usage Code
```python

from model.attention.CoTAttention import CoTAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
cot = CoTAttention(dim=512,kernel_size=3)
output=cot(input)
print(output.shape)



```

***


### 23. Residual Attention Usage

#### 23.1. Paper

[Residual Attention: A Simple but Effective Method for Multi-Label Recognition---ICCV2021](https://arxiv.org/abs/2108.02456) 


#### 23.2. Overview

![](./model/img/ResAtt.png)

#### 23.3. Usage Code
```python

from model.attention.ResidualAttention import ResidualAttention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
resatt = ResidualAttention(channel=512,num_class=1000,la=0.2)
output=resatt(input)
print(output.shape)



```

***



### 24. S2 Attention Usage

#### 24.1. Paper

[S²-MLPv2: Improved Spatial-Shift MLP Architecture for Vision---arXiv 2021.08.02](https://arxiv.org/abs/2108.01072) 


#### 24.2. Overview

![](./model/img/S2Attention.png)

#### 24.3. Usage Code
```python
from model.attention.S2Attention import S2Attention
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(50,512,7,7)
s2att = S2Attention(channels=512)
output=s2att(input)
print(output.shape)

```

***



### 25. GFNet Attention Usage

#### 25.1. Paper

[Global Filter Networks for Image Classification---arXiv 2021.07.01](https://arxiv.org/abs/2107.00645) 


#### 25.2. Overview

![](./model/img/GFNet.jpg)

#### 25.3. Usage Code - Implemented by [Wenliang Zhao (Author)](https://scholar.google.com/citations?user=lyPWvuEAAAAJ&hl=en)

```python
from model.attention.gfnet import GFNet
import torch
from torch import nn
from torch.nn import functional as F

x = torch.randn(1, 3, 224, 224)
gfnet = GFNet(embed_dim=384, img_size=224, patch_size=16, num_classes=1000)
out = gfnet(x)
print(out.shape)

```

***


### 26. TripletAttention Usage

#### 26.1. Paper

[Rotate to Attend: Convolutional Triplet Attention Module---CVPR 2021](https://arxiv.org/abs/2010.03045) 

#### 26.2. Overview

![](./model/img/triplet.png)

#### 26.3. Usage Code - Implemented by [digantamisra98](https://github.com/digantamisra98)

```python
from model.attention.TripletAttention import TripletAttention
import torch
from torch import nn
from torch.nn import functional as F
input=torch.randn(50,512,7,7)
triplet = TripletAttention()
output=triplet(input)
print(output.shape)
```


***


### 27. Coordinate Attention Usage

#### 27.1. Paper

[Coordinate Attention for Efficient Mobile Network Design---CVPR 2021](https://arxiv.org/abs/2103.02907)


#### 27.2. Overview

![](./model/img/CoordAttention.png)

#### 27.3. Usage Code - Implemented by [Andrew-Qibin](https://github.com/Andrew-Qibin)

```python
from model.attention.CoordAttention import CoordAtt
import torch
from torch import nn
from torch.nn import functional as F

inp=torch.rand([2, 96, 56, 56])
inp_dim, oup_dim = 96, 96
reduction=32

coord_attention = CoordAtt(inp_dim, oup_dim, reduction=reduction)
output=coord_attention(inp)
print(output.shape)
```

***


### 28. MobileViT Attention Usage

#### 28.1. Paper

[MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer---ArXiv 2021.10.05](https://arxiv.org/abs/2103.02907)


#### 28.2. Overview

![](./model/img/MobileViTAttention.png)

#### 28.3. Usage Code

```python
from model.attention.MobileViTAttention import MobileViTAttention
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    m=MobileViTAttention()
    input=torch.randn(1,3,49,49)
    output=m(input)
    print(output.shape)  #output:(1,3,49,49)
    
```

***


### 29. ParNet Attention Usage

#### 29.1. Paper

[Non-deep Networks---ArXiv 2021.10.20](https://arxiv.org/abs/2110.07641)


#### 29.2. Overview

![](./model/img/ParNet.png)

#### 29.3. Usage Code

```python
from model.attention.ParNetAttention import *
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(50,512,7,7)
    pna = ParNetAttention(channel=512)
    output=pna(input)
    print(output.shape) #50,512,7,7
    
```

***


### 30. UFO Attention Usage

#### 30.1. Paper

[UFO-ViT: High Performance Linear Vision Transformer without Softmax---ArXiv 2021.09.29](https://arxiv.org/abs/2110.07641)


#### 30.2. Overview

![](./model/img/UFO.png)

#### 30.3. Usage Code

```python
from model.attention.UFOAttention import *
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(50,49,512)
    ufo = UFOAttention(d_model=512, d_k=512, d_v=512, h=8)
    output=ufo(input,input,input)
    print(output.shape) #[50, 49, 512]
    
```

-

### 31. ACmix Attention Usage

#### 31.1. Paper

[On the Integration of Self-Attention and Convolution](https://arxiv.org/pdf/2111.14556.pdf)

#### 31.2. Usage Code

```python
from model.attention.ACmix import ACmix
import torch

if __name__ == '__main__':
    input=torch.randn(50,256,7,7)
    acmix = ACmix(in_planes=256, out_planes=256)
    output=acmix(input)
    print(output.shape)
    
```

### 32. MobileViTv2 Attention Usage

#### 32.1. Paper

[Separable Self-attention for Mobile Vision Transformers---ArXiv 2022.06.06](https://arxiv.org/abs/2206.02680)


#### 32.2. Overview

![](./model/img/MobileViTv2.png)

#### 32.3. Usage Code

```python
from model.attention.MobileViTv2Attention import MobileViTv2Attention
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(50,49,512)
    sa = MobileViTv2Attention(d_model=512)
    output=sa(input)
    print(output.shape)
    
```

### 33. DAT Attention Usage

#### 33.1. Paper

[Vision Transformer with Deformable Attention---CVPR2022](https://arxiv.org/abs/2201.00520)

#### 33.2. Usage Code

```python
from model.attention.DAT import DAT
import torch

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = DAT(
        img_size=224,
        patch_size=4,
        num_classes=1000,
        expansion=4,
        dim_stem=96,
        dims=[96, 192, 384, 768],
        depths=[2, 2, 6, 2],
        stage_spec=[['L', 'S'], ['L', 'S'], ['L', 'D', 'L', 'D', 'L', 'D'], ['L', 'D']],
        heads=[3, 6, 12, 24],
        window_sizes=[7, 7, 7, 7] ,
        groups=[-1, -1, 3, 6],
        use_pes=[False, False, True, True],
        dwc_pes=[False, False, False, False],
        strides=[-1, -1, 1, 1],
        sr_ratios=[-1, -1, -1, -1],
        offset_range_factor=[-1, -1, 2, 2],
        no_offs=[False, False, False, False],
        fixed_pes=[False, False, False, False],
        use_dwc_mlps=[False, False, False, False],
        use_conv_patches=False,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.2,
    )
    output=model(input)
    print(output[0].shape)
    
```

### 34. CrossFormer Attention Usage

#### 34.1. Paper

[CROSSFORMER: A VERSATILE VISION TRANSFORMER HINGING ON CROSS-SCALE ATTENTION---ICLR 2022](https://arxiv.org/pdf/2108.00154.pdf)

#### 34.2. Usage Code

```python
from model.attention.Crossformer import CrossFormer
import torch

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = CrossFormer(img_size=224,
        patch_size=[4, 8, 16, 32],
        in_chans= 3,
        num_classes=1000,
        embed_dim=48,
        depths=[2, 2, 6, 2],
        num_heads=[3, 6, 12, 24],
        group_size=[7, 7, 7, 7],
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False,
        merge_size=[[2, 4], [2,4], [2, 4]]
    )
    output=model(input)
    print(output.shape)
    
```

### 35. MOATransformer Attention Usage

#### 35.1. Paper

[Aggregating Global Features into Local Vision Transformer](https://arxiv.org/abs/2201.12903)

#### 35.2. Usage Code

```python
from model.attention.MOATransformer import MOATransformer
import torch

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = MOATransformer(
        img_size=224,
        patch_size=4,
        in_chans=3,
        num_classes=1000,
        embed_dim=96,
        depths=[2, 2, 6],
        num_heads=[3, 6, 12],
        window_size=14,
        mlp_ratio=4.,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        drop_path_rate=0.1,
        ape=False,
        patch_norm=True,
        use_checkpoint=False
    )
    output=model(input)
    print(output.shape)
    
```

### 36. CrissCrossAttention Attention Usage

#### 36.1. Paper

[CCNet: Criss-Cross Attention for Semantic Segmentation](https://arxiv.org/abs/1811.11721)

#### 36.2. Usage Code

```python
from model.attention.CrissCrossAttention import CrissCrossAttention
import torch

if __name__ == '__main__':
    input=torch.randn(3, 64, 7, 7)
    model = CrissCrossAttention(64)
    outputs = model(input)
    print(outputs.shape)
    
```

### 37. Axial_attention Attention Usage

#### 37.1. Paper

[Axial Attention in Multidimensional Transformers](https://arxiv.org/abs/1912.12180)

#### 37.2. Usage Code

```python
from model.attention.Axial_attention import AxialImageTransformer
import torch

if __name__ == '__main__':
    input=torch.randn(3, 128, 7, 7)
    model = AxialImageTransformer(
        dim = 128,
        depth = 12,
        reversible = True
    )
    outputs = model(input)
    print(outputs.shape)
    
```

***


# Backbone Series

- Pytorch implementation of ["Deep Residual Learning for Image Recognition---CVPR2016 Best Paper"](https://arxiv.org/pdf/1512.03385.pdf)

- Pytorch implementation of ["Aggregated Residual Transformations for Deep Neural Networks---CVPR2017"](https://arxiv.org/abs/1611.05431v2)

- Pytorch implementation of [MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer---ArXiv 2020.10.05](https://arxiv.org/abs/2103.02907)

- Pytorch implementation of [Patches Are All You Need?---ICLR2022 (Under Review)](https://openreview.net/forum?id=TVHS5Y4dNvM)

- Pytorch implementation of [Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer---ArXiv 2021.06.07](https://arxiv.org/abs/2106.03650)

- Pytorch implementation of [ConTNet: Why not use convolution and transformer at the same time?---ArXiv 2021.04.27](https://arxiv.org/abs/2104.13497)

- Pytorch implementation of [Vision Transformers with Hierarchical Attention---ArXiv 2022.06.15](https://arxiv.org/abs/2106.03180)

- Pytorch implementation of [Co-Scale Conv-Attentional Image Transformers---ArXiv 2021.08.26](https://arxiv.org/abs/2104.06399)

- Pytorch implementation of [Conditional Positional Encodings for Vision Transformers](https://arxiv.org/abs/2102.10882)

- Pytorch implementation of [Rethinking Spatial Dimensions of Vision Transformers---ICCV 2021](https://arxiv.org/abs/2103.16302)

- Pytorch implementation of [CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification---ICCV 2021](https://arxiv.org/abs/2103.14899)

- Pytorch implementation of [Transformer in Transformer---NeurIPS 2021](https://arxiv.org/abs/2103.00112)

- Pytorch implementation of [DeepViT: Towards Deeper Vision Transformer](https://arxiv.org/abs/2103.11886)

- Pytorch implementation of [Incorporating Convolution Designs into Visual Transformers](https://arxiv.org/abs/2103.11816)
***

- Pytorch implementation of [ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases](https://arxiv.org/abs/2103.10697)

- Pytorch implementation of [Augmenting Convolutional networks with attention-based aggregation](https://arxiv.org/abs/2112.13692)

- Pytorch implementation of [Going deeper with Image Transformers---ICCV 2021 (Oral)](https://arxiv.org/abs/2103.17239)

- Pytorch implementation of [Training data-efficient image transformers & distillation through attention---ICML 2021](https://arxiv.org/abs/2012.12877)

- Pytorch implementation of [LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference](https://arxiv.org/abs/2104.01136)

- Pytorch implementation of [VOLO: Vision Outlooker for Visual Recognition](https://arxiv.org/abs/2106.13112)

- Pytorch implementation of [Container: Context Aggregation Network---NeuIPS 2021](https://arxiv.org/abs/2106.01401)

- Pytorch implementation of [CMT: Convolutional Neural Networks Meet Vision Transformers---CVPR 2022](https://arxiv.org/abs/2107.06263)

- Pytorch implementation of [Vision Transformer with Deformable Attention---CVPR 2022](https://arxiv.org/abs/2201.00520)

- Pytorch implementation of [EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)

- Pytorch implementation of [ConvNeXtV2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)


### 1. ResNet Usage
#### 1.1. Paper
["Deep Residual Learning for Image Recognition---CVPR2016 Best Paper"](https://arxiv.org/pdf/1512.03385.pdf)

#### 1.2. Overview
![](./model/img/resnet.png)
![](./model/img/resnet2.jpg)

#### 1.3. Usage Code
```python

from model.backbone.resnet import ResNet50,ResNet101,ResNet152
import torch
if __name__ == '__main__':
    input=torch.randn(50,3,224,224)
    resnet50=ResNet50(1000)
    # resnet101=ResNet101(1000)
    # resnet152=ResNet152(1000)
    out=resnet50(input)
    print(out.shape)

```


### 2. ResNeXt Usage
#### 2.1. Paper

["Aggregated Residual Transformations for Deep Neural Networks---CVPR2017"](https://arxiv.org/abs/1611.05431v2)

#### 2.2. Overview
![](./model/img/resnext.png)

#### 2.3. Usage Code
```python

from model.backbone.resnext import ResNeXt50,ResNeXt101,ResNeXt152
import torch

if __name__ == '__main__':
    input=torch.randn(50,3,224,224)
    resnext50=ResNeXt50(1000)
    # resnext101=ResNeXt101(1000)
    # resnext152=ResNeXt152(1000)
    out=resnext50(input)
    print(out.shape)


```



### 3. MobileViT Usage
#### 3.1. Paper

[MobileViT: Light-weight, General-purpose, and Mobile-friendly Vision Transformer---ArXiv 2020.10.05](https://arxiv.org/abs/2103.02907)

#### 3.2. Overview
![](./model/img/mobileViT.jpg)

#### 3.3. Usage Code
```python

from model.backbone.MobileViT import *
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)

    ### mobilevit_xxs
    mvit_xxs=mobilevit_xxs()
    out=mvit_xxs(input)
    print(out.shape)

    ### mobilevit_xs
    mvit_xs=mobilevit_xs()
    out=mvit_xs(input)
    print(out.shape)


    ### mobilevit_s
    mvit_s=mobilevit_s()
    out=mvit_s(input)
    print(out.shape)

```





### 4. ConvMixer Usage
#### 4.1. Paper
[Patches Are All You Need?---ICLR2022 (Under Review)](https://openreview.net/forum?id=TVHS5Y4dNvM)
#### 4.2. Overview
![](./model/img/ConvMixer.png)

#### 4.3. Usage Code
```python

from model.backbone.ConvMixer import *
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    x=torch.randn(1,3,224,224)
    convmixer=ConvMixer(dim=512,depth=12)
    out=convmixer(x)
    print(out.shape)  #[1, 1000]


```

### 5. ShuffleTransformer Usage
#### 5.1. Paper
[Shuffle Transformer: Rethinking Spatial Shuffle for Vision Transformer](https://arxiv.org/pdf/2106.03650.pdf)

#### 5.2. Usage Code
```python

from model.backbone.ShuffleTransformer import ShuffleTransformer
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    sft = ShuffleTransformer()
    output=sft(input)
    print(output.shape)


```

### 6. ConTNet Usage
#### 6.1. Paper
[ConTNet: Why not use convolution and transformer at the same time?](https://arxiv.org/abs/2104.13497)

#### 6.2. Usage Code
```python

from model.backbone.ConTNet import ConTNet
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == "__main__":
    model = build_model(use_avgdown=True, relative=True, qkv_bias=True, pre_norm=True)
    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)


```

### 7 HATNet Usage
#### 7.1. Paper
[Vision Transformers with Hierarchical Attention](https://arxiv.org/abs/2106.03180)

#### 7.2. Usage Code
```python

from model.backbone.HATNet import HATNet
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    hat = HATNet(dims=[48, 96, 240, 384], head_dim=48, expansions=[8, 8, 4, 4],
        grid_sizes=[8, 7, 7, 1], ds_ratios=[8, 4, 2, 1], depths=[2, 2, 6, 3])
    output=hat(input)
    print(output.shape)


```

### 8 CoaT Usage
#### 8.1. Paper
[Co-Scale Conv-Attentional Image Transformers](https://arxiv.org/abs/2104.06399)

#### 8.2. Usage Code
```python

from model.backbone.CoaT import CoaT
import torch
from torch import nn

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = CoaT(patch_size=4, embed_dims=[152, 152, 152, 152], serial_depths=[2, 2, 2, 2], parallel_depth=6, num_heads=8, mlp_ratios=[4, 4, 4, 4])
    output=model(input)
    print(output.shape) # torch.Size([1, 1000])

```

### 9 PVT Usage
#### 9.1. Paper
[PVT v2: Improved Baselines with Pyramid Vision Transformer](https://arxiv.org/pdf/2106.13797.pdf)

#### 9.2. Usage Code
```python

from model.backbone.PVT import PyramidVisionTransformer
import torch
from torch import nn

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = PyramidVisionTransformer(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[2, 2, 2, 2], sr_ratios=[8, 4, 2, 1])
    output=model(input)
    print(output.shape)

```


### 10 CPVT Usage
#### 10.1. Paper
[Conditional Positional Encodings for Vision Transformers](https://arxiv.org/abs/2102.10882)

#### 10.2. Usage Code
```python

from model.backbone.CPVT import CPVTV2
import torch
from torch import nn

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = CPVTV2(
        patch_size=4, embed_dims=[64, 128, 320, 512], num_heads=[1, 2, 5, 8], mlp_ratios=[8, 8, 4, 4], qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), depths=[3, 4, 6, 3], sr_ratios=[8, 4, 2, 1])
    output=model(input)
    print(output.shape)

```

### 11 PIT Usage
#### 11.1. Paper
[Rethinking Spatial Dimensions of Vision Transformers](https://arxiv.org/abs/2103.16302)

#### 11.2. Usage Code
```python

from model.backbone.PIT import PoolingTransformer
import torch
from torch import nn

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = PoolingTransformer(
        image_size=224,
        patch_size=14,
        stride=7,
        base_dims=[64, 64, 64],
        depth=[3, 6, 4],
        heads=[4, 8, 16],
        mlp_ratio=4
    )
    output=model(input)
    print(output.shape)

```

### 12 CrossViT Usage
#### 12.1. Paper
[CrossViT: Cross-Attention Multi-Scale Vision Transformer for Image Classification](https://arxiv.org/abs/2103.14899)

#### 12.2. Usage Code
```python

from model.backbone.CrossViT import VisionTransformer
import torch
from torch import nn

if __name__ == "__main__":
    input=torch.randn(1,3,224,224)
    model = VisionTransformer(
        img_size=[240, 224],
        patch_size=[12, 16], 
        embed_dim=[192, 384], 
        depth=[[1, 4, 0], [1, 4, 0], [1, 4, 0]],
        num_heads=[6, 6], 
        mlp_ratio=[4, 4, 1], 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
    )
    output=model(input)
    print(output.shape)

```

### 13 TnT Usage
#### 13.1. Paper
[Transformer in Transformer](https://arxiv.org/abs/2103.00112)

#### 13.2. Usage Code
```python

from model.backbone.TnT import TNT
import torch
from torch import nn

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = TNT(
        img_size=224, 
        patch_size=16, 
        outer_dim=384, 
        inner_dim=24, 
        depth=12,
        outer_num_heads=6, 
        inner_num_heads=4, 
        qkv_bias=False,
        inner_stride=4)
    output=model(input)
    print(output.shape)

```

### 14 DViT Usage
#### 14.1. Paper
[DeepViT: Towards Deeper Vision Transformer](https://arxiv.org/abs/2103.11886)

#### 14.2. Usage Code
```python

from model.backbone.DViT import DeepVisionTransformer
import torch
from torch import nn

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = DeepVisionTransformer(
        patch_size=16, embed_dim=384, 
        depth=[False] * 16, 
        apply_transform=[False] * 0 + [True] * 32, 
        num_heads=12, 
        mlp_ratio=3, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        )
    output=model(input)
    print(output.shape)

```

### 15 CeiT Usage
#### 15.1. Paper
[Incorporating Convolution Designs into Visual Transformers](https://arxiv.org/abs/2103.11816)

#### 15.2. Usage Code
```python

from model.backbone.CeiT import CeIT
import torch
from torch import nn

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = CeIT(
        hybrid_backbone=Image2Tokens(),
        patch_size=4, 
        embed_dim=192, 
        depth=12, 
        num_heads=3, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    output=model(input)
    print(output.shape)

```

### 16 ConViT Usage
#### 16.1. Paper
[ConViT: Improving Vision Transformers with Soft Convolutional Inductive Biases](https://arxiv.org/abs/2103.10697)

#### 16.2. Usage Code
```python

from model.backbone.ConViT import VisionTransformer
import torch
from torch import nn

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = VisionTransformer(
        num_heads=16,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    output=model(input)
    print(output.shape)

```

### 17 CaiT Usage
#### 17.1. Paper
[Going deeper with Image Transformers](https://arxiv.org/abs/2103.17239)

#### 17.2. Usage Code
```python

from model.backbone.CaiT import CaiT
import torch
from torch import nn

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = CaiT(
        img_size= 224,
        patch_size=16, 
        embed_dim=192, 
        depth=24, 
        num_heads=4, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        init_scale=1e-5,
        depth_token_only=2
        )
    output=model(input)
    print(output.shape)

```

### 18 PatchConvnet Usage
#### 18.1. Paper
[Augmenting Convolutional networks with attention-based aggregation](https://arxiv.org/abs/2112.13692)

#### 18.2. Usage Code
```python

from model.backbone.PatchConvnet import PatchConvnet
import torch
from torch import nn

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = PatchConvnet(
        patch_size=16,
        embed_dim=384,
        depth=60,
        num_heads=1,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        Patch_layer=ConvStem,
        Attention_block=Conv_blocks_se,
        depth_token_only=1,
        mlp_ratio_clstk=3.0,
    )
    output=model(input)
    print(output.shape)

```

### 19 DeiT Usage
#### 19.1. Paper
[Training data-efficient image transformers & distillation through attention](https://arxiv.org/abs/2012.12877)

#### 19.2. Usage Code
```python

from model.backbone.DeiT import DistilledVisionTransformer
import torch
from torch import nn

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = DistilledVisionTransformer(
        patch_size=16, 
        embed_dim=384, 
        depth=12, 
        num_heads=6, 
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6)
        )
    output=model(input)
    print(output[0].shape)

```

### 20 LeViT Usage
#### 20.1. Paper
[LeViT: a Vision Transformer in ConvNet’s Clothing for Faster Inference](https://arxiv.org/abs/2104.01136)

#### 20.2. Usage Code
```python

from model.backbone.LeViT import *
import torch
from torch import nn

if __name__ == '__main__':
    for name in specification:
        input=torch.randn(1,3,224,224)
        model = globals()[name](fuse=True, pretrained=False)
        model.eval()
        output = model(input)
        print(output.shape)

```

### 21 VOLO Usage
#### 21.1. Paper
[VOLO: Vision Outlooker for Visual Recognition](https://arxiv.org/abs/2106.13112)

#### 21.2. Usage Code
```python

from model.backbone.VOLO import VOLO
import torch
from torch import nn

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = VOLO([4, 4, 8, 2],
                 embed_dims=[192, 384, 384, 384],
                 num_heads=[6, 12, 12, 12],
                 mlp_ratios=[3, 3, 3, 3],
                 downsamples=[True, False, False, False],
                 outlook_attention=[True, False, False, False ],
                 post_layers=['ca', 'ca'],
                 )
    output=model(input)
    print(output[0].shape)

```

### 22 Container Usage
#### 22.1. Paper
[Container: Context Aggregation Network](https://arxiv.org/abs/2106.01401)

#### 22.2. Usage Code
```python

from model.backbone.Container import VisionTransformer
import torch
from torch import nn

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = VisionTransformer(
        img_size=[224, 56, 28, 14], 
        patch_size=[4, 2, 2, 2], 
        embed_dim=[64, 128, 320, 512], 
        depth=[3, 4, 8, 3], 
        num_heads=16, 
        mlp_ratio=[8, 8, 4, 4], 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6))
    output=model(input)
    print(output.shape)

```

### 23 CMT Usage
#### 23.1. Paper
[CMT: Convolutional Neural Networks Meet Vision Transformers](https://arxiv.org/abs/2107.06263)

#### 23.2. Usage Code
```python

from model.backbone.CMT import CMT_Tiny
import torch
from torch import nn

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = CMT_Tiny()
    output=model(input)
    print(output[0].shape)

```

### 24 EfficientFormer Usage
#### 24.1. Paper
[EfficientFormer: Vision Transformers at MobileNet Speed](https://arxiv.org/abs/2206.01191)

#### 24.2. Usage Code
```python

from model.backbone.EfficientFormer import EfficientFormer
import torch
from torch import nn

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = EfficientFormer(
        layers=EfficientFormer_depth['l1'],
        embed_dims=EfficientFormer_width['l1'],
        downsamples=[True, True, True, True],
        vit_num=1,
    )
    output=model(input)
    print(output[0].shape)

```

### 25 ConvNeXtV2 Usage
#### 25.1. Paper
[ConvNeXtV2: Co-designing and Scaling ConvNets with Masked Autoencoders](https://arxiv.org/abs/2301.00808)

#### 25.2. Usage Code
```python

from model.backbone.convnextv2 import convnextv2_atto
import torch
from torch import nn

if __name__ == "__main__":
    model = convnextv2_atto()
    input = torch.randn(1, 3, 224, 224)
    out = model(input)
    print(out.shape)

```





# MLP Series

- Pytorch implementation of ["RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition---arXiv 2021.05.05"](https://arxiv.org/pdf/2105.01883v1.pdf)

- Pytorch implementation of ["MLP-Mixer: An all-MLP Architecture for Vision---arXiv 2021.05.17"](https://arxiv.org/pdf/2105.01601.pdf)

- Pytorch implementation of ["ResMLP: Feedforward networks for image classification with data-efficient training---arXiv 2021.05.07"](https://arxiv.org/pdf/2105.03404.pdf)

- Pytorch implementation of ["Pay Attention to MLPs---arXiv 2021.05.17"](https://arxiv.org/abs/2105.08050)


- Pytorch implementation of ["Sparse MLP for Image Recognition: Is Self-Attention Really Necessary?---arXiv 2021.09.12"](https://arxiv.org/abs/2109.05422)

### 1. RepMLP Usage
#### 1.1. Paper
["RepMLP: Re-parameterizing Convolutions into Fully-connected Layers for Image Recognition"](https://arxiv.org/pdf/2105.01883v1.pdf)

#### 1.2. Overview
![](./model/img/repmlp.png)

#### 1.3. Usage Code
```python
from model.mlp.repmlp import RepMLP
import torch
from torch import nn

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
```

### 2. MLP-Mixer Usage
#### 2.1. Paper
["MLP-Mixer: An all-MLP Architecture for Vision"](https://arxiv.org/pdf/2105.01601.pdf)

#### 2.2. Overview
![](./model/img/mlpmixer.png)

#### 2.3. Usage Code
```python
from model.mlp.mlp_mixer import MlpMixer
import torch
mlp_mixer=MlpMixer(num_classes=1000,num_blocks=10,patch_size=10,tokens_hidden_dim=32,channels_hidden_dim=1024,tokens_mlp_dim=16,channels_mlp_dim=1024)
input=torch.randn(50,3,40,40)
output=mlp_mixer(input)
print(output.shape)
```

***

### 3. ResMLP Usage
#### 3.1. Paper
["ResMLP: Feedforward networks for image classification with data-efficient training"](https://arxiv.org/pdf/2105.03404.pdf)

#### 3.2. Overview
![](./model/img/resmlp.png)

#### 3.3. Usage Code
```python
from model.mlp.resmlp import ResMLP
import torch

input=torch.randn(50,3,14,14)
resmlp=ResMLP(dim=128,image_size=14,patch_size=7,class_num=1000)
out=resmlp(input)
print(out.shape) #the last dimention is class_num
```

***

### 4. gMLP Usage
#### 4.1. Paper
["Pay Attention to MLPs"](https://arxiv.org/abs/2105.08050)

#### 4.2. Overview
![](./model/img/gMLP.jpg)

#### 4.3. Usage Code
```python
from model.mlp.g_mlp import gMLP
import torch

num_tokens=10000
bs=50
len_sen=49
num_layers=6
input=torch.randint(num_tokens,(bs,len_sen)) #bs,len_sen
gmlp = gMLP(num_tokens=num_tokens,len_sen=len_sen,dim=512,d_ff=1024)
output=gmlp(input)
print(output.shape)
```

***

### 5. sMLP Usage
#### 5.1. Paper
["Sparse MLP for Image Recognition: Is Self-Attention Really Necessary?"](https://arxiv.org/abs/2109.05422)

#### 5.2. Overview
![](./model/img/sMLP.jpg)

#### 5.3. Usage Code
```python
from model.mlp.sMLP_block import sMLPBlock
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(50,3,224,224)
    smlp=sMLPBlock(h=224,w=224)
    out=smlp(input)
    print(out.shape)
```

### 6. vip-mlp Usage
#### 6.1. Paper
["Vision Permutator: A Permutable MLP-Like Architecture for Visual Recognition"](https://arxiv.org/abs/2106.12368)

#### 6.2. Usage Code
```python
from model.mlp.vip-mlp import VisionPermutator
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(1,3,224,224)
    model = VisionPermutator(
        layers=[4, 3, 8, 3], 
        embed_dims=[384, 384, 384, 384], 
        patch_size=14, 
        transitions=[False, False, False, False],
        segment_dim=[16, 16, 16, 16], 
        mlp_ratios=[3, 3, 3, 3], 
        mlp_fn=WeightedPermuteMLP
    )
    output=model(input)
    print(output.shape)
```


# Re-Parameter Series

- Pytorch implementation of ["RepVGG: Making VGG-style ConvNets Great Again---CVPR2021"](https://arxiv.org/abs/2101.03697)

- Pytorch implementation of ["ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks---ICCV2019"](https://arxiv.org/abs/1908.03930)

- Pytorch implementation of ["Diverse Branch Block: Building a Convolution as an Inception-like Unit---CVPR2021"](https://arxiv.org/abs/2103.13425)


***

### 1. RepVGG Usage
#### 1.1. Paper
["RepVGG: Making VGG-style ConvNets Great Again"](https://arxiv.org/abs/2101.03697)

#### 1.2. Overview
![](./model/img/repvgg.png)

#### 1.3. Usage Code
```python

from model.rep.repvgg import RepBlock
import torch


input=torch.randn(50,512,49,49)
repblock=RepBlock(512,512)
repblock.eval()
out=repblock(input)
repblock._switch_to_deploy()
out2=repblock(input)
print('difference between vgg and repvgg')
print(((out2-out)**2).sum())
```



***

### 2. ACNet Usage
#### 2.1. Paper
["ACNet: Strengthening the Kernel Skeletons for Powerful CNN via Asymmetric Convolution Blocks"](https://arxiv.org/abs/1908.03930)

#### 2.2. Overview
![](./model/img/acnet.png)

#### 2.3. Usage Code
```python
from model.rep.acnet import ACNet
import torch
from torch import nn

input=torch.randn(50,512,49,49)
acnet=ACNet(512,512)
acnet.eval()
out=acnet(input)
acnet._switch_to_deploy()
out2=acnet(input)
print('difference:')
print(((out2-out)**2).sum())

```



***

### 2. Diverse Branch Block Usage
#### 2.1. Paper
["Diverse Branch Block: Building a Convolution as an Inception-like Unit"](https://arxiv.org/abs/2103.13425)

#### 2.2. Overview
![](./model/img/ddb.png)

#### 2.3. Usage Code
##### 2.3.1 Transform I
```python
from model.rep.ddb import transI_conv_bn
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,64,7,7)
#conv+bn
conv1=nn.Conv2d(64,64,3,padding=1)
bn1=nn.BatchNorm2d(64)
bn1.eval()
out1=bn1(conv1(input))

#conv_fuse
conv_fuse=nn.Conv2d(64,64,3,padding=1)
conv_fuse.weight.data,conv_fuse.bias.data=transI_conv_bn(conv1,bn1)
out2=conv_fuse(input)

print("difference:",((out2-out1)**2).sum().item())
```

##### 2.3.2 Transform II
```python
from model.rep.ddb import transII_conv_branch
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,64,7,7)

#conv+conv
conv1=nn.Conv2d(64,64,3,padding=1)
conv2=nn.Conv2d(64,64,3,padding=1)
out1=conv1(input)+conv2(input)

#conv_fuse
conv_fuse=nn.Conv2d(64,64,3,padding=1)
conv_fuse.weight.data,conv_fuse.bias.data=transII_conv_branch(conv1,conv2)
out2=conv_fuse(input)

print("difference:",((out2-out1)**2).sum().item())
```

##### 2.3.3 Transform III
```python
from model.rep.ddb import transIII_conv_sequential
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,64,7,7)

#conv+conv
conv1=nn.Conv2d(64,64,1,padding=0,bias=False)
conv2=nn.Conv2d(64,64,3,padding=1,bias=False)
out1=conv2(conv1(input))


#conv_fuse
conv_fuse=nn.Conv2d(64,64,3,padding=1,bias=False)
conv_fuse.weight.data=transIII_conv_sequential(conv1,conv2)
out2=conv_fuse(input)

print("difference:",((out2-out1)**2).sum().item())
```

##### 2.3.4 Transform IV
```python
from model.rep.ddb import transIV_conv_concat
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,64,7,7)

#conv+conv
conv1=nn.Conv2d(64,32,3,padding=1)
conv2=nn.Conv2d(64,32,3,padding=1)
out1=torch.cat([conv1(input),conv2(input)],dim=1)

#conv_fuse
conv_fuse=nn.Conv2d(64,64,3,padding=1)
conv_fuse.weight.data,conv_fuse.bias.data=transIV_conv_concat(conv1,conv2)
out2=conv_fuse(input)

print("difference:",((out2-out1)**2).sum().item())
```

##### 2.3.5 Transform V
```python
from model.rep.ddb import transV_avg
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,64,7,7)

avg=nn.AvgPool2d(kernel_size=3,stride=1)
out1=avg(input)

conv=transV_avg(64,3)
out2=conv(input)

print("difference:",((out2-out1)**2).sum().item())
```


##### 2.3.6 Transform VI
```python
from model.rep.ddb import transVI_conv_scale
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,64,7,7)

#conv+conv
conv1x1=nn.Conv2d(64,64,1)
conv1x3=nn.Conv2d(64,64,(1,3),padding=(0,1))
conv3x1=nn.Conv2d(64,64,(3,1),padding=(1,0))
out1=conv1x1(input)+conv1x3(input)+conv3x1(input)

#conv_fuse
conv_fuse=nn.Conv2d(64,64,3,padding=1)
conv_fuse.weight.data,conv_fuse.bias.data=transVI_conv_scale(conv1x1,conv1x3,conv3x1)
out2=conv_fuse(input)

print("difference:",((out2-out1)**2).sum().item())
```





# Convolution Series

- Pytorch implementation of ["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications---CVPR2017"](https://arxiv.org/abs/1704.04861)

- Pytorch implementation of ["Efficientnet: Rethinking model scaling for convolutional neural networks---PMLR2019"](http://proceedings.mlr.press/v97/tan19a.html)

- Pytorch implementation of ["Involution: Inverting the Inherence of Convolution for Visual Recognition---CVPR2021"](https://arxiv.org/abs/2103.06255)

- Pytorch implementation of ["Dynamic Convolution: Attention over Convolution Kernels---CVPR2020 Oral"](https://arxiv.org/abs/1912.03458)

- Pytorch implementation of ["CondConv: Conditionally Parameterized Convolutions for Efficient Inference---NeurIPS2019"](https://arxiv.org/abs/1904.04971)

***

### 1. Depthwise Separable Convolution Usage
#### 1.1. Paper
["MobileNets: Efficient Convolutional Neural Networks for Mobile Vision Applications"](https://arxiv.org/abs/1704.04861)

#### 1.2. Overview
![](./model/img/DepthwiseSeparableConv.png)

#### 1.3. Usage Code
```python
from model.conv.DepthwiseSeparableConvolution import DepthwiseSeparableConvolution
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,3,224,224)
dsconv=DepthwiseSeparableConvolution(3,64)
out=dsconv(input)
print(out.shape)
```

***


### 2. MBConv Usage
#### 2.1. Paper
["Efficientnet: Rethinking model scaling for convolutional neural networks"](http://proceedings.mlr.press/v97/tan19a.html)

#### 2.2. Overview
![](./model/img/MBConv.jpg)

#### 2.3. Usage Code
```python
from model.conv.MBConv import MBConvBlock
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,3,224,224)
mbconv=MBConvBlock(ksize=3,input_filters=3,output_filters=512,image_size=224)
out=mbconv(input)
print(out.shape)


```

***


### 3. Involution Usage
#### 3.1. Paper
["Involution: Inverting the Inherence of Convolution for Visual Recognition"](https://arxiv.org/abs/2103.06255)

#### 3.2. Overview
![](./model/img/Involution.png)

#### 3.3. Usage Code
```python
from model.conv.Involution import Involution
import torch
from torch import nn
from torch.nn import functional as F

input=torch.randn(1,4,64,64)
involution=Involution(kernel_size=3,in_channel=4,stride=2)
out=involution(input)
print(out.shape)
```

***


### 4. DynamicConv Usage
#### 4.1. Paper
["Dynamic Convolution: Attention over Convolution Kernels"](https://arxiv.org/abs/1912.03458)

#### 4.2. Overview
![](./model/img/DynamicConv.png)

#### 4.3. Usage Code
```python
from model.conv.DynamicConv import *
import torch
from torch import nn
from torch.nn import functional as F

if __name__ == '__main__':
    input=torch.randn(2,32,64,64)
    m=DynamicConv(in_planes=32,out_planes=64,kernel_size=3,stride=1,padding=1,bias=False)
    out=m(input)
    print(out.shape) # 2,32,64,64

```

***


### 5. CondConv Usage
#### 5.1. Paper
["CondConv: Conditionally Parameterized Convolutions for Efficient Inference"](https://arxiv.org/abs/1904.04971)

#### 5.2. Overview
![](./model/img/CondConv.png)

#### 5.3. Usage Code
```python
from model.conv.CondConv import *
import torch
from torch import nn
from torch.nn import functional as F





if __name__ == '__main__':
    input=torch.randn(2,32,64,64)
    m=CondConv(in_planes=32,out_planes=64,kernel_size=3,stride=1,padding=1,bias=False)
    out=m(input)
    print(out.shape)

```



## 其他项目推荐

-------

🔥🔥🔥 重磅！！！作为项目补充，更多论文层面的解析，可以关注新开源的项目 **[FightingCV-Paper-Reading](https://github.com/xmu-xiaoma666/FightingCV-Paper-Reading)** ，里面汇集和整理了各大顶会顶刊的论文解析



🔥🔥🔥重磅！！！ 最近为大家整理了网上的各种AI相关的视频教程和必读论文 **[FightingCV-Course
](https://github.com/xmu-xiaoma666/FightingCV-Course)**


🔥🔥🔥 重磅！！！最近全新开源了一个 **[YOLOAir](https://github.com/iscyy/yoloair)** 目标检测代码库 ，里面集成了多种YOLO模型，包括YOLOv5, YOLOv7,YOLOR, YOLOX,YOLOv4, YOLOv3以及其他YOLO模型，还包括多种现有Attention机制。


🔥🔥🔥 **ECCV2022论文汇总：[ECCV2022-Paper-List](https://github.com/xmu-xiaoma666/ECCV2022-Paper-List/blob/master/README.md)**


<!-- ![image](https://user-images.githubusercontent.com/33897496/184842902-9acff374-b3e7-401a-80fd-9d484e40c637.png) -->
