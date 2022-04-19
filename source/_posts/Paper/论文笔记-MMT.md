---
title: 论文笔记 | Multi-modal Transformer for Video Retrieval
tags: 
  - MultiModal
  - Transformer
  - Video-Text
  - Retrieval
  - CrossModal
  - MMT
categories:
  - Papers
comments: true
mathjax: true
date: 2021-05-20 13:34:18
urlname: Multi-modalTransformerforVideoRetrieval
---

<meta name="referrer" content="no-referrer" />

{% note info %}
论文《Multi-modal Transformer for Video Retrieval》笔记
{% endnote %}
<!--more-->
## 贡献

1. 提出了一种video encoder（transformer encoder结构）
2. 证明了在text embed方面，bert最好
3. 在text-video方面刷了sota

## 模型结构

![oOPbHp](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/oOPbHp.png)

![xE8s5u](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xE8s5u.png)

### video

#### Feature F

用N个feature extractors，每个expert是针对某个任务的video特征提取器，然后在它们之间学习一个一个跨模态的并且有长期时序关系的编码。

不同的expert输出的空间不一样，所以每个又接了一个linear层去统一到维度`$d_model$`。

对于每个video，每个expert会提取出K个feature。然后做一个maxpooling，让每个expert对一个video得到一个统一的feature。这样，每个expert对每个video就有K+1个feature。作为encoder的输入。

#### Expert embedding

为了得到跨模态信息，要确定每个expert的attention权重，以及指导每个expert embedding之间的区别，又对上面的每个feature，分别学一个expert embedding，一一对应，也是(K+1)*N个，看上面的图就很清楚，一一对应的。

#### Temporal embedding

用多模transformer提取的time维度的时序特征。

在每个视频片段的`$t_{max}$`上，训D个 `$d_{model}$`维度的特征，D就是这个时间段最长的那个位置。和上面的feature，expert embedding都是对应的。然后再学一个 `$T_{agg}, T_{unk}$`，分别表示这段时间特征的aggregate和位置的时序特征。

#### Multi-modal transformer

video embedding就是feature、expert、temporal embedding的加和，得到`$\Omega(v)$`。但是同样的，这个embedding也有（K+1）x N个。然后需要用MMT对它做一个aggregate，每个expert只保留一个整体特征，得到`$\Psi_{agg}(v)$`。

每个Multi-modal transformer都是transformer encoder架构。

这个MMT最终得到的embedding优势：
1. 输入的embedding不是一步到位直接得到，而是在迭代中优化
2. 保留了时序上面的所有特征，而不是只保留一个整体的特征，这个让模型知道每个位置的时序特征


### Caption embedding

用BERT的`[CLS]`得到caption embedding `$h_c$`，也有N个，和video embedding对应

### similarity

相似度是通过每个expert的video-caption相似度加权得到，每个expert的similarity直接可以通过加权得到。

![DrKxEB](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/DrKxEB.png)
![ZIkkSp](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ZIkkSp.png)

每个expert的权重是通过对caption embedding做一个FC+softmax得到，**可能caption对相似度的作用更大？？**

能看出来，这里的video embeddding，caption embedding，还有weight都是可以线下得到。

### 数据集

![7dKsYA](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7dKsYA.png)
![wG4GD8](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wG4GD8.png)

### 指标

R@N，MdR，MnR

![QZrQYl](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/QZrQYl.png)

### 预训练的Expert

1. Motion：S3D
2. Audio: VGGish
3. Scene: DenseNet-161
4. OCR: Overlaid text is ﬁrst detected using the pixel link text detection model. The detected boxes are then passed through a text recognition model trained on the Synth90K dataset. Finally, each character sequence is encoded with word2vec
5. Face: SSD=>ResNet
6. Speech: Google Cloud Speech
6. Appearance: ﬁnal global average pooling layer of SENet-154

![ZNLqFo](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ZNLqFo.png)

