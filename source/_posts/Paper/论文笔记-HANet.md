---
title: 论文笔记 | HANet:Hierarchical Alignment Networks for Video-Text Retrieval
tags:
  - MultiModal
  - Transformer
  - Video-Text
  - Retrieval
  - CrossModal
  - HANet
categories:
  - Papers
comments: true
mathjax: true
date: 2021-11-02 13:34:18
urlname: HANet-Hierarchical Alignment Networks for Video-Text Retrieval
---

<meta name="referrer" content="no-referrer" />

{% note info %}
论文《HANet: Hierarchical Alignment Networks for Video-Text Retrieval》笔记
{% endnote %}
<!--more-->
## 背景

1. 直接编码到joint latent space，会导致一些细粒度的信息丢失
2. 现有的用local的方式的模型，会导致文本和视频的不对称
  1. 用local的细粒度信息
  2. 把句子按照名词和动词拆分，用pos parse
  3. 用全局event和local action entity的层次图推理

## 不足

1. 个人感觉这种直接预定义concept的方式不太灵活，有些视频就是没有那么多concept
2. 分为多种粒度并且层次化构建模型值得学习
3. 感觉缺少了时序上的信息，并且用的是CNN，用一些时序模型来提取特征是不是更好点

## 模型

提出了Hierarchical Alignment Network (HANet)，把video和text划分为三个粒度，分别是event (video and text), action (motion and verb), and entity (appearance and noun)，然后对这三个粒度分别进行特征提取，基于这三个粒度，构建一种自底向上的层次化模型结构，从frame-word，到video clip and textual context，最后再到whole video and text。

这里的event，action，entity都是第二个层次的特征，其中event是全局的，action是entity是局部的。

![2BNJwN](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/2BNJwN.png)

### Video Parsing and Representations

#### local level

1. video：没有直接用分割、检测追踪等方式切分，而是让模型去学习是属于啥。这里是预定义了每个视频8个concept，然后计算每个帧属于每个concept的概率。![BR9iO0](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/BR9iO0.png)
2. text：![XMPgcw](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/XMPgcw.png)
3. 上面的是action和entity级别的feature，但是缺了全局的。
4. 通过fc能得到上面的 `$v_x^{Ind}$`![jE7emJ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/jE7emJ.png)
5. local level的通过SE模块提取，其实就是Squeeze-and-Excitation block，是CNN。![9zA7kg](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/9zA7kg.png)
6. each frame has 𝐾 𝑎 dimensional action concept confidence and 𝐾 𝑒 dimensional entity concept confidence
7. 对每个concept，挑选高置信度的帧
8. ![YUYhSg](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/YUYhSg.png)
8. ![BMGvAa](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/BMGvAa.png)

#### global level

![6LrSYb](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/6LrSYb.png)

### Text Representations

#### local level

1. individual level的直接用的Bi-GRU，`$v_g^Ind$`使用的注意力融合
2. local和global的是用一层GCN。![Qc2BZ9](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Qc2BZ9.png)


### Hierarchical Alignment

1. ![8I4gFZ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/8I4gFZ.png)
2. ![oQfeCZ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/oQfeCZ.png)