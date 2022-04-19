---
title: 论文笔记 | Less is More:CLIP BERT for Video-and-Language Learning via Sparse Sampling
tags:
  - ClipBERT
  - MultiModal
  - Transformer
  - Backbone
  - Video-Text-Architecture
categories:
  - Papers
comments: true
mathjax: true
date: 2021-11-09 13:34:18
urlname: ClipBERT
---

<meta name="referrer" content="no-referrer" />

{% note info %}
论文《Less is More: CLIP BERT for Video-and-Language Learning via Sparse Sampling》笔记
{% endnote %}
<!--more-->
## 背景和问题

![GORKRY](https://raw.githubusercontent.com/HanielF/ImageRepo/main/blog/GORKRY.png)

![fQ6h6S](https://raw.githubusercontent.com/HanielF/ImageRepo/main/blog/fQ6h6S.png)

![kKys61](https://raw.githubusercontent.com/HanielF/ImageRepo/main/blog/kKys61.png)

## 概述和总结

1. 针对task之间没有联系，以及多模特征之间没有联系的问题
2. 用采样的方式，降低复杂度，然后对特征进行finetune，以前都是固定的
3. 用预训练的权重提升性能，并且用2D的结构(也能降低计算量）就是用的ResNet50。后续训练会对权重做finetune

![rXNvoF](https://raw.githubusercontent.com/HanielF/ImageRepo/main/blog/rXNvoF.png)

![3DChYe](https://raw.githubusercontent.com/HanielF/ImageRepo/main/blog/3DChYe.png)

## 模型和方法

### 形式化表示
![71WXFr](https://raw.githubusercontent.com/HanielF/ImageRepo/main/blog/71WXFr.png)

### 模型结构

![huI56O](https://raw.githubusercontent.com/HanielF/ImageRepo/main/blog/huI56O.png)

![xD33Uz](https://raw.githubusercontent.com/HanielF/ImageRepo/main/blog/xD33Uz.png)

### 初始化

![VuB7ed](https://raw.githubusercontent.com/HanielF/ImageRepo/main/blog/VuB7ed.png)

## 实验结果

![smbj0l](https://raw.githubusercontent.com/HanielF/ImageRepo/main/blog/smbj0l.png)