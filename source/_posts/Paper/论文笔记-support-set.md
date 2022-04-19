---
title: 论文笔记 | Support-set bottlenecks for video-text representation learning
tags:
  - Support-Set
  - CrossModal
  - VideoTextRetrieval
  - MultiModal
  - ContrastiveLearning
categories:
  - Papers
comments: true
mathjax: true
date: 2021-11-04 10:34:18
urlname: support-set-video-text
---

<meta name="referrer" content="no-referrer" />

{% note info %}
论文《Support-set bottlenecks for video-text representation learning》笔记
{% endnote %}
<!--more-->
## 背景和概述

1. 目前的模型用的contrastive learning，直接强迫负样本对的距离要远，正样本对的距离要近，但是这样就导致，有一些相似样本，比如图像内容相似，但是做的事情不一样，这样的负样本就被强迫的远离了，这样会太严格了
2. 直接的想法就是放松约束
3. 这里结合了另一个loss，在minibatch里面的其他img来加权生成正样本的caption，就是decoder，然后让生成的caption和当前正样本的caption尽可能接近，就是autoencoder的思路
4. 把上述的方法称为cross instance caption

## 总结

无

## 模型和方法

1. cross modal discrimination and cross-captioning ![1DRih4QbvTPr6Eo](https://i.loli.net/2021/11/04/1DRih4QbvTPr6Eo.png)
2. 模型结构![z64vm3NXgEfrtYe](https://i.loli.net/2021/11/04/z64vm3NXgEfrtYe.png)

### 方法

1. ![tHjxvaZUqKfks5i](https://i.loli.net/2021/11/04/tHjxvaZUqKfks5i.png)
2. ![oCRTGIVqtWYjlun](https://i.loli.net/2021/11/04/oCRTGIVqtWYjlun.png)
3. ![9qbBpzKwDClGaVN](https://i.loli.net/2021/11/04/9qbBpzKwDClGaVN.png)

### 模型结构

![n2ou74ewLCTaXxr](https://i.loli.net/2021/11/04/n2ou74ewLCTaXxr.png)

## 结果

![CB91nw375mSse4f](https://i.loli.net/2021/11/04/CB91nw375mSse4f.png)

![23IHtw9ACdkBv7a](https://i.loli.net/2021/11/04/23IHtw9ACdkBv7a.png)