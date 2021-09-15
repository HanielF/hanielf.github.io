---
title: "论文笔记 | Diagnosing the Environment Bias in Vision-and-Language Navigation"
mathjax: true
date: 2021-09-12 20:11:01
tags:
  [
    EnvBiasVLN,
    VLN
  ]
categories: Papers
urlname: env-bias-vln
---

<meta name="referrer" content="no-referrer" />

{% note info %}
2020 IJCAI 《Diagnosing the Environment Bias in Vision-and-Language Navigation》论文笔记
{% endnote %}

<!--more-->

## Abstract and Instroduction

1. 发现agent在unseen的环境中效果显著下降
2. 环境重新分割和特征替换设计了新的诊断实验，研究了这种环境偏差的可能原因
3. 发现无论是Instruction还是底层导航图，ResNet 特征传达的低层次的视觉embedding会直接影响agent模型，并导致结果中的这种环境偏差
4. 探索了几种包含较少低级视觉信息的语义表示，使用这些特征学习的agent可以更好地泛化到unseen的测试环境。分别是ResNet ，ImageNet， Detection， GT-Seg，和 Learned-Seg
5. 结果是在R2R和R4R，和CVDN数据集上显著降低了可见和不可见之间的性能差距，并实现了在unseen上的sota模型