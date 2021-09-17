---
title: "论文笔记 | Diagnosing the Environment Bias in Vision-and-Language Navigation"
mathjax: true
date: 2021-09-12 20:11:01
tags:
  [
    EnvBiasVLN,
    Multimodal,
    Dataset,
    DeepLearning,
    VLN,
    Notes
  ]
categories: Papers
urlname: env-bias-vln
---

<meta name="referrer" content="no-referrer" />

{% note info %}
2020 IJCAI 《Diagnosing the Environment Bias in Vision-and-Language Navigation》论文笔记
{% endnote %}

<!--more-->

## 前言和背景

1. 发现agent在unseen的环境中效果显著下降
2. 环境重新分割和特征替换设计了新的诊断实验，研究了这种环境偏差的可能原因
3. 发现无论是Instruction还是底层导航图，ResNet 特征传达的低层次的视觉embedding会直接影响agent模型，并导致结果中的这种环境偏差
4. 探索了几种包含较少低级视觉信息的语义表示，使用这些特征学习的agent可以更好地泛化到unseen的测试环境。分别是ResNet ，ImageNet， Detection， GT-Seg，和 Learned-Seg
5. 结果是在R2R和R4R，和CVDN数据集上显著降低了可见和不可见之间的性能差距，并实现了在unseen上的sota模型

## 提出问题

1. 发现各个模型在seen和unseen环境下的效果有明显的gap
2. 因此研究了
   1. 偏差在哪里
      1. 在视觉特征里面
   2. 为什么存在偏差
      1. resnet-2048维的特征在seen environment上过拟合了
   3. 如何消除偏差
      1. 加入image的语义分割信息

## 模型和方法

<img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pl8fGR.png" width="500">

1. 发现在indoor（R2R, R4R, CVDN）和outdoor（TouchDown）的数据集上面都存在这种bias
2. 为了找到偏差在哪，分别研究了instruction和navigation graph，以及视觉特征对模型的影响
   1. 用了BLEU-4分数和ROUGE-L分数，用训练集的instructions作为references，最后发现：instruction之间的distance在seen和unseen之间没有明显差异；并且分数越高，即距离越小，SR越高；seen的SR明显比unseen的高。这说明bias不在instruction中
   2. 测试也发现不是navigation graph的影响
   3. 用Path-level, region-level，environment-level分别测试了visiual environment，发现这三个层次的bias都存在。
3. 直接指出了是low-level的视觉特征，就是Resnet中的特征导致了这种bias，是2048维的resnet特征在seen环境下过拟合了
4. 提出了
   1. 不用池化特征，而是从 ResNet 预训练中注入冻结的 1000 路分类层，用imageNet的label的概率作为视觉特征。尽管这样做让效果变差了，但是seen和unseen之间的gap变小了，因此后面又去找更好的environment feature 表示方法，表中的ImageNet行
   2. 计算所有环境中每个检测对象的总面积，并选择占环境比例较大的标签，创建维度为 152 的特征，表中的Detection行
   3. 计算语义分割的特征，把图片按照像素分割，并且给每个区域一个label
      1. 把语义分割信息作为附加信息，并且用真实语义分割区域，是表里的“GT-Seg”行
      2. 考虑到unseen中是没有ground truth的，所以用MLP单独来预测这些区域，它输出的结果作为语义特征，用Learned-seg表示
5. 结果就是，加入了语义分割信息的特征效果最好

## 总结和思考

1. 这是最直接的对数据集的改动，在后面实验的时候可以做对比实验，用它们的特征，或者加入语义分割信息