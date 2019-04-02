---
title: MNIST入门-手写数字识别
comments: true
mathjax: false
date: 2019-04-02 10:11:01
tags:
categories:
---

<meta name="referrer" content="no-referrer" />

{% note info %}
**Target**  
将训练一个机器学习模型用于预测图片里面的数字.  

理解TensorFlow工作流程和机器学习的基本概念

和官网不同的是只记录关键点
{% endnote %}
<!--more-->
**关键词**

[MNIST](http://yann.lecun.com/exdb/mnist/)
[TensorFlow](http://www.tensorfly.cn/tfdoc/tutorials/mnist_beginners.html)
[Softmax Regression](https://blog.csdn.net/google19890102/article/details/41594889)

# 数据集
- 数据集被分成两部分：60000行的训练数据集（mnist.train）和10000行的测试数据集（mnist.test）
- 把这些图片设为“xs”，把这些标签设为“ys”
  - 训练数据集的图片是 mnist.train.images ，训练数据集的标签是 mnist.train.labels
- mnist.train.images 是一个形状为 [60000, 784] 的张量，第一个维度数字索引图片，第二个维度数字索引每张图片中的像素点
  - 向量值表某个像素的强度值，值介于0和1之间
- mnist.train.labels 是一个 [60000, 10] 的数字矩阵
  - 标签数据是"one-hot vectors"
  - 10维，只有一个维度非0
  - 标签0将表示成([1,0,0,0,0,0,0,0,0,0,0])

# Softmax
softmax模型可以用来给不同的对象分配概率。即使在之后，我们训练更加精细的模型时，最后一步也需要用softmax来分配概率。

## 第一步

