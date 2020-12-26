---
title: 论文笔记 | 基于机器学习的siRNA沉默效率预测方法研究
tags:
  - Notes
  - siRNA
  - MachineLearning
  - Paper
categories:
  - Papers
comments: true
mathjax: false
date: 2019-10-17 10:56:51
urlname: Paper-note-Research-on-prediction-method-of-siRNA-silencing-efficiency-based-on-machine-learning
---

<meta name="referrer" content="no-referrer" />

{% note info %}
最近在看[基于机器学习的siRNA沉默效率预测方法研究](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CDFD&dbname=CDFDLAST2017&filename=1017152231.nh)这篇论文，论文一共98页，实在太长，不得不单独一篇笔记用于记录要点。

这篇论文在使用机器学习方法研究siRNA沉默效率这方面讲的还是很清楚的，用以入门。
{% endnote %}
<!--more-->

# 绪论
## 研究意义
这里主要讲述了RNAi的起源、作用机制和应用，可以参考上一篇文章[RNAi和siRNA设计基础](https://hanielxx.com/Notes/2019-10-01-basic-knowledge-of-RNAi.html)，这里就不记录了。

## 国内外研究动态
### 基于规则的第一代 siRNA 沉默效率预测方法
早期主要是用统计学方法寻找siRNA不同沉默效率的规则，因此沉默效率不能量化，只能分高效和低效siRNA

主要是在siRNA不同位置上的规则约束，这部分可以用一张表总结。
{% asset_img 1-1.png siRNA序列打分规则 %}

其次需要考虑siRNA热力学稳定性，主要是下面两种规则：
{% asset_img 1-2.png siRNA热力学稳定性规则 %}

**缺点是：**
1. 样本量太少
1. 规则不够具体
1. 没有设置权重区分规则重要性
1. 具有数据偏向性
### 基于机器学习的第二代 siRNA 沉默效率预测方法

#### 数据集方面
主要是使用Huesken数据集，Huesken 数据集到目前为止同样实验条件下提出的数量最多的数据集。

其他一些包括化学修饰的数据集可以有需要再去论文中查看。

### 特征方面
1. 碱基组成
1. 1-3mer的motif频率和位置特征

1. 热力学参数，包括siRNA 双链的G ，siRNA 反义链的内部分子结构稳定性，局部靶标的 mRNA 稳定性以及 siRNA 双链中每相邻两个碱基对的稳定性等等
1. 反义链的二级结构
1. 与mRNA有区别的结构特征
1. 靶mRNA的二级结构
1. 靶mRNA内多个反义链结合位点的能量

1. siRNA的3n+1位碱基组成，也意味着siRNA的绑定蛋白和效率有关
1. siRNA上下游碱基，可能是上下游特定的motif可能影响效率


#### 算法方面
统计如下：
1. GP算法
1. GSK和SVM
1. 神经网络
1. 线性回归
1. 决策树
1. 随机森林\>SVM
1. 后缀树
1. 规则矩阵，利用已知规则设置权重，结合半监督的回归算法

# 机器学习在siRNA沉默效率预测中的应用

## siRNA样本收集
此需要选择具有代表性、数量充足的样本集合。生物信息学使用的数据还需要注意数据是否存在冗余


