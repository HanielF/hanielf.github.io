---
title: 论文笔记 | Attributed Graph Clustering A Deep Attentional Embedding Approach
  - GNN
  - DAEGC
  - IJCAI-2019
categories:
  - Papers
comments: true
mathjax: true
date: 2021-05-26 19:34:18
urlname: gnn-daegc-IJCAI-2019
---

<meta name="referrer" content="no-referrer" />

{% note info %}

之前研究了DAEGC模型的源码和论文，补个笔记。

论文《Attributed Graph Clustering: A Deep Attentional Embedding Approach》，模型结果在 Node Clustering on Cora 上，Acc、NMI、ARI排第5， F1排第四。

{% endnote %}

<!--more-->

## Abstract & Introduction

1. Graph clustering
   1. 在网络中挖掘communities和groups
   2. 目标是将节点划分成不想交的group
   3. Attributed graph cluster关键问题是如何捕获结构关系和节点信息
2. 近期大多数工作都是学一个graph embedding，然后用传统的聚类方法，比如k-means或者谱聚类，谱聚类在之前文章[《GNN和图聚类》](https://hanielxx.com/MachineLearning/2021-05-09-gnn-graph-clustering)中有。
3. 之前的工作主要是two-step框架下的，文章认为这种方式不是goal-directed，比如面向一些特殊的clustering任务，所以提出了一个goal-directed的方法，
4. 使用了GAT来捕获邻居特征的重要性，同时encode网络的拓扑结构和节点信息，后面用简单的inner product decoder来重建图信息。用这个GAE生成预测的邻接矩阵A_pred和graph中每个node embedding，作为后面的初始的soft label，用生成的soft label来supervise后面的self-training。
5. 主要是分两个节点，一个是GAE阶段，得到一个初始的soft label，就是一个k-menas的结果，然后用self-training的一个算法进行迭代更新聚类中心
6. 这篇文章主要针对大图的复杂度问题和计算量，在对邻居aggregate的时候是sample

## Proposed Method

