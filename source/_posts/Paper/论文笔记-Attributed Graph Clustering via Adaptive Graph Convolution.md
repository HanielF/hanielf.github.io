---
title: 论文笔记 | Attributed Graph Clustering via Adaptive Graph Convolution
tags:
  - GNN
  - AGC
  - IJCAI-2019
  - GraphClustering
  - NodeClustering
categories:
  - Papers
comments: true
mathjax: true
date: 2021-05-29 11:34:18
urlname: gnn-agc-IJCAI-2019
---

<meta name="referrer" content="no-referrer" />

{% note info %}
提出当前对GCN如何影响聚类效果以及如何优化GCN聚类在不同图上的效果，还不是很清楚。存在的方法都是对固定的low-order几个邻居进行gcn，弱化了节点关联信息，并且忽视了图的多样性。

这篇论文主要提出了一种自适应的图卷积方法，应用在图聚类问题上。应用了high-order的GCN来捕获全局的结构信息，并且自适应地对不同的图选择合适的order。
{% endnote %}

<!--more-->

## Introduction

1. 主要还是强调了当前的方法
   1. 没有应用更多的节点信息
   2. 多数方法直接用gcn作为特征提取器，而没有研究如何让聚类效果最大化
   3. 低阶邻居，没有考虑大图中全局cluster结构
   4. 忽视了现实中图的多样性
2. 出发点是
   1. 相邻节点更可能在同一个cluster
   2. 对同一个cluster的有相似特征的节点聚类会更容易
3. 设计了k-order的GCN来filte node feature，得到更smooth的feature embedding，k是用intra-cluster distance自动确定
4. 模型主要分两步
   1. k-order的GCN得到embedding
   2. 对embedding进行spectral clustering
5. 感觉关键是能够自动选择合适的k，以及捕获全局结构信息

## The Proposed Method

1. 给定无向图 $G=(V, E, X)$，V是顶点集合，有n个点，E是边表，用邻接矩阵A表示，$A \in R^{n\times n}$，$X = [x_1, x_2, ···, x_n]^T \in R^{n\times d}$。如果 $v_j$可以让 $v_i$通过k个边到达，就说 $v_j$是$v_i$的 $k-hop$邻居。
2. 目标是将G分成m个不相交的clusters
3. 