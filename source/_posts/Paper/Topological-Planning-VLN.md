---
title: "论文笔记 | Topological Planning with Transformers for Vision-and-Language Navigation"
mathjax: true
date: 2021-09-10 20:11:01
tags:
  [
    Topological-Plannint-VLN,
    VLN,
    TopologicalMap,
    Transformer,
    GNN,
    PaperNotes,
    Multimodal,
    DeepLearning,
    CVPR,
  ]
categories: Papers
urlname: topological-planning-vln
---

<meta name="referrer" content="no-referrer" />

{% note info %}
2021 CVPR 《Topological Planning with Transformers for Vision-and-Language Navigation》论文笔记
{% endnote %}

<!--more-->

## Abstract and Introduction

1. 提出的方法主要是在transformer的基础上，结合了拓扑图，
2. 先是得到一个navigation plan，其实就是路径，然后执行low-level的actions
3. 用了模块化的方法
4. 拓扑图能够更好的适应于navigation任务，因为map提供的空间离散化可以让模型学到指令和空间位置之间的关系。
5. 用跨模态的transformer在拓扑图上面计算navigation plan，就是path。预测过程和语言模型有点像，语言模型每次预测一个word，这里是每次预测一个map node。拓扑图也可以让instruction、img feature和node对应起来。
6. 用了预训练的模型

## Method

1. problem setup
   1. 让机器人自己对环境进行探索，得到exploration trajectories，这和现实场景比较像，机器人需要对一个陌生的环境做一个自己的理解，而不是用人为预定义好的map
2. overview
   1. 在agent得到一个map之后，执行vln task，分成两个步骤，一个是planning，一个是control，其实就是得到path，然后机器人移动到下一个点
   2. planning阶段，用instr生成path，然后robot尝试path，直到到达目的地
   3. control阶段先预测小目标点，然后把这个小目标点转换成low-level的action，比如 $FORWARD(0.25m)$，$ROTATE(15°)$

### Topological Map representation

1. 对map的一些要求
   1. 覆盖广，信息充足，包含可达概率比较高的节点
2. 用graph表示这个map，刚开始建立map的时候，agent多次探索环境，然后得到一些稀疏路径，最后把这些路径merge起来，得到一个完整的。在得到稀疏路径之前是机器人走过的连续路径，这里用了一个Meng提出的reachability estimator,进行了稀疏化。
3. 在map上面用了粗略的集合信息，引入一个具有 8 个方向和 3 个距离（0-2m、2-5m、> 5m）的量化极坐标系。

### Cross-Modal Planning

1. 这个plan阶段其实就是生产path node sequence的阶段
2. 主要是两个模块，一个GNN，另一个是Transformer
3. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/WQcv3l.png" width="500">
4. 用了 **GNN**来对map提取特征信息和链接信息，得到每个节点的embedding
5. 用cross-modal Transformer来选择下一个node，并且重复这个操作，直到输出end，就像nlp里面的seq2seq
6. GNN，是为了捕获visual、semantics和链接信息，输入ResNet152的feature特征，每对节点之间还可以得到相对的方向信息和距离信息，最后是提取到每个node的embedding。
7. cross-modal transformer，是基于 **LXMERT**预训练模型。每一层cross-modal encoder都是由cross-modal attention，self-attention，FN组成，query来自其中一个模态，key和val都来自另一个模态，$CrossAttn(x_A, x_B) = Attn(Q_A,K_B,V_B)$
8. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1zmWcd.png" width="500">
9. Trajectory position encoding，是为了保持历史预测的信息
10. 用 $[STOP]$表示结束信息
11. 用adamw+linear warmup+cross entropy 训练gnn和transformer

### Controller: Executing the Navigation Plan

1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/oOLiCQ.png" width="500">
2. 其实就是负责把topological path转换成low-level的action
3. controller抽象成了两个，一个是low level的cotroller，负责移动agent，一个是high level的controller，负责预测waypoint
4. high level controller，用目标点的上一个节点、当前观测点，目标点，来预测下一个节点，包括角度和距离。整个空间被分成了24个扇区，每个扇区15°，用sector index和距离来定位waypoint位置，预测得到的distance适用于确定是否到达了subgoal
5. handling oscllation，存在感知重叠的可能性，其中代理的观察在多个方向上是相似的。为了缓解这个问题，我们在预测航向概率中引入了“顽固性”，使得预测方向偏向于前一个时间步长的预测。有点像moment的意思
6. mapping to low level action，high controller预测出的waypoint，需要用专门的controller来让机器人到达那个位置
7. localization and trajectory following，就是预测的点和graph node的距离在threshold distance d范围内，就认为那个node是下一个节点，然后认为plan中的下一个节点是一个新的subgoal，重复这个过程直到结束。