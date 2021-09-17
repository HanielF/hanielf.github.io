---
title: "论文笔记 | Structured Scene Memory for Vision-Language Navigation"
mathjax: true
date: 2021-09-12 20:11:01
tags:
  [
    SceneMemoryVLN,
    VLN,
    Graph,
    GNN,
    PaperNotes,
    Multimodal,
    DeepLearning
  ]
categories: Papers
urlname: scene-memory-vln
---

<meta name="referrer" content="no-referrer" />

{% note info %}
2021 CVPR 《Structured Scene Memory for Vision-Language Navigation》论文精读笔记
{% endnote %}

<!--more-->

## Abstract and Instruction

1. SceneMemoryVLN:
   1. 结构化的外部memory表示，分开捕获视觉信息和几何空间的信息。**就是graph结构，用node编码视觉信息，edge编码空间几何信息以及连接信息**
   2. 用图结构存储，能够精确的存储和直接调用过去的memory。这个图也是动态生成的
   3. 用一个collect-read controller来根据当前的text，动态读取memory中的内容
   4. 能够获得全局的环境布局，并且把action space设置成map上可达位置的集合，用于全局决策的path生成。避免了陷入局部最优决策
   5. 针对全局action空间会逐渐夸扩大的问题，提出了基于前言探索的决策策略，就是用了一个trontier node，尝试前一个node行不行，不行就回来，如果行就继续走下去，像是在map上试错
2. 目前的vln方法：
   1. 更强的学习范式
   2. 更多的监督信息
   3. 更高效的多模编码
   4. 更智能的path设计
   5. 基本上都是在基于seq2seq的框架，就是映射Instruction和observation到actions上
3. seq2seq框架的问题：
   1. 由于使用了recurrent hidden state，因此保存的信息都被混合在了一个hidden里面
   2. 这阻碍了agent直接访问其过去的观察并了解环境布局
   3. Decision被局限在一个local的action space，容易陷入局部最优
   4. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/lF9XSo.png" width="500">
   5. 就像Figure 1b，用recurrent hidden state，只能感知到蓝色区域的，并且让决策陷入了一个local action space，就是绿色区域的，而实际的最优决策是p2处走
4. 数据集：R2R和R4R
5. external memory参考：Alex Graves, Greg Wayne, and Ivo Danihelka. Neural turing machines. arXiv preprint arXiv:1410.5401, 2014
6. R4R参考：Vihan Jain, Gabriel Magalhaes, Alexander Ku, Ashish Vaswani, Eugene Ie, and Jason Baldridge. Stay on the path: Instruction ﬁdelity in vision-and-language navigation. In ACL, 2019

## Approach

1. 环境空间被视作一些离散点集合，点之间的可达性都是给出的
2. 符号表示：
   1. 文本特征：$X = {x_l}_l$
   2. 方向特征：$r_{t,k}=(cos \theta_{t,k}, sin \theta_{t,k},cos \phi_{t,k},sin\phi_{t,k})$
   3. 视觉特征：$f_{t,k}$
3. agent有一个navigation state $h_t$
   1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/5RCG8Q.png" width="500">
   2. 通过instruction的text feature和所有可达viewpoint的visual feature，以及上一个hidden得到

### SSM construction

1. 有向图 $G_t = (V_t, E_t)$，V里面的node都是对应之前走过的地方，edge表示它们的连边
2. 节点embedding，主要是视觉特征，编码位置v处的全景特征
   1. $v = \{f_{v,k}\}_{k=1}^{K_v}$
   2. $f_{v,k}$是第k个img的特征
   3. $K_v$是在位置v处可达位置的数量
3. 边embedding：
   1. $e_{u,v} = r_{u,k_v} = (cos \theta_{t,k_v}, sin \theta_{t,k_v},cos \phi_{t,k_v},sin\phi_{t,k_v})$
   2. $r_{u,k_v}$表示u是当前位置v的可达节点，第k个viewpoint的方向特征，u是v的这个方向的可达节点
4. global action space含义
   1. 每个节点有 $K_v$ 个可达点，因此给每个可达点都设置一个子节点，即 $\{o_{v,k}\}_{k=1}^{K_v}$ ，每个子节点表示一个可达的viewpoint $o_{v,k}$
   2. 图里面**所有的sub nodes**构成了action space $A_t = \{o_{v,k}\}_{v=1,k=1}^{|V_t|,K_v}$
   3. 相比之前的VLN agent只能访问当前可达的viewpoint，对应这里的当前节点的sub node
   4. 能够看到图里面所有sub node的好处就是，能够从当前的action直接跳到好几个时间步之前访问的sub node
5. graph的更新
   1. 动态生成graph，初始化是开始节点
   2. 在时间步t，选择了 $A_t$的action，转移到了 $u$，这时候是一个sub node，然后会产生一个新的节点 $u$，这时候原本的sub node就被移除了
   3. 同一个时间段，可能有多个sub node指向同一个place，这些sub node属于不同的node，就是从不同的地方都能到达那个place

### Fine-Grained Instruction Grounding based Iterative Reasoning

用一个collect-read controller来控制下一步决策，它从Instruction中获得percepti和object这两个相关的信息

#### Perception-and-Action Aware Textual Grounding

1. Instruction有的是描述action的，比如turn left，有的是和landmarks相关的，比如bedroom
2. 因此用了两个不同的attention model从文本中学习
3. 用当前的hidden state和Instruction，计算当前perception和action相关的文本信息，就是算了score之后加权和
   1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7flSm6.png" width="500">
4. 提取出信息后，就可以生成perception和action对应当前hidden state的state
   1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ObQG9A.png" width="500">
5. 文本相关信息提取完了，开始提取视觉信息，因为视觉信息和perception 信息是相关的，所以这里相当于做了一个perception部分的融合
   1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Mq1HxS.png" width="500">
6. 然后还需要做一个action部分的融合，这里是把action state和边信息做融合
   1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/6oycom.png" width="500">

### Iterative Algorithm based Long-Term Reasoning

1. 通过在节点之间信息传递进行视觉和方向上的特征更新
2. 在更新S次之后
   1. 此时的visual和orientation特征，分别表示为 $\hat{f}_v^S$和 $\hat{r}_v^S$
   2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/JCoVN5.png" width="500">
   3. 是对节点v的邻居，进行visual和orientation的特征求和，得到当前节点的邻居信息 $m_v^f, m_v^r$
   4. 然后用了GRU，把邻居信息和自己的visual，orientation信息融合，得到更新后的节点信息

## Frontier-Exploration based Decision Making

### Global Action Space based One-Step Decision Making

1. 每个节点都有好几个sub node，他们都是可达的，要计算对每个sub node的分数
2. 当前主节点的visual和orientation信息：$\hat{f}_v^S, \hat{r}_v^S$
3. 当前主节点的perception state和action state：$h_t^p, h_t^a$
4. visual信息对应文本的perception state，orientation信息对应文本的action state
5. 当前主节点的visual feature和子节点的visual feature进行拼接，然后和主节点的perception state一起做一个计算，得到perception的分数
6. 当前主节点的action state和子节点的edge feature进行拼接，得到action的分数
7. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/qCj72D.png" width="500">
8. 需要把这两个分数融合成一个分数，每个sub node都有两个分数，再做一个加权的softmax，权重是根据主节点的perception state和action state得到。权重控制perception和action哪个部分对decision的占比更高
   1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Fi839N.png" width="500">
9. 最后的action $a_t$是取argmax：$a_t = argmax_{v,k} p(o_{v,k})$

### Navigation Utility Driven Frontier Selection

1. 因为action space是图上所有的sub node，因此随着加入的node越多，sub nodes越多，action space就越大，这里提出了frontier Selection。
2. frontier其实就是在当前节点做action 选择的时候，先选一个frontier node，然后选择它的一个sub node去尝试性走一步。
3. frontier相当于是把走到下一个node的过程分成两个部分，如果选择下一个node是一个goal，那么frontier node的选择就是一个sub goal
4. 怎么选frontier node？
   1. 根据argmax概率选
   2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/JRRLsL.png" width="500">
   3. w是选择的frontier node
   4. $\hat{f}_w^{ST}$是frontier节点visual feature
   5. $\hat{r}_v^{ST}$是frontier节点的action state
   6. $h_t^p，h_t^a$是当前节点的perception state 和action state
   7. frontier node选择的机制是RL的方式，如果选出来的更接近destination，就奖励。对比以前的启发式选择方式
5. 怎么选frontier node 的 sub node?
   1. Frontier node和当前不一样就移动，沿着G上面的最短路径移动。节点状态也在move过程中更新。如果一样就呆在原地
   2. 计算sub node的概率和计算frontier node的概率一样，用sub node的visual 特征和action特征，来和当前frontier node的perception stte和action state算概率
      1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/c0o11r.png" width="500">
   3. 选择概率最大的sub node作为新的位置 $u$，加入到SSM的graph中，更新成 $G_{t+1}$
   4. 这时候还要**更新agent的navigation state**
      1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/5RCG8Q.png" width="500">

## Implement Details

