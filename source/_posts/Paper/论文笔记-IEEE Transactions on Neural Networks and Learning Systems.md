---
title: 论文笔记 | A Comprehensive Survey on Graph Neural Networks
tags:
  - GNN
  - Survey
  - Review
  - IEEE
  - 2021
categories:
  - Papers
comments: true
mathjax: true
date: 2021-05-20 13:34:18
urlname: gnn-survey-IEEE-2021
---

<meta name="referrer" content="no-referrer" />

{% note info %}

- 一篇关于GNN的综述，将GNN分为
  - namely recurrent graph neural networks,
  - convolutional graph neural networks,
  - graph autoen- coders,
  - spatial-temporal graph neural networks
- 主要内容是
  - GNN发展
  - GNN分类
  - GNN模型的overview
  - GNN模型在各个领域的应用
  - 总结了开源代码和benchmark dataset
  - 当前的challenges和research directions

{% endnote %}

<!--more-->

## Introduction

- 有很多场景中的数据是graph格式的，e-commence，用户和商品，社交网络，化学分子，引文网络
- 机器学习的一个核心假设：样本独立性，在图数据中不成立，各个节点对邻居有依赖关系
- 图卷积可以被认为是特殊的2D convolution
  - <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/CfhMIc.png" width="400" >
- 主要贡献
  > • New taxonomy We propose a new taxonomy of graph neural networks. Graph neural networks are categorized into four groups: recurrent graph neural networks, convo- lutional graph neural networks, graph autoencoders, and spatial-temporal graph neural networks.
  > • Comprehensive review We provide the most compre- hensive overview of modern deep learning techniques for graph data. For each type of graph neural network, we provide detailed descriptions on representative models, make the necessary comparison, and summarise the cor- responding algorithms.
  > • Abundant resources We collect abundant resources on graph neural networks, including state-of-the-art models, benchmark data sets, open-source codes, and practical applications. This survey can be used as a hands-on guide for understanding, using, and developing different deep learning approaches for various real-life applications.
  > • Future directions We discuss theoretical aspects of graph neural networks, analyze the limitations of exist- ing methods, and suggest four possible future research directions in terms of model depth, scalability trade-off, heterogeneity, and dynamicity.

## Background and Definition

### Background

- 1997年首先在图上用神经网络
- GNN在2005年被首次提出
- 早期是研究RecGNN，通过迭代地传播邻居信息直到收敛，得到node representation
- CNN火了之后开始有ConvGNN，分为谱域卷积和空域卷积the spectral-based approaches and the spatial-based approaches
- 近几年发展了GAEs和STCNN，图自编码器和时空图神经网络，他们都可以建立在RecGNN和ConvGNN等架构之上

1. GNN和network embedding区别：
   1. network embedding将网络表示为低维向量，并保存网络结构和节点信息，用于下游分类，聚类，推荐等任务
   2. GNN用端到端的方式解决类似的问题
   3. 区别在于，GNN包括了很多神经网络模型，能够解决network embedding同样的问题，但是network embedding还包括了一些非DL的方法，比如矩阵分解和随机游走random walk
2. GNN和graph kernel methods区别:
   1. graph kernel 是以前用于解决图分类的主要方法，都是用一些核函数为基础的方法，比如支持向量机SVM。和GNN一样会把graph和node通过核函数映射到高维空间
   2. 区别在于，核函数的方法是pair-wise的，计算量大，有计算瓶颈。并且核函数的方法是确定性的，而不是科学系的

### Definition

1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/ZZh1nn.png" width="500">
2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/FYBPOW.png" width="500">
3. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/Bmr5Di.png" width="500">

## Categorization

### 分类

1. RecGNN
   1. 最早期的GNN，目标是学习节点表示，用循环神经网络结构，他们假设节点会和邻居节点交换信息，直到一个稳定的状态
   2. 消息交换的思路被后来的空域图卷积继承
2. ConvGNN
   1. 在图上使用卷积操作。主要思路是通过aggregate当前节点和邻居节点的信息来生成节点embedding。
   2. ConvGNN会堆叠很多层来提取高维特征表示
3. GAEs
   1. 将节点或者图编码到latent vector space并且从encoded information中重构图（比如邻接矩阵），使得重构的图和原图损失最小。
   2. GAE可以用于学习network embedding和graph generative distributions
4. 时空图神经网络STGNN
   1. 主要是学习时空图的潜在特征
   2. 假定图中的信息是随时间变化的
   3. 同时考虑空间依赖和实践依赖

### 网络结构

1. node-level：输出和节点回归or分类有关，一般是提取高维特征后用MLP或者softmax分类
2. edge-level: 输出和边分类or连接预测相关。两个节点的GNN编码作为输入，用相似度函数或者NN来预测连接关系和强度
3. graph-level：输出和图分类先关的。得到一整个图的表示，经常和pooling，readout等操作结合。

### 训练模式

1. 半监督学习下的节点分类：给定单独的网络和部分标注节点，其他节点未标注。目标是识别未标注的节点。可以用stack一堆卷积层然后接softmax实现多分类
2. 监督学习下的图分类：给定真个图，预测图类别。一般会结合GCN，pooling层，readout层等。graph pooling一般用于down-sampling，每次获得部分图结构。readout用于收集节点表示然后生成graph embedding。
3. 无监督学习下的graph embedding：一般使用GAE来编码整个图。最小化重构损失。还会用负采样来得到部分节点对作为负样本，存在连接的是正样本。用逻辑回归分辨是正例还是负例样本。

### 图示

<img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/rML4yp.png" width="600">

## Recurrent GNN

1. RecGNN对图上所有的点使用相同的parameters，来提取高维节点representations
2. 早起主要受限算力，多用于有向无环图
3. 基于信息传播机制，用所有的邻居节点来更新自己节点，周期性交换，直到稳定下来
4. $h_v^{(t)} = \sum_{u\in N_{(v)}} f(x_v, x^e_{(v, u)}, x_u, h_u^{(t-1)})$
5. 初始化是随机的
6. 对所有节点做上面的操作
7. 为了确保收敛，f函数必须是**收缩性映射**，就是会缩小两个点在投影空间的距离
8. 如果f是一个NN，需要在参数的Jacobian matrix雅可比矩阵上加惩罚项。其实就是w加惩罚项。
   1. NN从n到m维的话，相当于是m个f函数
   2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/g5uRAV.png" width="500">
9. 收敛之后把最后一个时间步的hidden进行readout

- GraphESN包括了一个可训练的encoder
- Gated Graph Neural Network(GGNN)是用GRU作为recurrent function。好处是不需要约束参数一定要收敛。用BPTT进行优化。但是因为它要对所有节点进行好几次recurrent function，并且都是存在memory中，对于大图来说不友好
- Stochastic Steady-state Embedding (SSE)对大图更加友好，它只采样一个batch的接地那来更新state，并且只对一个batch的节点进行梯度计算。SSE的recurrent function是对历史状态和新状态的weighted average。$\alpha$是超参
  - <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/wpZYPJ.png" width="500">

## Convolutional GNN

1. GCN用固定层数但权重不同的层，在结构上，处理循环相互依存问题
2. 比RecGNN更加高效和方便
3. 分两类，谱域图卷积和空域图卷积
   1. Spectral based GCN，从图信号处理的角度引入滤波器，卷积操作被解释为去除图信号中的噪声
   2. Spatial based GCN，继承了RecGNN的思想，用GCN来进行信息传递
   3. 自从弥合了基于频谱的方法和基于空间的方法之间的差距以来，基于空间的方法由于其引人注目的效率，灵活性和通用性而迅速发展

### Spectral based GCN

1. 它有一整台完备的图信号处理理论基础
2. 假定是无向图
3. normalized Laplacian matrix图拉布拉斯矩阵，是无向图的数学表示，定义为 $L = I_n - D^{-\frac{1}{2}} A D ^{-\frac{1}{2}}$，其中D是节点的度矩阵，A是邻接矩阵，$D_{ii} = \sum_j (A_{i, j})$
4. 更多的拉普拉斯矩阵看[wikipedia](https://zh.wikipedia.org/wiki/%E8%B0%83%E5%92%8C%E7%9F%A9%E9%98%B5)和[别人的blog](https://www.cnblogs.com/yyl424525/p/14361357.html)
5. 性质：
   1. L是实对称正半定的，
   2. 最小特征值大于等于0，
   3. 特征值中0出现的次数就是图连通区域的个数，
   4. 最小特征值是0，因为拉普拉斯矩阵（普通形式：𝐿=𝐷−𝐴）每一行的和均为0，
   5. 最小特征值对应的特征向量是每个值全为1的向量
   6. 最小非零特征值是图的代数连通度。
6. L可以进行特征值分解，就是矩阵分解为由其特征值和特征向量表示的矩阵之积，表示成 $L = U \Lambda U^T，U=[u_0, u_1,..,u_{n-1}\in R^{n\times n}$，U是是列向量为单位特征向量的矩阵，$\Lambda$的对角线是特征值。$U^TU=I$。注意，特征分解最右边的是特征矩阵的逆，只是拉普拉斯矩阵是对称矩阵才可以写成特征矩阵的转置。
7. 在图信号处理中，图信号 $x \in R^n$是所有节点的特征向量，$x_i$是第i个节点的值。
8. 对一个信号x的，图傅里叶变换graph Fourier transform，定义为 $\mathcal{F}(x) = U^T x$，逆傅里叶变换是 $\mathcal{F}( \hat{x} ) = U \hat{x}$，其中 $\hat{x}$是图傅里叶变换的结果信号
9. 图傅里叶将输入图信号投影到正交空间，正交空间的基础上由规范化图拉普拉斯算子的特征向量形成
10. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/pottwG.png" width="500">
11. Spectral CNN假定 $g_{\theta}$是一组科学系的参数，并且认为graph signals是有多个通道的。
    1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/yJiXBV.png" width="500">
    1. 缺点是任何的对图的扰动都会导致特征基的变化，并且filter是domain dependent，不可以应用到其他结构上，最后，特诊值分解需要 $O(n^3)$的时间复杂度

#### ChebNet

1. 是对spectral cnn即第二代GCN的改进
2. 用特征值对角阵的切比雪夫多项式近似滤波器 $g_\theta$，
3. 切比雪夫多项式用的是第一类切比雪夫多项式，用递归关系得到的
   1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/KtvsJj.png" width="400">
4. 具体的还得看论文or别人的博客
5. 关于切比雪夫多项式：[wikipedia](https://zh.wikipedia.org/wiki/%E5%88%87%E6%AF%94%E9%9B%AA%E5%A4%AB%E5%A4%9A%E9%A1%B9%E5%BC%8F)
6. 主要是把卷积核用 $g_\theta = \sum_{i=0}^K \theta_i T_i(\hat{\Lambda})$表示
7. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/np24Qf.png" width="500">
8. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/HIIOvQ.png" width="500">

#### GCN

1. 用cheb net的一阶邻居近似
2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/lWZSXn.png" width="500">

#### DGCN

用normalized 邻接矩阵 $\hat{A}$和PPMI矩阵来捕获local和global的结构信息。使用两个parallel gcn。说的比较简单，想了解可以再看paper。

### Spatial based conv gnn

1. 定义在节点空间关系上
2. 图片是特殊的graph，cnn卷积核相当于是对周围的节点加权平均，gcn也是用邻居节点来更新中心节点
3. 另一个角度来看，是继承了rec gnn的消息传递的思路
4. spatial base gcn是通过边来传递节点信息的

#### Neural Network for Graphs (NN4G)

1. first work towards spatial-based ConvGNNs
2. 增量式地扩展邻居节点
3. 直接对邻居节点进行summing up
4. 应用了残差连接residual connection、skip connection来记忆化每层信息
5. 每层节点状态更新为：
   1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/Ii0ztO.png" width="400">

#### Contextual Graph Markov Model (CGMM)

1. 用马尔科夫概率模型建模

#### Diffusion Convolutional Neural Network (DCNN)

1. 认为gcn是信息传播过程，以确定性概率 $P = D^{-1}A$传播
2. $H ^{(k)} = \mathcal{f}(W^{(k)} \bigodot P^k X$
3. 如果P是固定的X也是那么H之间的差异就在W不同类似多头atten？
4. 最后拼接所有的H作为模型输出

#### Diffusion Graph Convolution (DGC)

1. 相比上面的DCNN直接对所有的H求和，而不是拼接
2. $H = \sum_{k=0}^K \mathcal{f}(P^k X W^{(k)})$

#### PGC-DGCNN

1. 结合了最短路径的思想，符合条件的为1，否则为0，代替邻接矩阵
2. 超参控制接受域

#### Message Passing Neural Network (MPNN)

1. 提出通用的spatial base gcn
2. 认为gcn是消息沿着边有向传播过程
3. 用k-step消息传播iteration，让消息传的更远
4. hk can be passed to an output layer to perform node-level prediction tasks or to a readout function to perform graph-level prediction tasks.
5. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/48ecPv.png" width="400">
6. M和U都是可训练参数

#### GIN

1. 认为MPNN第不同图结构的分辨能力不足
2. 调整了中心节点的权重
3. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/s8zUwE.png" width="400">

#### GraphSage

1. 大图上节点的邻居可能很多，所以直接把所有邻居都考虑进来很低效
2. 使用采样的方法得到固定数量的邻居
3. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/vUj7Xn.png" width="400">

#### Graph Attention Network (GAT)

1. 2017
2. 认为每个邻居对中心点的贡献是不一样的
3. 而GraphSAGE中是固定的，gcn是预先确定
4. 使用了注意力机制来学习相对权重
5. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/HTLQiY.png" width="500">

#### Gated Attention Network (GAAN)

1. 2018
2. 引入自监督机制
3. 在GAT基础上计算每个atten head的权重

- 后续还有人在此基础上用LSTM类似的门控机制来控制信息在卷积层的流动
- 也有其他的graph attention model提出，但是不属于conv gnn框架
- 也有MoNet提出用不同的方式来分配节点邻居的权重
- 后面有很多比如GCNN、ACNN等都是MoNet的基础上做的

#### 获取权重的另一种方式

1. PATCHY-SAN
   1. 对邻居排序，只选top q的邻居
   2. 只考虑结构信息来rank
   3. 然后用卷积来提取邻居特征
2. Large- scale Graph Convolutional Network (LGCN)
   1. 按照邻居的node feature information进行rank
   2. 构建一个包含所有邻居的feature matrix
   3. 对这个矩阵按照每一列排序，排序后选前q个作为中心节点的输入
   4. 没懂这个对feature matrix 排序是啥意思

#### 加快训练和提高效率

1. GraphSage
   1. It samples a tree rooted at each node by recursively expanding the root node’s neighborhood by K steps with a fixed sample size. For each sampled tree, GraphSage computes the root node’s hidden representation by hierarchically aggregating hidden node representations from bottom to top.
2. Fast-GCN
   1. Fast-GCN对每个gcn层采样固定数量的节点，而不是对每个节点采样固定数量邻居.
   2. 它将图卷积解释为概率测度下节点嵌入函数的积分变换。
   3. 用蒙特卡洛近似和方差降低技术
3. Huang
   1. 自适应的layer-wise采样
   2. 较低层的节点采样以顶层为条件
   3. 比Fast-GCN准确率更高
4. StoGCN
   1. 降低GCN接受域到一个随机的小的范围
   2. 用历史节点表示作为控制变量
   3. 缺点是要保存所有节点的中间状态，标记哦消耗内存
5. Cluster-GCN
   1. 使用图聚类算法采样子图，并在子图上进行gcn
   2. 能够适应大规模的图，以及更深的架构
6. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/wKjzfW.png" width="500">

### Comparison between Spectral and Spatial

1. spectral based
   1. 有图信号处理的理论基础
   2. 效率更低，要么需要特征值分解，要么要同时处理整个图
   3. 依赖图傅里叶变换，对图的任何扰动都会造成特征向量，就是正交空间的基向量的变化
   4. 只能处理无向图
2. spatial based
   1. 更高效，通用，灵活
   2. 可扩展性更强，对大图友好
   3. 在图的局部进行gcn，并且权重可以在不同位置和结构之间共享
   4. 能够处理multi-source的图输入，更加灵活，因为它们都可以很容易的被aggregation方法处理

### Graph Pooling

1. 问题：直接利用gcn产生的feature计算量很大，需要一个down-sampling策略
2. 主要分两种
   1. 用于降低参数量，得到一个更小的表示，避免overfit
   2. readout操作来产生graph-level表示
3. 最常用还是mean，max，sum pooling
4. Henaff提出在网络开始加一个简单的max/mean pooling对降维很重要
5. 有人提出用attention机制提升mean/sum pooling
6. sum pooling
   1. 可能会导致embedding不够好，因为它无视了graph size生成同样固定大小的embedding。
   2. Vinyal提出用lstm提取顺序信息到embedding，在reduce之前，这样会让memory的大小随着输入变大
7. Defferrard在了ChebNet中提出重排列图节点的方法。
   1. 先粗化得到多个level的，
   2. 然后和输入节点一起重排得到一个平衡二叉树。
   3. 对这个二叉树从底向上进行aggregate，会把相似节点arrange到一起。
   4. 对重排后的信号进行pooling会比直接pooling更高效
8. DGCNN中提出一个类似的SortPooling策略
   1. 也是节点重排序后pooling。
   2. 按照节点在图的结构上的role排序
   3. 认为spatial graph得到的节点特征是连续的，并按照这个排序节点
   4. 调整graph size到q，多的删掉，少的补0
9. DiffPool
   1. 对比前面的强调了结构信息
   2. 可以生成图的层次化表示
   3. 不是简单的聚集节点信息，而是通过在第K层学习一个聚类分配向量 $S^{(k)} \in R^{n_k \times n_{k+1}}$，$n_k$是第k层节点数
   4. 这个S中的概率值是通过节点特征和拓扑结构得到的：
   5. $S^{(k)}=\mathcal{softmax}(\mathcal{Conv}GNN_k(A^{k},H^{k}))$，这个ConvGNN可以是任何标准的GCN，核心思想是学习一个综合了拓扑结构和特征的assignment
   6. 缺点是在pooling之后生成了dense graphs，会导致计算复杂度变成 $O(n^2)$
10. SAGPool
    1. 用self-attention的方式
    2. 考虑node features和graph topology
    3. 相当于就是把上面的gcn换成attention

### 理论层面的讨论分析

1. receptive field的大小
   1. 接受域是能够对当前节点最后的node embedding产生影响的节点集合
   2. stack多个spatial gcn的时候，会导致这个receptive field越来越大，每次向更远的邻居增长一步
   3. spatial gcn层数是有限的，最终结果就是，每个节点的接受域都覆盖了所有节点，这时候，一个gcn就可以提取整个全局信息
2. VC维
   1. VC维是衡量模型复杂度的，定义成模型可以shatter的最多的点的数量
   2. GNN的VC维方面的研究比较少
   3. Scarselli等推出，如果用sigmoid或者切线双曲线激活，GNN的VC维是 $O(p^4n^2)$，n是节点数，p是模型参数
   4. 如果是分段多项式激活函数，即使 $O(p^2n)$
   5. 模型复杂度会随着p和n的数量上升飞快
3. 图同构，Graph isomorphism
   1. 如果两个图的拓扑结构是相同的，就是同构的
   2. 如果两个图可以通过WL测试，得到不同的embedding，就是不同构的
   3. 被证明，如果GNN可以将两个图映射到不同的embedding，那这两个图就可以通过WL测试，证明是不同构的
   4. 被证明，如果aggregation和readout方法是内射函数，gnn就可以实现和WL test一样的效果
   5. Weisfeiler-Lehman (WL) test，参考[什么是Weisfeiler-Lehman(WL)算法和WL Test？](https://zhuanlan.zhihu.com/p/90645716)
4. 等方差和不变形
   1. 在node level任务上，GNN必须是等方差的，并且在graph-level任务上，必须是不变函数
   2. GNN的组件必须对节点顺序不变
   3. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/0o8y9o.png" width="500">
5. Universal approximation，通用逼近
   1. 一个hidden layer的MLP可以近似任何Borel measurable functions
   2. GNN的近似能力还没有相关研究
   3. 有研究表明，RecGNN可以近似以任何精度保留展开等价性的任何函数
   4. 被证明，一个不变图网络可以近似图上定义的任意不变函数

## GAE

1. GAE是将节点映射到latent feature space，并且将其decode为graph information的深度网络架构
2. 可以用于学习网络embedding，或者生成新图
3. 主要的GAE如下
4. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/Zrwv2c.png" width="500">

### Network Embedding

1. network embedding是node的低维向量表示，保存了节点的拓扑信息
2. 用encoder提取网络信息进行encode，然后用decoder让网络保存graph拓扑信息，比如PPMI和邻接矩阵
3. DNGR
   1. 用 stacked denoising autoencoder来encode
4. SDNE
   1. 用stacked autoencoder来保存接地那first-order proximity and second-order proximity
   2. 对encoder和decoder用两个损失函数
   3. encoder的损失函数是minimize节点embedding和邻居embedding的距离
   4. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/ky7lyS.png" width="400">
   5. 第二个损失函数确保embedding保存节点second-order proximity节点二阶接近度，minimize节点输入和重构输入之间的distance
   6. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/SLG6Ne.png" width="400">
5. DNGR和SDNE只考虑了节点关于连接的结构信息，忽略了节点可能包含描述节点本身属性的特征信息
6. GAE based on GCN
   1. 用GCN来encode节点结构信息和feature信息
   2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/nsbTxn.png" width="500">
7. (VGAE) Variational graph auto-encoders
   1. 认为简单重构图的邻接矩阵会导致过拟合
   2. VGAE是GAE变种，用于学习数据分布
   3. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/qcpSsV.png" width="500">
8. (ARVGA)Adversarially Regularized Variational Graph Autoencoder
   1. 用GAN的训练模式，就是discriminator和generator进行competition，discriminator要努力分辨出generator生成的假样本
   2. ARVGA训练一个encoder来产生经验分布 $q(Z|X,A)$，尽量和先验分布 $p(Z)$相同
9. GraphSage
   1. 和GAE相似，用两层gcn来encode节点特征
   2. 不是优化重构损失，而是认为可以通过负采样的方式来保存两个节点之间的相关性信息
   3. ？<img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/cRUEl8.png" width="500">
   4. 这个损失函数强迫距离近的节点有相似的embedding，距离远的节点有不相似的embedding
10. DGI
    1. 通过最大化local互信息来驱动local embedding捕获全局结构信息。
    2. 比GraphSAGE提升很多
11. 对于上述方法，他们本质上是通过解决link预测问题来学习网络嵌入的。
12. 网络稀疏性问题
    1. 为了减轻学习网络嵌入中的数据稀疏性问题，有人通过随机排列或随机游走将图转换为序列。
    2. 变成序列之后，很多dl的模型都可以用上了
13. DRNE，Deep Recursive Network Embedding
    1. 假定节点embedding，需要近似邻居embedding的aggregation
    2. 用lstm进行aggregation
    3. 使用lstm训练embedding而不是直接用lstm生成embedding，避免了LSTM网络对于节点序列的排列不是不变的问题
14. NetRA，Network Representations with Adversarially Regularized Autoencoders
    1. 提出了一个通用的encoder-decoder结构，并带有通用loss function
    2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/CShiaF.png" width="500">
    3. dis是node embedding z和重构的z之间的distance
    4. encoder和decoder都是带有随机游走的lstm，输入是从每个节点开始的随机游走seq
    5. 和ARVGA一样，用一个对抗训练的prior来正则化，想起了DAEGC那个里面也是用类似的思想训练P分布和Q分布，但是不是对抗训练
    6. 虽然无视了node permutation variant问题，但是结果是可以的

### Graph Generation

1. GAE可以通过encode-decode得到图的generative distribution，想起来这个也被用在了DAEGC中
2. 很多用GAE来生成graph generation的都被用来解决分子图生成问题，在药物发现里面有很大实用价值
3. 一般生成新的网络分两种，一是sequential的一个个生成，另一个是global的一次性生成

#### Sequential manner

1. 一步步的生成节点和边
2. SMILES
   1. 把生成过程建模为string representation of molecular graphs
   2. 用CNN和RNN作为encoder和decoder
   3. 是domain-specific的，可以迭代将节点和边迭代添加到增长的图直到满足特定条件，替代解决方案适用于一般图
3. DeepGMG
   1. 假定图的可能性是所有节点permutation的sum，$p(G) = \sum_\pi p(G, \pi )$，其中 $\pi$是一个节点ordering
   2. 生成过程是做一系列决策，决定是否添加节点和边以及连接到哪个节点上
   3. 每步的决策是依赖node state 和生成的图的graph state，用RecGNN
4. GraphRNN
   1. 用node级别和graph级别的RNN来建模生成过程
   2. graph级别的RNN每次添加一个node进node seq，node级别的决定新节点和哪些节点相连，就是一串binary seq

### Global manner

1. 就是一次生成整个graph
2. GraphVAE，Graph Vari- ational Autoencoder
   1. 将存在的节点和边建模成独立的随机变量
   2. 假定encoder定义的后验分布q，还有decoder定义的generative distribution p，p分布是高斯先验分布，GraphVAE优化变分下界
   3. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/ssh1X3.png" width="500">
3. RGVAE，Regularized Graph Variational Au- toencoder
   1. 在GVAE的基础上，进一步对变分自编码器增加有效性约束，来regularize decoder的输出分布
4. MolGAN，Molecular Generative Adversarial Network
   1. 结合了convGNN、GAN、强化学习目标来生成带有所需特点的图
   2. 主要是为了提高generator的真实性authenticity
   3. 生成器输出fake graph和feature matrix，discriminator来分辨fake sample
   4. 还加了一个和discriminator并行的reward network来让生成器获得特定的property
5. NetGAN结合LSTM和GAN和随机游走方法
6. 总结来说，sequential manner可能会丢失结构信息，而global manner一次性产生整个图，对大图不友好。

## STGNN

暂时用不到，先留着后面再看

## Applications

### Dataset

1. 引文网络
2. 生物图网络
3. 社交网络
4. 其他

![alt](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/gpXhMJ.png)

### Evaluation and Implementations

1. node classification
   1. 大多方法把benchmark data分为train、valid和test
   2. 用在test上的多次运行结果的acc和F1作为评价指标
   3. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/sUzsrL.png" width="600">
2. graph classification
   1. 一般是hi十折交叉验证
   2. 很多实验的setting比较模棱两可，并且是很多在模型选择和模型评估的时候可能都有问题
   3. 常见的问题是，每折的外部测试集都用于模型选择和风险评估
   4. 可以用十折cv来进行模型选择，或者用两个cv，一个内部的k折进行模型选择一个外部的k折进行评估
3. 开源实现
   1. 安利了[pytorch geometric](https://github.com/rusty1s/pytorch_geometric)和[DGL](https://www.dgl.ai/)两个图方面的python库

### 实际应用

1. node classification
2. graph classification
3. network embedding
4. graph generation
5. spatial-temporal graph forecasting
6. node clustering
7. link prediction
8. graph partitioning

#### CV中应用

![alt](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/FOw2K1.png)

#### NLP中应用

<img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/nGQPPN.png" width="600">
<img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/nGzlcr.png" width="600">

#### 其他

还有在Traffic、Recommender system和Chemistry中的应用

## Future Directions

1. Model depth
   1. 现在的研究依赖深度网络结构
   2. 有研究表明gcn层数越来越高的时候，模型表现反而下降
   3. 在有限数量的gcn层里，所有节点的representation都会converg到一个单独的node
   4. 所以深度网络结构在gnn上是否是一个好的策略还需要验证
2. scalability trade-off
   1. GNN的可伸缩性是以破坏图形完整性为代价的。
   2. 无论使用抽样还是聚类，模型都会丢失部分图形信息。抽样会丢失一些有影响的节点，聚类会损失一些独特的结构
   3. 需要在scalability和integrity之间trade off
3. **Heterogenity异质性**
   1. 当前的大多数GNN都采用同质图。很难将当前的GNN直接应用于异构图，该图形可能包含不同类型的节点和边，或不同形式的节点和边输入，例如图像和文本。 因此，应开发新的方法来处理异构图
4. **Dynamicty动态性**
   1. 实际场景中图应该是动态变化的，比如节点或者边会新增或者删减，节点输入和边输入也可能随时间变化。
   2. 需要新的graph convolutions来适应这种图的动态特性
   3. STGNN可以解决部分动态图，但是很少有考虑如何在dynamic spatial relation中使用graph convolutions。

## Significant Reference

待补充