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
  - <div style="width:400px;word-wrap:break-word;white-space:normal"><img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/CfhMIc.png" width="400" ></div>
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
  - <div style="width:500px;word-wrap:break-word;white-space:normal"><img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/wpZYPJ.png" width="500"></div>

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

### ChebNet

1. 是对spectral cnn的改进
2. 用特征值对角阵的切比雪夫多项式近似滤波器 $g_\theta$，
3. 切比雪夫多项式用的是第一类切比雪夫多项式，用递归关系得到的
   1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/KtvsJj.png" width="400">
4. 具体的还得看论文or别人的博客
5. 关于切比雪夫多项式：[wikipedia](https://zh.wikipedia.org/wiki/%E5%88%87%E6%AF%94%E9%9B%AA%E5%A4%AB%E5%A4%9A%E9%A1%B9%E5%BC%8F)
6. 看不下去了，待后面补充

### GCN



## GAE

## STGNN

## Applications

## Future Directions

## Conclusion

## Significant Reference