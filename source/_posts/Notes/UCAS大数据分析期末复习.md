---
title: UCAS大数据分析
tags:
  - UCAS
  - BigData
  - Review
  - Notes
categories:
  - Notes
comments: true
mathjax: true
date: 2020-12-08 23:34:18
urlname: ucas-big-data-analysis-review
---

<meta name="referrer" content="no-referrer" />

{% note info %}

国科大大数据分析期末复习笔记。

垃圾大数据，谁选谁倒霉。早点退选课，改为非学位。

{% endnote %}

<!--more-->
## 目录
<!-- TOC -->

- [目录](#目录)
- [简介和分布式计算](#简介和分布式计算)
- [统计分析](#统计分析)
  - [相关性分析](#相关性分析)
  - [找相似文档](#找相似文档)
    - [整体流程](#整体流程)
    - [Shingling](#shingling)
    - [Jaccard similarity](#jaccard-similarity)
    - [Min Hashing 最小哈希](#min-hashing-最小哈希)
    - [最小哈希签名](#最小哈希签名)
    - [LSH 最小敏感哈希](#lsh-最小敏感哈希)
    - [Count-Min Sketch CMS](#count-min-sketch-cms)
- [机器学习](#机器学习)
  - [SVD 奇异值分解](#svd-奇异值分解)
  - [PCA](#pca)
  - [Sketch Linear Regression](#sketch-linear-regression)
  - [Decision Tree](#decision-tree)
  - [Random Forest](#random-forest)
  - [Gradient Boost Decision Tree(GBDT)](#gradient-boost-decision-treegbdt)
  - [Deep Learning](#deep-learning)
- [数据驱动的自然语言处理](#数据驱动的自然语言处理)
  - [语法分析](#语法分析)
    - [词法分析-LSTM](#词法分析-lstm)
    - [句法分析](#句法分析)
  - [语义分析](#语义分析)
    - [语义表示](#语义表示)
  - [内容分析](#内容分析)
    - [信息抽取](#信息抽取)
    - [文本分类](#文本分类)
    - [情感分析](#情感分析)
      - [情感词抽取](#情感词抽取)
      - [情感词极性判定](#情感词极性判定)
      - [句子级情感分析](#句子级情感分析)
      - [篇章级情感分析](#篇章级情感分析)
      - [属性级情感分析](#属性级情感分析)
  - [机器翻译](#机器翻译)
  - [问答系统](#问答系统)
- [文本大数据分析](#文本大数据分析)
  - [单词表示方法](#单词表示方法)
    - [隐性语义索引LSI](#隐性语义索引lsi)
    - [概率隐性语义索引PLSI](#概率隐性语义索引plsi)
    - [排序学习模型C&W](#排序学习模型cw)
    - [上下文预测模型Word2Vec](#上下文预测模型word2vec)
    - [全局上下文模型Glove](#全局上下文模型glove)
  - [句子的表示方法](#句子的表示方法)
    - [词袋模型Bag of Words](#词袋模型bag-of-words)
    - [TF-IDF模型](#tf-idf模型)
  - [文本匹配问题](#文本匹配问题)
    - [基于启发式规则](#基于启发式规则)
    - [基于隐语义表达](#基于隐语义表达)
    - [基于学习](#基于学习)
    - [评价方法](#评价方法)
- [知识图谱与知识计算](#知识图谱与知识计算)
  - [实体抽取](#实体抽取)
    - [基于机器学习的方法](#基于机器学习的方法)
      - [隐马尔科夫模型](#隐马尔科夫模型)
      - [条件随机场CRF](#条件随机场crf)
      - [线性链条件随机场模型](#线性链条件随机场模型)
    - [关系抽取](#关系抽取)
      - [远程监督关系抽取](#远程监督关系抽取)
      - [实体对齐](#实体对齐)
      - [实体链接](#实体链接)
      - [知识推理](#知识推理)
      - [基于分布式表达的知识计算](#基于分布式表达的知识计算)
      - [基于分布式表达的推理TransE](#基于分布式表达的推理transe)
- [大图挖掘与分析](#大图挖掘与分析)
  - [patterns](#patterns)
  - [检测密集子图](#检测密集子图)
    - [算数平均度数和几何平均度数](#算数平均度数和几何平均度数)
    - [算法](#算法)
    - [理解](#理解)
    - [证明$g(X_{best})> \frac{1}{2}g(X_{opt})$](#证明gx_best-frac12gx_opt)
  - [EigenSpoke 最密子图检测](#eigenspoke-最密子图检测)
  - [HoloScope](#holoscope)
  - [D-cube](#d-cube)
  - [EigenPulse](#eigenpulse)
    - [问题定义](#问题定义)
    - [Sketching 和 AugSVD](#sketching-和-augsvd)
- [社交媒体分析](#社交媒体分析)
  - [PageRank](#pagerank)
    - [矩阵表示](#矩阵表示)
    - [Power iteration](#power-iteration)
    - [存在的问题](#存在的问题)
      - [dead ends](#dead-ends)
      - [Spider traps](#spider-traps)
    - [Teleports](#teleports)
    - [Matrix Formulation](#matrix-formulation)
    - [Sparse Matrix Encoding](#sparse-matrix-encoding)
    - [实际迭代更新](#实际迭代更新)
  - [其他](#其他)
- [题型](#题型)

<!-- /TOC -->

## 简介和分布式计算

1. 什么是大数据：
   1. 技术能力的视角：指**规模超过现有数据处理工具获取、存储、管理和分析能力**的数据集。并不是超过某个特定数量级的数据集才是大数据。
   2. 大数据内涵的视角：是具备**海量、高速、多样、可变**等特征的多维数据集，需要通过**可伸缩的体系结构**实现高效的存储、处理和分析。
2. 大数据基本特征：
   1. 体量巨大
   2. 速度极快
   3. 模态多样
   4. 真伪难辨
3. 和物联网、互联网、云计算、人工智能关系
   1. 来源：物联网和互联网
   2. 核心硬件和软件支撑：云计算
   3. 驱动新应用：人工智能
   4. 大数据为人工智能提供必要燃料，二者缺一不可
4. 大数据分析：
   1. ![EXF4HM](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/EXF4HM.png)
   2. ![fzTko1](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/fzTko1.png)
5. 数据指数增长
   1. 因此$O(n^2)$的算法是不可解的
6. 可扩展算法 scalable
   1. $s = scalability_A(n) = \frac{T_A(n)}{n} = O(log^c n)$
   2. c=0, 则为线性可扩展 linear-scalable
   3. $s =o(1)$，则 A 为 super-scalable，即复杂度比 linear 小，如$n^{0.8}, n^{0.9}, logn$
7. 数据分析算法和传统算法
   1. 数据分析算法
      1. 优化为中心
      2. 多轮迭代直到收敛![yMV1SK](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/yMV1SK.png)
      3. 算法目标为最优，没有确定答案，最优点往往无法达到
   2. 传统算法
      1. 以操作为中心
      2. 是确定性算法
      3. 目标是正确答案，有一个或多个
8. 参数更新方案
   1. 同步更新  
   ![U3PcWY](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/U3PcWY.png)
   1. 异步更新  
   ![uzs4Qk](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/uzs4Qk.png)
   1. 半同步更新  
   ![CFvoUi](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/CFvoUi.png)
9. Map Reduce
   1. 大体介绍![xCWlNp](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xCWlNp.png)
   2. map 接受 document name 和内容，输出中间键值对，然后 reduce 接受一个 word 作为 key 和一个 values 集合，输出每个 key 的 reduce 结果
10. cost measures 成本计算?
    1. total communication cost，总体通信代价，就是所有进程的 IO 代价，和交换的 size 有关，
    2. elapsed communication cost，最大的 IO 路径，就是最慢的 map 进程+最慢的 reduce 进程
    3. computation cost，计算代价和上面的通信代价类似，但是只计算进程运行时间，而不是 IO 时间
    4. elapsed computation cost，和上述类似
    5. 以 map reduce 为例：![tY9EM1](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/tY9EM1.png)
    6. 通信时间和 IO 时间其中有一个会占主导，就可以忽略另一个
11. 什么导致大数据现象？是：信息技术的不断廉价化和互联网及其延伸带来的无处不在的信息技术应用。四个驱动：
    1. 摩尔定律驱动的计算速度指数增长，成本指数下降模式；
    2. 技术低成本化驱动的万物数字化；
    3. 宽带、移动、泛在互联驱动的人机物广泛联接；
    4. 云计算模式驱动的数据大规模汇聚
12. 大数据是存储、算力、网络、智能发展的产物

## 统计分析

### 相关性分析

1. 折线图
2. 协方差矩阵
3. 散点图
4. 互信息
   1. ![PuTNN3](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/PuTNN3.png)
   2. ![e0m7QW](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/e0m7QW.png)

### 找相似文档

Shingling-Min-Hashing-Locality-Sensitive Hashing (LSH)

[参考博客](https://blog.csdn.net/liujan511536/article/details/47729721)

#### 整体流程

![vPzLxF](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/vPzLxF.png)

#### Shingling

1. 对大文档进行切分，得到 k-shingle/k-gram，即文档中长度为 k 的任意字符子串，然后对这些 k-shingle 进行 hash，因为原始的 k-shingle 占内存太大了，如果原文档是 100k，那么 4-shingle 保存下来就是 400k，因此要进行 min Hashing
2. 文章内容为 abcdabd，令 k=2，那么这篇文章中所有的 2-shingle 组成的集合为{ab,bc,cd,da,bd}，需要注意的是，ab 在文章中出现了两次，但是在集合中只出现一次，这是因为集合中不能有相同的元素。

#### Jaccard similarity

1. 交集除以并集
2. 用来描述两个集合相似度

#### Min Hashing 最小哈希

1. 首先对特征矩阵的行进行打乱（也即随机调换行与行之间的位置），这个打乱是随机的
2. 然后某一列的最小哈希值就等于打乱后的这一列第一个值为 1 的行所在的行号
3. 经过行打乱后的两个集合计算得到的最小哈希值相等的概率等于这两个集合的 Jaccard 相似度，证明略

#### 最小哈希签名

1. 用一个行打乱来处理特征矩阵，然后就可以得到每个集合最小哈希值，这样多个集合就会有多个最小哈希值，这些值就可以组成一列。
2. 对于矩阵 S，用多个随机行打乱（假设为 n 个，分别为 h1,h2…hn）来处理特征矩阵时，然后计算最小哈希值，得到 n 个最小哈希值，组成一个列向量
3. 如果有 m 个集合，就会有 m 个列向量，每个列向量中有 n 个元素。把这 m 个列向量组成一个矩阵，这个矩阵就是特征矩阵的签名矩阵；
4. 签名矩阵的列数与特征矩阵相同，但行数为 n，也即哈希函数的个数
5. 通常来说，n 都会比特征矩阵的行数要小很多，所以签名矩阵就会比特征矩阵小很多
6. 计算最小 hash 签名有更简单的法子
   1. 初始化$S=[n,m]$的矩阵为 inf，第 i 行表示第 i 个 hash 函数，第 j 列表示第 j 个特征矩阵。$S[i,j]$表示了第 j 个矩阵在第 i 个 hash 函数下的 min hash
   2. 遍历原始矩阵的每一行，对每个值为 1 的列，更新 S 矩阵的对应列：比较原始行的 hash 值和当前 S 保存的 hash 值，存储最小的那个。就是新出现的这个可能是这个文档的 min hash
   3. 具体看上面的参考连接

#### LSH 最小敏感哈希

1. 依次两两比较签名矩阵的相似度，缺点是当文档数量很多时，要比较的次数会非常大
2. 在 MinHash 基础上，得到了 n\*m 的哈希签名矩阵
3. 对这个矩阵按行划分成 b 个部分（band），每个部分有 r 行
4. 存在一个哈稀函数把每个 band 中的列向量映射到一个桶（bucket）中，不同部分使用不同桶
5. 这意味着，在同一个 band，如果两列的哈希值相同，则这两列是一样的，这两个集合也被认为是相似的，就当做候选集合
6. 对剩下的 band 继续进行哈希，找在同一个桶中的列
7. 对选出来的候选集合（candidate pair）进行比较，其他的忽略
8. 如果在两列在所有的 band 中都不相似，就是非候选集
9. LSH 适合用来计算相似度超过某个值的文档相似度，而不适用于计算所有文档的相似度，因为那些相似度很低的文档已经被直接忽略了
10. 计算遗漏某两个相似文档的概率![expDGL](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/expDGL.png)

#### Count-Min Sketch CMS

1. 主要思路是把通过 hash，使用很少的数据描述全部数据的特性，牺牲来一定准确性
2. 参考链接：[1](https://zhuanlan.zhihu.com/p/84688298)，[2](https://zouzhitao.github.io/posts/count-min-sketch/)
3. 数据是 n 维，随着时间 t 不断变化的，$a(t)=[a_1(t), ....a_n(t)]$
4. 使用 d 个 hash 函数，将其变成$d*w$维，w 是 hash 的值域范围
5. $a_i(t)$是编号为 i 的值出现的次数，每次更新都在修改这个计数器
6. 更新操作是将 hash 的结果指加上要更新的值
   1. $a_{i_t} (t)= a_{i_t}(t-1) + c_t$
   2. $a_{i'}(t) = a_{i'}(t-1)，i' \ne i_t$
   3. 每次更新相当于算了 d 个 hash 值，对应到 d 行，然后在对应的值那边加一
   4. ![pwsKOg](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pwsKOg.png)
7. 查询某个值的操作是，对这个值进行 d 次的 hash，取得对应位置的值，返回其中的最小值。![D81QUq](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/D81QUq.png)
8. w 和 d 的取值问题，$w=\left\lceil\frac{2}{\varepsilon}\right\rceil, d=\left\lceil\ln \frac{1}{\delta}\right\rceil$，意义是在 $1-\delta$ 的概率下，总误差（所有元素查询误差之和）小于$\varepsilon ||a||_1$

## 机器学习

### SVD 奇异值分解

1. 对于作业 1 中的问题，证明如下：![Xq4vgO](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Xq4vgO.png)
2. $A \approx U\sum V^T = \sum_i \sigma_i u_i \circ{v_i^T}$，这里的 u 和 v 分别是左奇艺向量和右奇异向量，$\sigma$是奇异值
3. 左奇艺向量 U 是通过计算$A A^T$得到的特征向量，每一列是一个特征向量
4. 右奇异向量 V 是通过计算$A^T A$得到的特征向量
5. 奇异值可以通过$\sigma_i = \sqrt{\lambda_i}$得到
6. 计算举例：![yWIw4k](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/yWIw4k.png)

### PCA

1. PCA 是为了计算协方差矩阵$C=\frac{1}{m}XX^T$的特征值和特征向量
2. 转换到 SVD 上面，就是令$A=\frac{X^T}{\sqrt{m}}$，则$A^TA = \left(\frac{X^T}{\sqrt{m}}\right)^T\frac{X^T}{\sqrt{m}}=\frac{1}{m}XX^T$，这样就可以通过 SVD 进行 PCA 降维了
3. 注意到 PCA 仅仅使用了我们 SVD 的左奇异矩阵，没有使用到右奇异值矩阵，那么右奇异值矩阵有什么用呢？**左奇异矩阵可以用于对行数的压缩；右奇异矩阵可以用于对列(即特征维度)的压缩**。
4. 使用 SVD 的另一个好处：有一些 SVD 的实现算法可以先不求出协方差矩阵也能求出我们的右奇异矩阵 V。这个方法在样本量很大的时候很有效。实际上，scikit-learn 的 PCA 算法的背后真正的实现就是用的 SVD，而不是特征值分解

### Sketch Linear Regression

1. 缩写介绍：$A:[n, d], x:[d, 1], b:[n, 1], S:[k,n]$，A 是输入，x 是 weight，b 是对应 label
2. 大体思想：参考了 SVD 的思路，用矩阵$S\circ A$和$S\circ b$对 A 和 b 进行了降维
3. 原始目标是：$min_x|Ax-b|_2$
4. 现在的目标是： $x' s.t. |Ax'-b|_2 \leq (1+\epsilon)min_x|Ax-b|_2$，相当于对原始的线性回归进行了松弛
5. 做法是：找一个$k\times n$的矩阵 S，计算$SA$和$SB$，使得$min_{x'}|(SA)x-(Sb)|_2$，$x' = (SA)^-Sb$
6. 如何选择这个矩阵 S 呢？
   ![DCftE3](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/DCftE3.png)也就是令$k=O(\frac{d}{\epsilon^2})$，这里的$\epsilon$就是允许的和 label $b$的误差范围，并且 S 是非常稀疏的，因此只要每列随机设置一个数非 0（比如 1 或-1）就行
7. 时间复杂度：$T(S\circ A) = nnz(A)<<nd<<nd^2$，其中$nnz(A)$是 A 中非零元素个数

### Decision Tree

1. 计算信息熵：
   $$ H(S) = - \sum_{c=1}^Cp(c)\log p(c) $$
2. 选择信息增益最大的节点划分子树，具体看数据挖掘的部分
3. Information Gain：
   - 全集 S 中有 p 个 P 种类和 n 个 N 种类，原本就按照**属性 F1**这个分两类
   - 则经验熵：
   - $$I(p,n) = -\frac{p}{p+n}log_2 \frac{p}{p+n}-\frac{n}{p+n} log_2 \frac{n}{p+n}$$
   - 若这时训练集按照**属性 F2**被分为${S_1, S_2, ..., S_v}$，每个$S_i$包含$p_i+n_i$个样本，则条件熵为：
   - $E(F2) = \sum_{i=1}^v \frac{p_i+n_i}{p+n}I(p_i,n_i)$
   - 信息增益为：
   - $Gain(F2) = I(p,n)-E(F2) = 经验熵-条件熵$
   - ![ifVlWa](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ifVlWa.png)
   - **选信息增益最大的**

### Random Forest

1. 建立多棵树
2. 每棵树 sample 自己的数据集$S_i$和特征集$D_i$
3. 最后投票决定结果或者 average
4. 好处是避免过拟合，提高稳定性和准确率

### Gradient Boost Decision Tree(GBDT)

1. 这个参考链接讲的比较清楚：[GBDT：梯度提升决策树](https://www.jianshu.com/p/005a4e6ac775)
2. 简单理解就是 GBDT 是回归树（调整后才可以做分类），然后下一棵树拟合的是之前的负梯度，简单理解成残差。原本不做 boost 拟合的就是原本的 label 标签。最后的结果就是所有树的和。
3. 举个栗子：
   - ![Pr63Qt](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Pr63Qt.png)
4. 可见，$y_i - f(x_i)$只是一种特例
   - ![6u68Nc](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/6u68Nc.png)

### Deep Learning

1. 多层 NN 计算：这里看数据挖掘的那部分比较清楚
2. 训练 NN：
   1. 模型选择
   2. 训练集和验证集
   3. early stopping
   4. mini-batch
   5. momentum：一定程度上保留前几次的更新方向，并用当前 batch 的方向微调

## 数据驱动的自然语言处理

### 语法分析

#### 词法分析-LSTM

中文以字为基本书写单位，词语之间没有明显的区分标记。

**中文分词：** 由计算机在文本中词与词之间加上标记

英文分词比较容易，识别出文本中词：tokenization；对识别出的词进行还原，如boys -> boy：lemmatization

LSTM的细胞结构：
- ![oPj8jz](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/oPj8jz.png)
- ![xbk1q1](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xbk1q1.png)
- 输入门：
  - 更新细胞状态，或认为是控制对输入的接收程度
  - 先对前一隐藏状态信息$h_{t-1}$和当前输入信息$x_t$做非线性tanh变换，得到新的输入 $\tilde{c_t} = tanh(W_c[h_{t-1},x_t] + b_c)$
  - 然后将前一个隐藏状态 $h_{t-1}$和当前输入$x_t$输入sigmoid得到输入门门控向量：$g_i = \sigma (W_i[h_{t-1,x_t}]+b_i)$
  - 最终，待写入记忆的向量为：$g_i \bigodot \tilde{c}_t$
- 输出门：
  - 用于确定写一个隐藏状态的值
  - 计算$g_o = \sigma (W_o[h_{t-1}, x_t]+b_o)$
  - 将细胞状态通过tanh：$tanh(c_t)$
  - 得到输出：$h_t = g_o \bigodot tanh(c_t)$
  - $y_t = \sigma ( W'[h_t])$
- 遗忘门：
  - 决定应丢弃或保留哪些信息，作用于LSTM的状态向量c上，用于控制上一时间戳的记忆$c_{t-1}$对当前时间戳的影响。
  - 前一隐藏状态的信息$h_{t-1}$和当前输入信息$x_t$同时传递到sigmoid函数中去，得到遗忘门门控向量。
  - $g_f = \sigma (W_f[h_{t-1}, x_t] +b_f)$
  - 当前状态向量为$g_f \bigodot c_{t-1}$
  - 更新细胞状态为：$c_t = g_f \bigodot c_{t-1} + g_i \bigodot \tilde{c}_t$

基于LSTM的中文分词：
1. 首先将输入的句子中的每个字都映射成向量表示
2. 分别使用四种不同的LSTM+移动窗口的结构提取特征
   - 只用一层lstm
   - 只使用双层lstm
   - 使用一层lstm，并且联接窗口内LSTM单元输出
   - 使用双层LSTM，并且联接窗口内顶层LSTM单元输出
   - 可能考模型结构：![VivxXy](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/VivxXy.png)
3. 通过标签推理层(LSTM层的输出H作为输入)输出每个字对应的标签
   - 输出得分最高的一组句子标注：
   - $$s(c^{(1:n)}, y^{(1:n)}, \theta ) = \sum_{t=1}^n (A_{y^{t-1}y^{(t)}}+y_{y^{(t)}}^{(t)})$$

#### 句法分析

略

### 语义分析

略

#### 语义表示

略

### 内容分析

略

#### 信息抽取

略

#### 文本分类

文本分类是在自然语言理解中一项基础的任务，是指将一段文本划分到预定义的标签类别中。文本分类的效果会影响到一些下游任务，如问答系统，自然语言推理等。

![KZYtRd](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/KZYtRd.png)

**TextCNN：**

- 在由单词的分布式表示组合成的句子表示上，卷积神经网络可以利用过滤器捕捉局部特征
- 基于卷积神经网络的模型在其他一些传统自然语言处理任务上都取得了不错的效果

基本结构：  
embedding -> convolution -> max pooling -> fully connetcion -> softmax

![aANma2](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/aANma2.png)

#### 情感分析

情感分析是一种重要的信息 组织方式，其研究目标是自 动挖掘和分析文本中的**立场 、观点、看法、情绪和喜恶**等主观信息；

情感分析在社会管理、商业决策、信息预测等各个方面有着广泛而重要的应用价值

##### 情感词抽取

1. 基于启发式规则：
   - 利用连词，and、but、either or；利用副词，very；利用评价对象词，等等
   - 优点是：简单、针对性强
   - 缺点是：人工定义的规则有局限性，可扩展性差
2. 基于统计的方法：
   - 话题模型
   - 图模型
3. 基于**统计的方法**抽取出的情感词**召回率高**，而基于**规则的方法**抽取出的情感词**准确率高**，因此在实际工程任务中，往往采用规则和统计相 结合的方法

##### 情感词极性判定

![VoiTmO](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/VoiTmO.png)

##### 句子级情感分析

![wJutyu](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wJutyu.png)

##### 篇章级情感分析

![u7hknA](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/u7hknA.png)

##### 属性级情感分析

![wLZbVa](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wLZbVa.png)

### 机器翻译

略

### 问答系统

略

## 文本大数据分析

### 单词表示方法

1. 局部性表示
   1. one hot独热表示
   2. 在将单词表示为向量时，每个单词使用向量中**独有且相邻的维度**。在这种表示下，**单词之间是相互独立的**
2. 分布式表示
   1. 将单词映射到特征空间中，每个单词由刻画它的多个特征来高效表示；在形式上使用稠密实数向量（向量多于一个维度非0，通常为低维向量）来表示单词。
   2. **分布式表示都是利用某种上下文的统计信息来学习单词的分布式表示，可以编码不同单词之间的语义关联**
   3. 横向组合关系：隐性语义索引LSI，概率隐性语义索引PLSI，隐性狄利克雷分析LDA
   4. 纵向聚合关系：神经网络概率语言模型NPLM，排序学习模型C&W，上下文预测模型Word2vec，全局上下文模型Glove。
   5. ![pyeJoc](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pyeJoc.png)
3. 分布式语义假设：单词的语义来自其上下文context。

#### 隐性语义索引LSI

1. 词项-文档矩阵，$c_{ij}$表示第i个单词在第j篇文档中的出现次数。
2. 对C进行SVD分解，保留k个奇异值
3. 缺点是：不是概率模型，缺乏统计基础，难以直观解释，且K值对结果的影响大，很难选择合适的k值。
4. ![eVfa3i](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/eVfa3i.png)
5. ![DW77y3](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/DW77y3.png)

#### 概率隐性语义索引PLSI

对LSI引入概率图模型，得到概率化解释。

1. 假设文档中隐含主题信息，从文档到主题到词项之间建立概率模型。
2. 简单说就是拿到一篇文档有个概率，这个文档里出现某个主题有个概率，得到这个主题后出现某个词又有个概率。
3. ![hStekg](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hStekg.png)
4. ![jwwigd](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/jwwigd.png)
5. ![z3L1zj](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/z3L1zj.png)
6. ![wh4Ayy](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wh4Ayy.png)

#### 排序学习模型C&W

1. 同时使用单词上下文
2. 对单词序列打分使用排序损失函数，而不是基于概率的极大似然估计
3. 损失函数：$max(0,\ 1-s(w,c)+s(w', c))$，其中，c 表示单词 w 的上下文(context)，w' 表示将当前上下文中的单词w替换为一个随机采样出的无关单词w'（负样例）；s为打分函数，分数越高表明该段文本越合理
4. $1 - [ s(w, c) - s(w', c) ]$, $s(w, c)$越合理，$s(w', c)$越不合理，那么$1 - \delta$之后值越低，损失越小
5. 模型的目标是，尽量使正确短语的得分 比 随机替换后的短语的得分高于1

#### 上下文预测模型Word2Vec

![xubcGw](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xubcGw.png)

#### 全局上下文模型Glove

1. ![9eZ4dz](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/9eZ4dz.png)
2. ![Qy58k9](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Qy58k9.png)

### 句子的表示方法

1. 传统方法
   - 词集模型
   - 词袋模型
   - TF-IDF表示
2. 分布式表示方法
   - 主题模型
   - 基于单词分布式表示组合的表示方法
   - 由原始语料直接学习的表示方法

#### 词袋模型Bag of Words

词袋模型(Bag of Words)是在词集模型的基础上，考虑了单词出现的次数，因此，在词袋模型中，句子向量中每个单词对应的位置上记录的是该**单词出现的次数**，这也体现了各个单词在该句子中的重要程度

![njryam](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/njryam.png)

#### TF-IDF模型

TF-IDF (Term Frequency–Inverse Document Frequency)是一种用于信息检索与数据挖掘的常用加权技术。

TF是词频(Term Frequency)，IDF是逆向文档频率(Inverse Document Frequency)。

TF-IDF的主要思想是：如果某个词或短语在一篇文章中出现的频率TF高，并且在其他文章中很少出现，则认为该词或者短语具有很好的类别区分能力，适合用来分类

![sFA60m](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/sFA60m.png)

### 文本匹配问题

#### 基于启发式规则

模型直接建模了两段文本共同出现的词的分布。不难理解，在两段文本中同时出现的词，对于度量文本的匹配程度是至关重要的

经典模型：
- BM25
- 查询似然模型（Query Likelihood Model）

#### 基于隐语义表达

除了两段文本中共同出现的词对计算匹配度有贡献，那些词与词之间的关系（近义词，包含关系等）也应该考虑进匹配度的计算，因此提出了**隐语义表达的文本匹配模型**

这类方法将一段文本**映射到一个向量**（离散稀疏向量或者连续稠密向量），然后通过**计算向量的相似度来表示文本的匹配度**。文档表示的各类方法（如TF-IDF和BM25）可以参考上一节的相关内容

#### 基于学习

分两类：
- 基于人工特征的排序学习模型
  ![xoLpLO](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xoLpLO.png)
- 基于表达学习的排序学习模型
  ![HMDkZX](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/HMDkZX.png)

#### 评价方法

1. **准确率Acc**：使用分类准确
率可以方便的评价模型对每一对文本的分类是否正确
2. **P@k(Precision at k)**：表示前k个文档的排序准确率。假定**预测结果排序**后，前k个文档中相关文档的数量为$Y_k$，那么P@k可以定义为：$P@k = \frac{Y_k}{k}$
3. R@k(Recall at k)：表示前k个文档的排序召回率。按照**标注的相关度**排序后，前k个文档中相关文档的数量为$G_k$，那么可定义R@k为：$R@k = \frac{G_k}{k}$
4. **MAP(Mean Average Precision)：**：该指标综合考虑了所有相关文档的排序状况。将所有相关文档在预测结果排序中的位置定义为$r_1, r_2,...r_G$, 则**平均精度均值指标**可定义为：$MAP = \frac{\sum_{i=1}^G P@r_i}{G}$
5. **MRR(Mean Reciprocal Rank)**：如果只考虑预测结果排序中第一个出现的相关文档的位置$r_1$，可以定义MRR指标为：$MRR = P@r_1 = \frac{1}{r_1}$
6. nDCG(normalized Discounted Cumulative Gain) 归一化折扣累计收益：
   1. 有些任务当中标注的相关度本身就有大小之分而不是单纯的匹配和不匹配两个级别，这个时候nDCG这个指标就会更加有效。**nDCG让相关度越高的排在越前面**
   2. ![571UXp](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/571UXp.png)

## 知识图谱与知识计算

知识图谱本质上是一种语义网络(Semantic Net)，其结点代表实体(entity)或概念(concept)，边代表实体/概念之间的各种语义关系。知识图谱把不同来源、不同类型的信息连接在一起形成关系网络，提供了*从关系的角度去分析问题的能力*

知识表示三元组：
- head：头实体/概念
- relation：关系/属性
- tail：尾实体/概念
- ![llRiMR](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/llRiMR.png)

知识图谱的生命周期：  
知识建模 -> 知识获取 -> 知识融合 -> 知识存储 -> 知识计算 -> 知识应用

### 实体抽取

定义：从原始语料中自动识别出**指定类型的命名实体**，主要包括**实体名**（如人名、地名、机构名、国家名等）、**缩略词**，以及一些数学表达式（如货币值、百分数、时间表达式等）

#### 基于机器学习的方法

序列标注，实体标注一半使用**BIO模式**（B-begin，I-inside，O-outside），还有**BIOES标注模式**（B-begin，I-inside，O-outside，E-end，S-single）
![P1O5PB](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/P1O5PB.png)

##### 隐马尔科夫模型

![klbqfY](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/klbqfY.png)
![R16KVS](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/R16KVS.png)

##### 条件随机场CRF

![SaqvKa](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/SaqvKa.png)
意思就是，$P(Y_v|X,Y_w, w /= v)$只和有相邻边的节点有关。

##### 线性链条件随机场模型

![ATN7FB](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ATN7FB.png)
![3TuBVI](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3TuBVI.png)
![s32JfP](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/s32JfP.png)

#### 关系抽取

举例：
![LjYS75](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/LjYS75.png)

分为：
- 有监督关系抽取
- 半监督
- 远程监督
- 无监督

##### 远程监督关系抽取

动机：通过外部知识库代替人对语料进行标注，从而低成本地获取大量有标注数据

核心思想：如果知识库中存在三元组$〈e_1,R,e_2〉，那么语料中所有出现实体对$〈e_1,e_2〉的语句，都标注为表达了关系R。用分类方法解决关系抽取问题

后来，Riedel等认为Mintz的假设过强，可能引入噪声，因此提出“at-least-once”：如果知识库中存在三元组$〈e_1,R,e_2〉，那么语料中所有出现实体对$〈e_1,e_2〉的语句，**至少有一句**体现了关系R在这两个实体上成立的事。

引入 **多实例学习机制**，将所有$<e_1, e_2>$共现的句子聚成一个句袋，并将任务由对句子分类变为对句袋分类。

##### 实体对齐

也称为**实体匹配或实体解析**，指的是**判断相同或不同的知识库中的两个实体是否指向同一个对象的过程**

方法有成对实体对齐和集体实体对齐。

##### 实体链接

指，利用知识库中的实体对知识抽取阶段所获得的的实体指称词**进行消岐的过程**。（看是不是想要的那个实体）

NIL实体，指的是知识库中找不到对应实体。实体链接还需要对NIL进行预测。

因此，实体链接为每一个实体指称词在知识库中找到对应的映射或者给出NIL实体的标签

##### 知识推理

就是用已知的知识，推出新知识的过程

很迷，不知道老师画这个重点啥意思。。。

##### 基于分布式表达的知识计算

这部分不知道怎么讲，自己也看的迷迷糊糊的，特别是目标函数写成分量那里，感觉shape对不上号

用张量表示知识图谱：
![eGcy6V](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/eGcy6V.png)

张量分解的目标函数：
![PdI0Lb](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/PdI0Lb.png)
![5VAVc0](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/5VAVc0.png)
![AqqMsE](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/AqqMsE.png)

##### 基于分布式表达的推理TransE

**基本思想：** 把关系看作是头尾实体之间的平移(翻译)操作

![RmgVZk](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/RmgVZk.png)

**模型的学习：**  
![gWb8vH](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/gWb8vH.png)
![qo2HCv](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/qo2HCv.png)

损失函数的不足：
- 最优边界在一个候选集中选取
- 不同的知识图谱共享相同候选集

两个改进：  
![02Oj8U](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/02Oj8U.png)

## 大图挖掘与分析

### patterns

1. degree distribution
   1. 分布不是高斯分布![abQ72L](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/abQ72L.png)
   2. ![1S7tIF](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1S7tIF.png)
   3. 幂率是指一个值出现的概率正比于此值的负指数。因此取 log 的话，就会是一条直线。幂律分布通俗的讲就是小的东西用得多，大的东西用得少。
   4. 网络中的度，邻接矩阵的特征值等很多 pattern 都满足幂律分布
2. Eigenvalue Power Law 1.邻接矩阵的正负特征值都满足幂律分布，如果是拉布拉斯矩阵，那就只有非负的特征值
3. Triangle Law
   1. 2x 的 friends，有多少三角形？答案是 3x
   2. n 个 friends，有$n^{1.6}$个三角形
   3. 大图中三角形的计数![zxuoKY](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/zxuoKY.png)
4. Connected components
   1. 计数 count 和 size 还是呈现幂率关系
5. Small and Shrinking Diameter
   1. 真实数据中，图的直径在缩小，类似幂律分布![UDRtyw](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/UDRtyw.png)
6. Densification Power law
   1. 节点从 N 变成 2N，图的边数如何变化？不是变成 2 倍，而是$2^{1.66}$，符合幂律分布，和之前数三角形一样
   2. ![9xBzq3](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/9xBzq3.png)

### 检测密集子图

下面是一个用于检测图中密集 block，且算法复杂度近似线性的搜索算法

> 为什么要检测密集子图？
>
> - 因为，通常密集子图比较值得注意，要么是有异常，要么就是重要的信息，想想微博的网络

#### 算数平均度数和几何平均度数

![0RVsgk](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/0RVsgk.png)

#### 算法

![HZEeRA](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/HZEeRA.png)

#### 理解

1. 先每列或者每行找到一个度数和最少的，然后把那一行或者那一列删了
2. 如果此时的密度（剩余节点度数和除以总结点数）比上一轮保存的大，就更新图结构和最优值。直到图被删光。
3. 最后返回具有最大平均度的子图

#### 证明$g(X_{best})> \frac{1}{2}g(X_{opt})$

1. 符号说明：
   1. S：当前图的全部节点集合
   2. $w_i$：节点 i 在 S 中的权重，通过累加 node i 的出边和入边权重得到，如果每个边的权重为 1，那$w_i$就是节点度数。![MICbPg](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/MICbPg.png)
   3. f(S)：当前图的所有边的权重和；如果每个边权重为 1，即边总数
   4. |S|：节点数
   5. S\*：度数最大的子图
   6. $g(S)=\frac{f(S)}{|s|}$
2. 定理 1：
   1. OPT 子图中，每个节点的权重（度数）都大于等于平均度数
   2. ![odRkFB](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/odRkFB.png)
3. 引理 1：
   1. 符号说明
      1. S’：原图
      1. S\*：最优集合
      1. vi：要删除的 node i，恰好在最优集合 S\*中
   1. 引理 1：要删除的节点 i，在原图中的节点权重，大于等于最优集合中的对应节点权重（这是肯定的，因为 S\*是 S’删除了一部分节点得到的）
   1. ![YRd0ti](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/YRd0ti.png)
4. 引理 2：
   1. 要删除的节点是图中节点权重最小的（度最少的）
   2. ![J2MiSj](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/J2MiSj.png)
5. 引理 3：
   1. 因为一条边能最多赋值给连个节点，因此$f(S') \ge \frac{|S'|w_i}{2}$。这里意味着用$v_i$的权重替代所有的节点权重。相当于设置了一个近似下界。$\frac{w_i}{2}$相当于是对节点 i 的度数的平均，每条边都被算了两次呀。因此也能得到$\frac{f(S')}{|S'|} \ge \frac{w_i}{2}$。
   2. ![GPIYEv](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/GPIYEv.png)
6. 引理 4：
   1. 因为算法返回最优解，即平均度数最小的，所以$g(A \cup B)>g(S')$。$A \cup B$是算法找到的最优解。
7. 综上：
   $$ g(A \cup B) \ge g(S') = \frac{f(S')}{|S'|} \gt \frac{w_i}{2} \ge \frac{w_i(S*)}{2} \ge \frac{g(S*)}{2}$$

### EigenSpoke 最密子图检测

spectral-based method，基于频谱的方法

根据奇异值向量找到密集子图，每个奇艺向量都对应一个密集子图。
![Xl5cDP](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Xl5cDP.png)

1. 对邻接矩阵进行 SVD 分解
2. 绘制 Spectral Subspace Plot，即$(u_1, u_2)$等的散点图
3. 密集区域在 Spectral subspace plot 上会有明显的聚类现象，而不是零散的。因此谱空间的每一个簇，都能对应一个密集子图 ![d6Y7hr](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/d6Y7hr.png)
4. 两条线垂直是因为 U 和 V 都是正交矩阵，u1 和 u2 是正交的，因此如果把他们当成正交基来看，就是垂直的。
5. 现象：
   1. 短的镭射线：高密度无伪装的异常![B3LCsq](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/B3LCsq.png)
   2. 长的镭射线：低密度无伪装异常。![QXWT7O](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/QXWT7O.png)
   3. 倾斜镭射线：包含了伪装的异常。![BmbyGl](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/BmbyGl.png)
   4. 珍珠状的：有重叠的密度块。![1sWz1w](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1sWz1w.png)

### HoloScope

如果存在欺诈行为，那么就会产生一些 false positive 的密集子图，如下
![XlBEwq](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/XlBEwq.png)

子图密度定义为图中所有节点权重之和除以边数和节点数
$$ Density D(A, B) = \frac{\sum\_{v_i \in B} f_A(v_i)}{|A|+|B|}, A \in U, B \in V $$

$$f_A(v_i) = \sum_{(u_j, u_i)\in E \land u_j \in A }\sigma_{ji} e_{ji}$$
$$i.e. nums \ of \ edges \ from\ A \ to \ v_i,\ \sigma_{ji} \ is\ edge\ weight$$

沿用 Fraudar 模式，对边进行加权。如果可疑账户组为 A，则目标$v_i$的可疑程度就由 A 连接到$v_i$的边与所有用户连接到$v_i$的边的比例所决定。

如果是流行商品，那么所有账户都连边，分母就比较大，这个比例就很低，如果恶意刷好评，那么只有欺诈账户指向$v_i$，分母就比较小，比例就很高。

![kcjpQp](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/kcjpQp.png)
![S6ZjIK](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/S6ZjIK.png)

算法执行过程是只删去一行，因为我们设定的是行代表账户，列代表商家或者商品，目标是在行里面找异常行为。也是启发式的贪心算法。

伪代码：

- ![Z5rNMy](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Z5rNMy.png)

优点：

- 综合考虑许多属性信息作特征
- 复杂度近似线性
- 检测更精确

### D-cube

问题定义：

1. 在随着时间变化的数据中找到密集子块
2. ![or0xV6](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/or0xV6.png)
3. 返回的结果是 k 个子块

步骤：

1. 在 slice 最多的那个维度上，计算每个 slice 的平均度数![uqozHD](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/uqozHD.png)
2. 然后去掉低于平均度数的那些 slice
   ![buC0N3](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/buC0N3.png)
   ![eEJPtB](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/eEJPtB.png)
3. 再找剩余 slice 最多的维重复上述操作，直到 tensor 为空
   ![LyHOLH](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/LyHOLH.png)
4. 每次迭代的过程要记录下子块密度，最后返回最密集的子块
5. ![II2i21](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/II2i21.png)

### EigenPulse

#### 问题定义

如何在 streaming graphs 中检测密集子图？

1. DenseAlter Algorithm
2. Graph spectral, Row-Augmented Matrix, AugSVD
3. EigenPulse Algorithm，流式 SVD 分解

#### Sketching 和 AugSVD

1. 对于流式数据，设置一个滑动窗口，提取部分数据，设为 A
   ![2rTf5v](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/2rTf5v.png)
2. Sketching，使用矩阵 Q 和 B 来近似 A。  
   ![CkyoWW](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/CkyoWW.png)
   1. B 是通过 Q 和 A 算出来的，$B = Q^T A = \Omega^T A^T A$。
   2. Q 是对 A 使用$\Omega$进行 Sketch 得到的。
   3. 因此只要生成一个$\Omega$就可以。
3. 生成 Q 和 B，没太懂怎么生成的
   1. 通过每次读 A 的一行 a 来构造 G 和 H
   2. 通过 G 和 H 构造 Q 和 B
   3. ![uyOZx5](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/uyOZx5.png)
4. 对 B 进行 SVD 分解![Ovucto](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Ovucto.png)

那到底如何进行 EigenPulse 算法呢？

1. 在每个时间段，拿到矩阵 A
2. 流式奇异值分解后，选择前几个奇艺向量
3. 根据前几个奇艺向量选择密集块，也没太懂怎么做的
   ![6ck6KX](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/6ck6KX.png)
4. 如果要在小的选择的块里检测密集块，可以使用 Fraudar 或者 HoloScope
5. 画出 EigenPulse，然后检测异常点![1fZXgM](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1fZXgM.png)

## 社交媒体分析

### PageRank

用于组织网络搜索结果

网页搜索过程中：

1. 入链接 in-coming links 越多，网页越重要
2. 越重要的页面的出链接的权重也越大

定义页面和链接的权重：

1. 每个 out link 的重要性是 page 重要性的$\frac{1}{n}$，如页面 importance 为$r_j$，且有$n$个 out link，则每个 out link 的重要性为$\frac{r_j}{n}$
2. 每个页面的重要性为所有 in-link 的权重之和
   $$r_j = \sum_{i\rightarrow j}\frac{r_i}{d_i}$$

**Note:**  
权重矩阵和图的邻接矩阵不一样，权重矩阵是$i->j，M_{ji}$，邻接矩阵是$i->j, M_{ij}$

**是否存在唯一解？**

- 如果有三个节点，则计算三个页面权重需要三个等式。使用消元可以去掉一个等式，相当于两个等式。所以还是没有唯一解的。
- 因此添加一个约束$\sum_i r_i=1$，所有页面权重和为 1。
- 但还是存在问题，高斯消元在大图计算中太慢了。

#### 矩阵表示

引入随机邻接矩阵$M$

- $shape=[d, i]$
- 每一列表示一个页面，然后那一列的每个元素表示一个 out link
- 如果$i\rightarrow j$，则$M_{ji}=\frac{1}{d_i}$，否则的话$M_{ji}=0$
- 所有页面权重和为 1，$\sum_i r_i=1$
- **更新权重：**
  - $r = M \cdot r = \sum_i r_i \cdot M_{*i}$。
  - 相当于 M 的一行乘以所有页面的权重$r$
  - rank vector r 相当于 M 矩阵的特征向量。最大的特征向量对应特征值 1。

#### Power iteration

知道表示了，那么如何计算权重 r？使用 Power iteration，它是一个简单的迭代模型

1. 假设 N 个页面
2. 初始化：$r^{(0)} = [\frac{1}{N},...,\frac{1}{N}]^T$
3. 迭代：$r^{(t+1)} = M \cdot r^{(t)}$，$r_j^{(t+1)} = \sum_{i\rightarrow j} \frac{r_i^{(t)}}{d_i}$
4. 终止：$|r^{(t+1)}-r^{(t)}|< \epsilon$，这里表示的是 L1，偏差的绝对值之和，也可以用 L2 norm
5. 例子：![enlqVe](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/enlqVe.png)
6. 多次迭代相当于：$r^{(n)} = M^n \cdot r^{(0)}$。并且这一序列过程相当于在近似矩阵 M 的占据主导地位的特征向量。
   1. 把$r^{(0)}$用 M 的特征向量表示，以特征向量为基进行 depose，得到$r^{(0)}=c_1x_1+c_2x_2+...+c_nx_n$
   2. ![QTCYTU](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/QTCYTU.png)
   3. ![jSVxWj](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/jSVxWj.png)

也可以用 Random Walk 来解释，使用的是一个 Stationary Distribution，就是把 r 矩阵当成所有 pages 上的 probability distribution，是 stationary distribution of a random walk

#### 存在的问题

PageRank 存在两个问题，分别是 dead ends 和 spider traps。

##### dead ends

简单理解就是页面最终指向了一个页面，但是这个页面没有 out link，因此 random walk 就到了 dead ends

![3vtif1](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3vtif1.png)

##### Spider traps

简单理解就是，所有的 out link 形成了一个环，就在这个环里出不去了，最终所有的 importance 都被汇聚在这个环里。

![D2Fa9L](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/D2Fa9L.png)

#### Teleports

能够解决上面两个问题

1. 对于 Dead ends，遇到 dead ends 就以 1 的概率 teleport 随机跳到其他页面。
   - ![71xuTE](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/71xuTE.png)
   - dead ends 是一个真的 problem，某一列全为 0 就会导致这个问题
2. 对于 Spider traps，在每个 time step，有两个选择：
   - 在$\beta$的概率下随机 follow a link
   - 在$1-\beta$的概率下，跳转到随机的页面，就是说可以到**没有链接**的其他页面。
   - 因此在有限步内就会跳出这个 spider trap
   - spider traps 其实不是算法的问题，只是它的结果不是我们想要的，因此只要让它跳出去就好

因此加上了 Random Teleports 的算法，公式变为：

$$r_j = \sum_{i\rightarrow j} \beta \frac{r_i}{d_i} + (1-\beta) \frac{1}{N}$$

其中$\beta$是跟随 out link 的概率，$1-\beta$是跳转到随机页面。$\beta=0.8\ or\ 0.9$

#### Matrix Formulation

添加 Teleports 后，原本的 M 矩阵可以改写为：

$$A = \beta M + (1-\beta ) \left[ \frac{1}{N} \right]_{N\times M}$$

$$r' = A \cdot r = \beta M \cdot r + \left[ \frac{1-\beta}{N} \right]$$

推导：![Ga4Dl1](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Ga4Dl1.png)

注意到，如果存在 dead ends 的话，后面加上$\frac{1-\beta}{N}$是没办法让 importance 和为 1 的，也就是$\sum_j r_j^{new} < 1$。所以令 $S = \sum_j {r'}_j^{new}$。
![JSQfne](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/JSQfne.png)

#### Sparse Matrix Encoding

很显然，整个 A 矩阵（or M 矩阵）是非常大且稀疏的，因此没必要存整个，也存不下，只存每个节点的 degree 和对应的 destination nodes。
![viUeeC](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/viUeeC.png)

#### 实际迭代更新

1. 如果有足够的内存读入$r^{new}$，然后把$r^{old}$存在磁盘中，则进行如下更新，每次读一行，更新所有目标节点权重：
   - ![cYXVuj](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/cYXVuj.png)
   - 每次迭代需要先读入$r^{old}$和 M，然后写入 r，开销是$2|r|+|M|$
2. 如果没有足够内存读入$r^{new}$，就把$r^{new}$拆成 k 份，对于每一份，进行上面那样的更新。因此要进行上面的 k 次更新。
   - ![BicUB3](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/BicUB3.png)
   - 时间开销是：$k(|M|+|r|) + |r| = k|M|+(k+1)|r|$
   - 问题是，M 非常大，读 k 次这种操作应该避免
3. 对于 2 的问题，可以把 M 也拆分开读，因为每次读入的 M，只更新对应 block 内的$r^{new}$，所以其他的相当于是冗余的。
   - 把 M 拆开，每个 stripes 只保存那些 destination 到当前$r^{new}$ block 的节点。
   - 时间花费是：$|M|(1+\epsilon )+(k+1)|r|$
   - ![MketkP](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/MketkP.png)

### 其他

还有 SimRank 和 Topic-Specific PageRank(or Personalized
PageRank)，以及 Random Walk with Restarts，这里不介绍。

## 题型

1. 单选，15*2'
2. 判断，5*2'
3. 简答题，3*6'
4. 证明题/应用题，3\*10'+1\*12'
