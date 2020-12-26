---
title: Attention Is All You Need
comments: true
mathjax: true
date: 2020-08-15 18:42:24
tags:
  - DeepLearning
  - Attention
  - Transformer
  - SelfAttention
  - Multi-Head-Attention
categories: MachineLearning
urlname: attention-is-all-you-need
---

<meta name="referrer" content="no-referrer" />

{% note info %}

Transform, Self Attention, Multi-Head Attention的一些笔记。

{%endnote%}

<!--more-->

## Self Attention自注意力

1. 什么是self attention？
   - 传统的attention是靠对比如lstm的hidden state进行内积得到的energy权重
   - 这里是通过对当前层的输入，直接得到，而不是hidden，因此叫self attention（来自输入向量本身）
2. 怎么实现的？通过三个向量实现，分别是：
   - $Query$向量：用于对所有位置进行atten to。
     - 怎么atten的？是通过$query \circ keys$得到。
     - atten的得到的有什么用？可以想象是原始attention机制中的energy，其实就是一个权重得分，最后要拿去对$Values$向量进行加权平均，得到context向量。
   - $Keys$向量：表示这个位置，要被query向量检索的信息，因为每次query向量attend到你的时候，都是拿key向量去给它进行内积
   - $Value$向量，表示这个位置真正的内容，最后是要拿来给energy加权平均的。
3. 怎么得到Query，Keys，Values向量的？
   - 对输入向量分别进行线性变换就得到了
   - 具体点就是使用三个矩阵，$W^Q$，$W^K$，$W^V$
   - 因此，原始attention相当于是self attention的特殊情况，keys和values都是向量本身，然后Query是隐状态hidden，然后计算energy，在softmax变成概率，在把所有hidden加权平均得到context
4. 和原始attention结构对比
   1. 原始attention
   ![FrPNLi](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/FrPNLi.png)
5. 计算过程
   1. 单个位置计算过程
   2. ![1Y5WvG](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1Y5WvG.png)
   3. ![1utV3a](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1utV3a.png)
   4. divide by $8(\sqrt{d_k}$是说这样更stable
   5. 多个位置同时计算，使用矩阵
   6. ![M1H5sK](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/M1H5sK.png)
   7. ![ecayu5](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ecayu5.png)

## Multi-Head Attention多头注意力

1. 直观理解什么是多头注意力
   1. 前面的self attention只有1组Q、K和V，那么我们设置多组Q、K、V就可以学习不同的上下文。这个不同是由训练数据驱动的。
   2. 有点像是卷积核设置多个channel
2. 多个head得到很多$Z$矩阵，怎么处理？
   1. 按照第二个维度concatent，每个词得到一个超长的feature
   2. 不感觉太大了吗？所以使用了一个权重矩阵$W^0$对Z进行线性变换，得到一个比较小的Z
3. 图示
   1. ![QFd6xQ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/QFd6xQ.png)
   2. ![qxQguu](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/qxQguu.png)
   3. ![NjaFkr](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/NjaFkr.png)
   4. ![l3yxSv](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/l3yxSv.png)
4. 多头注意力作用表现：
   1. 其中一个head的语义：  
   ![gh9JUm](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/gh9JUm.png)
   1. 另一个head的语义：  
   ![MvNU7m](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/MvNU7m.png)

## Positional Encoding位置编码

1. 为啥要位置编码？
   1. 因为从上面的self attention可以看出来，它并没有获取RNN里面那种位置信息，因此如果想要达到RNN那样的效果，就要加入位置编码。
   2. 距离，北京到上海的机票，和上海到北京的机票，这两个北京的embedding相同，但是我们希望他的语义不同，一个是出发地，一个是目的地。如果用RNN就有可能学出来。
2. 怎么实现？
   1. 在原始的embedding上加入一层positional embedding，然后两个相加得到最后的embedding。
   2. ![tW3AQV](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/tW3AQV.png)
   3. ![zgpKJ2](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/zgpKJ2.png)

## Layer Normalization

1. 什么是Layer Normalization？
   1. 简单说就是对每个样本的所有特征进行normalization
   2. batch norm是对每个特征进行norm
2. 相比Batch Normalization？
   1. Batch Normalization是对每个特征而言，计算每个特征维度下的均值和方差
   2. Layer Normalization是针对每个样本而言，计算每个样本的均值和方差
   3. 因此相比Batch Normalization，Layer Normalization可以在minibatch为1的情况下进行，而Batch Normalization下的minibatch不能太小。
3. Batch Normalization相比之下的缺点
   1. 需要一个minibatch的数据，而且这个minibatch不能太小(比如1)
   2. 不能用于RNN，因为同样一个节点在不同时刻的分布是明显不同的。当然有一些改进的方法使得可以对RNN进行Batch Normalization，比如论文[Recurrent Batch Normalization](https://arxiv.org/abs/1603.09025)
4. 计算
   1. ![sr8FZp](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/sr8FZp.png)
   2. ![qJlcaR](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/qJlcaR.png)

## 残差连接

1. 什么是残差连接？
2. 为什么要用残差连接？
3. 怎么实现的？
   1. 输入𝑥1,𝑥2经self-attention层之后变成𝑧1,𝑧2，然后和残差连接的输入𝑥1,𝑥2，加起来，然后经过LayerNorm层输出给全连接层。全连接层也是有一个残差连接和一个LayerNorm层，最后再输出给上一层。

![EOrWX6](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/EOrWX6.png)

![UGNR6b](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/UGNR6b.png)