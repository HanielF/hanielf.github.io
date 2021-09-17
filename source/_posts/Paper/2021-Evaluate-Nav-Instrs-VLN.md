---
title: "论文笔记 | On the Evaluation of Vision-and-Language Navigation Instructions"
mathjax: true
date: 2021-09-11 20:11:01
tags:
  [
    EvalauteNavInstrsVLN,
    VLN,
    InstrMetrics,
    CompatibilityVLN,
    Benchmark,
    NLG,
    DataAugmentation,
    PaperNotes,
    Multimodal,
    DeepLearning,
    EACL,
  ]
categories: Papers
urlname: evaluate-nav-instrs
---

<meta name="referrer" content="no-referrer" />

{% note info %}
2021 EACL《On the Evaluation of Vision-and-Language Navigation Instructions》论文笔记
{% endnote %}

<!--more-->

## Abstract and Introduction

1. 导航提示自动生成模型目前都没有被很好的验证性能，并且自动验证的指标也没有验证过。
2. 论文发现目前很多生成器的效果仅仅和模板化的生成器差不多，验证了Speaker-Follower和EnvDrop两个模型，对比试验用了一些instruction 扰动和Trajectory扰动，包括：1. 方向变换 2. 实体变换 3. 短语交换
3. 这篇论文实验了多种指标比如BLEU，ROUGE，METEOR等，发现他们在验证真实导航语句上面效果很差。为了改进指令的评估，**提出了一种instruction-trajectory 兼容性模型，它不需要额外的参考指令。**对于存在参考说明的模型选择，我们建议使用 SPICE 指标。
4. 指令生成模型通常是用于提升VLN任务中agent的性能的，一方面能够起到数据增强的作用，另一方面能够在实用推理环境中扮演概率说话者的角色
5. 但是验证生成出来的instruction也是很重要的，文章验证了目前instruction的效果，并验证了自动化验证的指标的效果
6. 论文指出，文本评估指标在应用于新领域时应始终根据人类判断进行验证

## Human Wayﬁnding Evaluations

1. 为了对当前最先进的导航指令生成进行基准测试，我们通过让人为跟随它们来评估 Speaker-Follower 和 EnvDrop 模型的输出
2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/89NYFo.png" width="500">
3. 对比实验了Crafty模型（一种基于模板的instruction 生成器）、Human Instruction、Instruction 扰动（1. 方向变换 2. 实体变换 3. 短语交换）、Speaker-Follower模型和EnvDrop模型

## Compatibility Model

1. 验证Instruction和path兼容性的模型
2. 模型结构式一个对偶式的encoder，分别encode Instruction和trajector，并且把它们映射到相同的空间中。
3. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/4jfD6s.png" width="500">

### Instruction encoder

1. 输出：$h^w$，是bilstm最后输出的hidden的拼接
2. 输入：Instruction tokens $W = {w_1, w_2, ..., w_n}，用bert进行了embedding，然后直接输入lstm

### Visual encoder

1. 也是两层lstm
2. 输入：一系列的viewpoint信息：$V = {(I_1, p_1), (I_2, p_2)...,(I_t, p_t)}$，$I_t$是在位置 $p_t$处捕获的全景图，按照elevation和heading角度分成了36个图片
3. 输出：
   1. $a_t = Attention(h_{t-1}^v, e_{pano}, t)$
   2. $v_t = f([e_{prev,t},e_{next,t},a_t])$
   3. $h_t^v = LSTM(v_t, h_{t-1}^v)$
   4. $e_{prev,t},e_{next,t}$分别是当前viewpoint的前一个viewpoint和后一个viewpoint的图片信息
   5. Attention直接用的dot-productAttention
   6. $f$是一个projection layer，应该是fc
4. 整个的意思就是先提取当前信息，然后和前后的信息联合起来，用LSTM提取当前的视觉信息hidden

### 其他trick

1. 对比学习提升了9-10%的AUC
2. 往返回译
3. 数据增强
   1. 指令Instruction增强
   2. trajectories增强
   3. 方向变换，实体变换，短语交换
4. Loss：
   1. 分类loss：交叉熵loss和focal loss，
      1. $L_{FL} = (1-p_{i,j}^{\gamma=2} L_{CE}$，
      2. $p_{i,j}= \sigma(aS_{i,j}+b)$表示匹配概率，
      3. a和b都是学习出来的，
      4. $\sigma$是sigmoid
   2. 对比学习loss：
      1. $L_C(S) = \frac{1}{\sum_i M_i}\sum_{i=1}^N(L_r(S_i)+L_r(S_i^T))$
      2. $L_r(S_i) = -log \frac{e^{S_{i,i}/\theta}}{\sum_j^N e^{S_{i,j}/\theta}}$
   3. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/By7Oup.png" width="500">
5. 采样：按照ground truth，Instruction扰动，trajectories扰动，2:1:1来采集正负样本，对于每个扰动，按照扰动类型，等比例采样
6. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ZfzO4q.png" width="500">
7. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/9qcCuW.png" width="500">
8. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/UHYzUI.png" width="500">
9. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/8g2UmR.png" width="500">