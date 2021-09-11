---
title: "论文笔记 | VLN BERT: A Recurrent Vision-and-Language BERT for Navigation"
mathjax: false
date: 2021-09-10 20:11:01
tags:
  [
    Recurrent-VLN-BERT,
    VLN,
    BERT,
    PaperNotes,
    Multimodal,
    DeepLearning,
    CVPR,
  ]
categories: Papers
urlname: recurrent-vln-bert
---

<meta name="referrer" content="no-referrer" />

{% note info %}
2021 CVPR 《VLN BERT: A Recurrent Vision-and-Language BERT for Navigation》论文精读笔记
{% endnote %}

<!--more-->

## Abstract and Introduction

1. 将BERT 结构应用到部分观测的马尔科夫决策过程是很难的
2. 这篇文章提出了一种recurrent BERT 模型，用BERT和一个recurrent function来维持跨模态的agent state信息
3. 这种方法可以泛化成transformer based结构，并且能够解决navigation和referring expression问多任务
4. 之前的方法都是把VLN BERT用于encode或者计算instruction path的能力，这个方法可以直接用来学习去navigate
5. VLN可以摆看做马尔科夫决策过程，未来的决策会依赖当前的状态。每次决策时，只对应一部分instruction，并且需要agent保持对过去路径的追踪，然后能确定现在对应哪个sub instruction，然后再作出决定
6. VLN需要很大的计算量，因为每个episode都很长，所以直接self attention的话，训练的计算量很大。所以他们用预训练模型来初始化，结合一个recurrent function，并让language token只是作为key来进行query，这样也能降低显存的使用

## Proposed Model

<img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wwAkK3.png" width="500">

1. VLN其实就是给出一系列的instruction $U$，每个时间步 $t$，agent观察到环境并作出一个action $a_t$，将agent的状态从 $s_t = <C_t, \theta_t, \phi_t>$变成 $s_{t+1}$，$C_t$是viewpoint，$\theta_t$是heading的角度，$\phi_t$是elevation的角度。如果agent决定stop，那就结束任务。
2. 形式化：
   1. $s_t, p_t^a, p_t^o = VLN \circlearrowright BERT( s_{t-1}, X, V_t O_t)$
3. 输入：
   1. 上次的agent state $s_{t-1}$
   2. language tokens $X$
   3. visual tokens $V$
   4. object tokens $O$
4. 输出：
   1. 当前的agent state $s_t$
   2. action probability $p_t^a$
   3. object grounding probability $p_t^o$
5. language processing:
   1. 用预训练模型OSCAR和PREVALENT
   2. CLS token被用于聚合相关的语义信息，用了对比学习
   3. 形式化为：$s_0, X = VLN \circlearrowright EBRT([CLS], U, [SEP])$
   4. navigation阶段，返回的X只是用于key和value，不会像 $V_t， O_t$那样self attention，认为language token已经是一个很深层次的语义表达了，不需要进一步encoded
6. vision processig：
   1. 先把image特征转换到BERT的相同空间 $V_t = I_t^vW^{I_v}$，$I_t^v$是指在这个方向
   2. 然后把 $V_t$和 $s_t，X$拼接到一起，送到模型中，即 $[V_t, s_t, X]$
   3. 对于REF 任务，$O_t = I_t^o W^{I^o}$投影到方向空间中，然后送到模型中
7. state representation：
   1. $s_t$ 用来表示在第t个时间步时，所有文本和图像的信息汇总，以及agent在之前做的所有decision
   2. 模型依赖BERT的原始结构来识别时间上依赖的输入，以及recurrent 更新state
8. state refinement：
   1. 没有使用state信息直接预测decision
   2. 对language和visual tokens分别：
      1. 计算attention score $A_{l,k}^{s,x} = \frac{{Q_{l,k}^s K_{l,k}^x}^T}{\sqrt{d_h}}$
      2. 得到多头注意力的平均score，然后softmax这些score得到对每个token位置的score。 $\hat{A_l}^{s,x} = Softmax(\bar{A_l}^{s,x}) = Softmax(\frac{1}{K}\sum_{k=1}^K A_{l,k}^{s,x})$
   3. 然后再和原有的tokne乘起来，得到加权和。 $F_t^x = \hat{A_l}^{s,x}X, F_t^v = \hat{A_l^{s,v}V_t}$
   4. 在原始的text和visual 特征之间进行跨模态对齐，是element-wise product的方式。$s_t^f = [s_t^r;F_t^x \circledcirc F_t^v]W^T$。这里的 $s_t^r$是最后一层的输出state特征。简单来说就是文本和图像特征对齐后和输出的state拼接在一起做了一个线性变换。REVERIE任务中不需要文本特征，因为它的instruction都是比较高层次的
   5. 最后得到当前的state：$s_t = [s_t^f;a_t]W^s$，其中 $a_t$是selected action。
9. decision makig：
   1. 用state和visual 特征做一个inner product来检验这个方向的state和vision是否对应
   2. 用了VisualBERT中类似的方式，直接对所有的attention head的最后一层进行权重平均，就像 $p_t^a = \hat{A_l}^{s,v}$一样.
   3. 对于remote referring expression task，$p_t^o = \hat{A_l}^{s,o}$，这里是对所有candidate object的
10. training：
    1. 结合了reinforcement learning和imitation learning
    2. 在RL方面用了A2C，agent在action中通过 $p_t^a$采样一个，并在每个时间步计算 $A_t$的优势
    3. 在IL方面只用teacher action，就是ground truth，对decision算一个交叉熵损失，$L = - \sum_t a_t^s log(p_t^a) A_t - \lambda \sum_t a_t^\* log(p_t^a)$，其中，$a_t^*$是teacher action，在REVERIE任务中，还有一项 $\sum_t o_t^* log(p_t^o)$

## Conclusion

1. 引入了recurrent 机制，对state的重复使用，就像LSTM中cell state一样，充分利用BERT原有的结构来识别time-dependent的输入。其实就是为了解决马尔科夫链中对过去状态的依赖。
2. 用BERT直接作为navigator，而不是像以往作为encoder。就是说输出部分直接得到action probability和object grounding probability（虽然不懂做啥的）
3. 用了多任务
4. 个人感觉主要是用了预训练模型提升的点，主要是从Table 3里面的3和4实验对比，直接从比较差的效果提升到了接近sota，后面加入的state、decision、matching也只是在这个基础上提升了一点点
5. 用了RL和IL
6. 用了对比学习