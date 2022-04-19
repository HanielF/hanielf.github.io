---
title: 2021-CIKM-QQ浏览器AI算法大赛-赛后分享会笔记
tags:
  - CIKM
  - Competition
  - MultiModal
  - VideoEmbed
  - Notes
categories:
  - Notes
urlname: cikm-qq-competition
comments: true
mathjax: true
date: 2021-11-05 14:30:35
---

<meta name="referrer" content="no-referrer" />

{% note info %}

CIKM-QQ浏览器 联合举办的 2021AI算法大赛，赛道一，多模态视频相似度，赛后top分享会笔记。

{% endnote%}
<!--more-->

## 赛题介绍

赛题一：
信息流场景下，短视频消费引来爆发式增长，视频的语义理解对于提升用户消费效率至关重要。视频Embedding采用稠密向量能够很好的表达出视频的语义，在推荐场景下对视频去重、相似召回、排序和多样性打散等场景都有重要的作用。本赛题从视频推荐角度出发，提供真实业务的百万量级标签数据(脱敏)，以及万量级视频相似度数据(人工标注)，用于训练embedding模型，最终根据embedding计算视频之间的余弦相似度，采用Spearman’s rank correlation与人工标注相似度计算相关性，并最终排名。

## 开源方案汇总

1. [第一名](https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st)
2. [第三名](https://github.com/chenghuige/pikachu2/tree/main/projects/ai/qqbrowser )
4. [第四名](https://github.com/kywen1119/Video_sim)
6. [第六名](https://github.com/763337092/AIAC2021_task1_rank6)
11. [第十一名](https://github.com/cgxcompetition/AIAC_qq_browser_2021_task1_rank11)
17. [第十七名](https://github.com/chenjiashuo123/AIAC-2021-Task1-Rank17)

## Top1方案

[github链接](https://github.com/zr2021/2021_QQ_AIAC_Tack1_1st)

 分三个阶段

### pretrain

![oWwbhM](https://raw.githubusercontent.com/HanielF/ImageRepo/main/blog/oWwbhM.jpg)


(1) Video tag classify 任务
tag 为人工标注的视频标签，pointwise 和 pairwise 数据集合中提供。
和官方提供的 baseline 一致，我们采用了出现频率前1w 的tag 做多标签分类任务。
Bert 最后一层的 [CLS] -> fc 得到 tag 的预测标签，与真实标签计算 BCE loss

(2) Mask language model 任务
与常见的自然语言处理 mlm 预训练方法相同，对 text 随机 15% 进行 mask，预测 mask 词。
多模态场景下，结合视频的信息预测 mask 词，可以有效融合多模态信息。

(3) Mask frame model 任务
对 frame 的随机 15% 进行 mask，mask 采用了全 0 的向量填充。
考虑到 frame 为连续的向量，难以类似于 mlm 做分类任务。
借鉴了对比学习思路，希望 mask 的预测帧在整个 batch 内的所有帧范围内与被 mask 的帧尽可能相似。
采用了 Nce loss，最大化 mask 帧和预测帧的互信息

(4) 多任务联合训练
预训练任务的 loss 采用了上述三个任务 loss 的加权和，
L = L(tag) * 1250 / 3 + L(mlm) / 3.75 + L(mfm) / 9
tag 梯度量级比较小，因此乘以了较大的权重。
注：各任务合适的权重对下游 finetune 的效果影响比较大。

(5) 预训练 Setting 
初始化：bert 初始化权重来自于在中文语料预训练过的开源模型 https://huggingface.co/hfl/chinese-roberta-wwm-ext-large
数据集：预训练使用了 pointwise 和 pairwise 集合，部分融合模型中加上了 test 集合（只有 mlm 和 mfm 任务）
超参：batch_size=128, epoch=40, learning_rate=5e-5, scheduler=warmup_with_cos_decay, warum_ratio=0.06
注：预训练更多的 epoch 对效果提升比较大，从10 epoch 提升至 20 epoch 对下游任务 finetune 效果提升显著。

### finetune

![Mudbig](https://raw.githubusercontent.com/HanielF/ImageRepo/main/blog/Mudbig.jpg)

(1) 下游任务
视频 pair 分别通过 model 得到 256维 embedding，两个 embedding 的 cos 相似度与人工标注标签计算 mse

(2) Finetune header
实验中发现相似度任务中，使用 mean_pooling 或者 attention_pooling 聚合最后一层 emb 接 fc 层降维效果较好。

(3) Label normalize
评估指标为 spearman，考查预测值和实际值 rank 之间的相关性，因此对人工标注 label 做了 rank 归一化。
即 target = scipy.stats.rankdata(target, 'average')

(4) Finetune Setting
数据集：训练集使用了 pairwise 中 (id1%5!=0) | (id2%5 !=0) 的部分约 6.5w，验证集使用了(id1%5==0) & (id2%5==0) 的部分约 2.5k
超参：batch_size=32, epoch=10, learning_rate=1e-5, scheduler=warmup_with_cos_decay, warum_ratio=0.06

### ensemble

![image.png](https://note.youdao.com/yws/res/1842/WEBRESOURCEa32c21d61b8e081db2d8dee9fdf31e8a)

(1) 融合的方法
采用了 weighted concat -> svd 降维 方法进行融合，发现这种方法降维效果折损较小。
concat_vec = [np.sqrt(w1) * emb1, np.sqrt(w2) * emb2, np.sqrt(w3) * emb3 ...]
svd\_vec = SVD(concat\_vec, 256)

(2) 融合的模型
最终的提交融合了六个模型。 模型都使用了 bert-large 这种结构，均为迭代过程中产出的模型，各模型之间只有微小的 diff，各个模型加权权重均为 1/6。
下面表格中列出了各模型的diff部分，验证集mse，验证集spearman

![vCyFGW](https://raw.githubusercontent.com/HanielF/ImageRepo/main/blog/vCyFGW.png)

(3) 单模型的效果与融合的效果
单模的测试集成绩约在 0.836
融合两个模型在 0.845
融合三个模型在 0.849
融合五个模型在 0.852

### QA

1. 如何处理ASR？没有处理ASR，因为计算量会大很多，但是倾向于加入ASR
2. label normalize怎么做的？把pairwise的label按照数值排序，然后均匀归一化到0-1之内。因为是spearman rank，所以想看label和rank之间的关系
3. 融合的模型区别？在于模型是否加了某个预训练任务，比如MLM，或者是模型是30个epoch还是100个epoch，这样一些模型进行融合。
4. 是否用了K折？没有，因为一折和五折没啥差别
5. tag标签是否处理？没有，直接follow官方的top 1w
6. tag是1w分类？不是，是1w个二分类
7. svd降维提升多少？千分之一
8. mask frame是怎么构造正负样例？用NCEloss，batch中的其他所有帧都是负例，只有和它自己是正例
9. loss叠加的时候，怎么设计那个权重的？主要是看到tag的loss比较低，所以觉得应该乘以比较大的权重，这个权重的值是试出来的。
10. 


## Top2
四个pretrained task：MLM(1.5%)，Inverse Close Task(1.5%), 对比学习SimCSE（0.5%）Match（0.5%）融合模型（1.2%）

剩下的都在做

![image.png](https://note.youdao.com/yws/res/9/WEBRESOURCE9db439e28ce254955266b1f6e0e35229)

### QA

1. ensemble? 五折，然后svd降维到256
2. inverse close task？用输入的三个部分，video，title，speech，用其中一个部分作为query，然后去query其他的，就是去检索。用检索的loss，就是用dot product，然后做一个softmax
3. pretrain task 消融的结果？四个pretrained task：MLM(1.5%)，Inverse Close Task(1.5%), 对比学习（0.5%）Match（0.5%）融合模型（1.2%）
4. 没有用tag？因为感觉tag没有用，用了tag会掉点
5. 对比学习？SimCSE，在batch里面其他的是负样本，然后正样本构造是用同一个样本做两次dropout。


## Top3

### overview

![VB9aJi](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/VB9aJi.png)

### language model pretrain

![wwAatq](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wwAatq.png)

### pointwise training

![WkldCJ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/WkldCJ.jpg)

### pairwise wise

![WqZhxI](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/WqZhxI.png)

### offline evaluation strategy

![SbBfnQ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/SbBfnQ.png)

### Ablation

![ZawTwr](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ZawTwr.png)

### QA

1. 会用两个nextvlad，分别处理bert和vision，然后用fc把他们维度变成一致的？？？？
2. 激活函数，relu会有很多0，配合layer norm会好很多。relu'直接换成dice也会好很多
3. layernorm是在输出之前
4. ensemble，模型输出的space不对齐的话，会有比较大的影响。如果模型的cos相似度，在0.5+，就比较好哦，如果都是0相似度，就没意义
5. pairwise的label adjust，和第一名的思路差不多，但是都是把label的分布调整下。

## Top4

### Analysis

![LUiSuz](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/LUiSuz.png)

![LB9WPz](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/LB9WPz.png)

![ZKF1LL](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ZKF1LL.png)

### Model

base model
![cN6HsI](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/cN6HsI.png)

data augmentation and spatiotemporal constrastive
![sXefoz](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/sXefoz.png)

pretrain improve
![IN37MN](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/IN37MN.png)

Spatiotemporal Contrastive
![bKvCuA](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/bKvCuA.png)

SimCSE
![image.png](https://note.youdao.com/yws/res/f/WEBRESOURCEdcb9375112a06c50a6856d6c9b5e2c2f)

Text improve (ASR clean)
![1OltfE](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1OltfE.png)

Ablation
![K7igZG](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/K7igZG.png)

### QA

1. clean asr?用统计语言模型，先对每个标点分句，然后对每个句子算n-gram，然后用同一个视频的title和它算一个混淆度
2. 没做category的分类？因为category对subject的贡献不高，同category的存在subject不一样的，所以相对来说，tag信息对subject贡献高。
3. ensemble分类模型输出的向量长度？都是256
