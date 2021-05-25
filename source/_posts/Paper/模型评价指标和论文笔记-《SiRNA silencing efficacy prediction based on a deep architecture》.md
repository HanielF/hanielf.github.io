---
title: 模型评价指标 | 论文笔记 | SiRNA silencing efficacy prediction based on a deep architecture
tags:
  - DeepLearning
  - Paper
  - Note
  - AUC
  - ROC
  - siRNA
categories:
  - Papers
comments: true
mathjax: true
date: 2020-08-04 09:55:11
urlname: CNN-DNN-siRNA-AUC
---

<meta name="referrer" content="no-referrer" />

{% note info %}

## 前言

论文《SiRNA silencing efficacy prediction based on a deep architecture》笔记，以及对这篇论文的看法、理解。最后是记录了模型的一些评价指标。

{% endnote %}

<!--more-->

## 基本信息

| 论文名    | SiRNA silencing efficacy prediction based on a deep architecture    |
| --------- | ------------------------------------------------------------------- |
| 作者      | Han, Ye, He, Fei,Chen, Yongbing, Liu, Yuanning, Yu, Helong          |
| 时间      | 2018                                                                |
| 会议/期刊 | Spinger BMC Genomics                                                |
| 领域      | DL/Bioinformatics                                                   |
| 原文      | [链接](https://link.springer.com/article/10.1186/s12864-018-5028-8) |

## 研究背景

siRNA 通常是 19-29bp 长度的短 RNA 双链，在 RNA 干扰机制中有很重要的作用。但是人体的 RNA/DNA 并不像平时的语言一样具有可识别的语义，我们对这种序列的认知还比较少，同时，由于 siRNA 数据集比较小，大规模的有对应沉默效率的数据集难以获得，因此在预测 siRNA 沉默效率上具有一定的难度。

大部分的 siRNA 数据是没有对应的沉默效率的，最大的 Husken 数据集还是使用高通量技术在相同环境下实验得到。

现有的预测 siRNA 效率的方法不够好，很多已有的方法基于特征工程，可能导致偏差和特征的不完整。

数据集使用了 Huesken(2431) , Reynolds(248) , Vickers (80) , Haborth(44) , Takayuki(702) , Ui-Tei (62) and siRNAdb(500)

## 研究问题

本文主要使用深度学习中的 CNN 和 DNN 模型，结合 siRNA 的热力学参数，预测 RNA 干扰中的 siRNA 沉默效率。

## 解决方法

1. 使用 CNN 作为特征提取器，DNN 用于回归
2. 结合了热力学参数和序列特征，每个碱基使用 one-hot 编码，热力学参数的影响由[引文 21][21]和[引文 25][25]证明，计算由[引文 28][28]得到
3. 结合了靶基因上下游的部分核苷酸，见[引文 27][27]，21+2\*n 的长度
4. CNN 中实验了 $m \times 4 (2 \le m \le 20)$ 19 个卷积核的效果，挑出$PCC \ge 0.6$的作为子集
5. 实验了上下游碱基数量
6. 实验了 CNN 中和 DNN 中使用`Sigmoid`和`ReLU`的组合哪个最好
7. pooling 使用 avg pool 和 max pool
8. 使用 PCC 和 AUC 度量模型

模型结构图如下：

{% asset_img model.png %}

## 总结与思考

1. CNN 结合 DNN 在这个任务上 PCC 达到了 0.725，比她之前单纯的 CNN，PCC 为 0.717 的高，可能是增加的热力学参数起了作用，后面思路也可以往这个方向试试
2. 对卷积和和上下游碱基数都做了实验还是值得参考的，后面的可以自己实验上下游到底是多长比较适合自己
3. 数据集确实比单纯用 Husken 大了很多，后面还是得扩充自己数据集

4. 论文里面使用了 PCC 度量模型可以理解，但是为什么使用了 AUC 这个指标，AUC 不是只能用于分类吗。后面再次去理解了下 AUC 和 ROC 曲线。感觉还是应该只能用于分类问题，不然正例负例怎么划分？

## 关于 ROC 曲线和 AUC

### 混淆矩阵

假设有两个类别 0 和 1，模型预测得到结果，其中：

- 预测为 1 的为阳性(Positive)，预测为 0 的为阴性(Negative)
- 预测正确为真(True)，预测错误为假（False）

这样就可以得到混淆矩阵如下：

{% asset_img matrix.jpg %}

### 真阳率和假阳率

即 True Positive Rate 和 False Positive Rate

- $TPRate = {TP \over TP+FN}$，即：预测为 1 的里面，正例数占所有正样本的比例，也就是数据集中正例有多少预测对了
- $FPRate = {FP \over FP+TN}$，即：预测为 1 的里面，负例数占所有负样本的比例，也就是数据集中负例有多少预测错了

### ROC 曲线

- ROC 曲线的横轴就是`FPRate`，纵轴是`TPRate`
- 绘制 ROC 曲线对每个混淆矩阵计算 TPRate 和 FPRate 得到（x,y），然后连接（0,0）和（x,y）和(1,1)
- 在预测的结果是概率的时候，把每个结果概率作为阈值，把结果分为正例和负例，计算每个 TPRate 和 FPRate，然后描点，连线。

- 期望结果是`y>x`，总是想正例预测正确的比例大
- 二者相等就`y=x`时，分类器不起作用，和抛硬币一样随机
- `y<x`时即总是把正例预测为负例，负例预测为正例。可以把预测的类别反一下就可以得到`y>x`

### AUC

AUC 是 ROC 曲线下面的面积，越大越好。是能够度量分类器的分类能力的。

AUC 同时考虑了分类器对于正例和负例的分类能力，在样本不平衡的情况下，依然能够对分类器作出合理的评价。如在异常检测和欺诈场景之类。

## 关于其他性能度量标准

- 准确率：${TP+TN} \over {TP+TN+FP+FN}$，即预测正确的结果占总样本的百分比
- 精准率（Precision/查准率）：$TP \over {TP+FP}$，即在所有被预测为正的样本中，实际为正的样本占被预测为正的样本的比例
- 召回率（Reacll/查全率/真阳率）：$TP \over {TP+FN}$
- P-R(查准率-查全率)曲线，横轴是查全率，纵轴是查准率
- F1 分数：$2 \times 查准率 \times 查全率 \over {查准率+查全率}$，是 P-R（查准率-查全率）的平衡点
- 灵敏度（Sensitivity）：真阳率，$TP \over {TP+FN}$
- 特异度（Specificity）：1-假阳率，$TN \over {FP+TN}$

**条件概率表示，假设 X 为预测值，Y 为真实值：**

- 精准率=P(Y=1|X=1)
- 召回率=灵敏度=查全率=真阳率=P(X=1|Y=1)
- 特异度=P(X=0|Y=0)

## 其他参考文献

数据集来源：

| Dataset                                                                                                                                                                                               | Title                                                                                                                                    |
| ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- | ---------------------------------------------------------------------------------------------------------------------------------------- |
| [Huesken(2431)](https://www.researchgate.net/profile/Fred_Asselbergs/publication/7719918_Design_of_a_genome-wide_siRNA_library_using_an_artificial_neural_network/links/02bfe5100040c1a916000000.pdf) | Design of a genome-wide siRNA library using an artificial neural network                                                                 |
| [Reynolds(248)](https://sci-hub.tw/10.1038/nbt936)                                                                                                                                                    | Rational siRNA design for RNA interference                                                                                               |
| [Vickers(80)](https://sci-hub.tw/10.1074/jbc.m210326200)                                                                                                                                              | Efficient reduction of target RNAs by small interfering RNA and RNase H-dependent antisense agents                                       |
| [Haborth(44)](https://sci-hub.tw/10.1089/108729003321629638)                                                                                                                                          | Sequence, chemical, and structural variation of small interfering RNAs and short hairpin RNAs and the effect on mammalian gene silencing |
| [Takayuki(702)](https://sci-hub.st/10.1093/nar/gkl1120)                                                                                                                                               | Specific residues at every third position of siRNA shape its efficient RNAi activity                                                     |
| [Ui-Tei(62)](https://sci-hub.tw/10.1093/nar/gkh247)                                                                                                                                                   | Guidelines for the selection of highly effective siRNA sequences for mammalian and chick RNA interference                                |
| [siRNAdb(500)](https://sci-hub.st/10.1093/nar/gki294)                                                                                                                                                 | siRNAdb: a database of siRNA sequences                                                                                                   |

---

[21]: https://academic.oup.com/nar/article/35/18/e123/2402822
[25]: https://academic.oup.com/nar/article/32/3/936/2904484
[27]: https://www.sciencedirect.com/science/article/pii/S0888754313001468
[28]: https://www.researchgate.net/profile/John_Santalucia/publication/243786835_Parameters_for_an_expanded_nearest-neighbor_model_for_formation_of_RNA_duplexes_with_Watson-Crick_pairs/links/5ab420acaca272171003cb09/Parameters-for-an-expanded-nearest-neighbor-model-for-formation-of-RNA-duplexes-with-Watson-Crick-pairs.pdf
