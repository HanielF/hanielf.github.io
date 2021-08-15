---
title: UCAS自然语言处理
tags:
  - UCAS
  - NLP
  - Review
  - Notes
categories:
  - Notes
comments: true
mathjax: true
date: 2021-07-05 13:34:18
urlname: ucas-nlp-review
---

<meta name="referrer" content="no-referrer" />
{% note info %}

国科大自然语言处理（宗成庆）笔记

{% endnote %}

<!--more-->

## 期末提示

- 并列句计算，好像是边界距离计算，好像是编辑距离计算
- 概念弄清楚
- 分类 聚类 评价的指标
- 句法分析
  - 依存句法分析
  - 怎么评价性能
- 分词
  - 分词错误类型
  - 汉语分词
  - 并列句计算
- 机器翻译怎么度量译文质量

## 数学基础

1. 熵
   1. $H(X) = - \sum_{x\in X} p(x) log_2p(x)$
   2. $0log0 = 0$
   3. 单位为二进制位比特 (bit)
   4. 熵又称为自信息(self-information)，表示信源X每发一个信号所提供的平均信息量
   5. 熵也可以被视为描述一个随机变量 的不确定性的数量。一个随机变量的熵越大， 它的不确定性越大。那么，正确估计其值的可 能性就越小
2. 联合熵
   1. $H(X, Y) = - \sum_{x\in X}\sum_{y\in Y} p(x,y) log_2 p(x, y)$
   2. 就是描述一对随机变量平均所需要的信息量
3. 条件熵
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/4KtAKn.png' alt='4KtAKn'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/oKA1so.png' alt='oKA1so'/></div>
   3. 可以不用直接求条件熵，而是通过连锁法则算
   4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/IpeTzR.png' alt='IpeTzR'/></div>
   5. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/cDZ51B.png' alt='cDZ51B'/></div>
4. 熵率
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/n6X5IA.png' alt='n6X5IA'/></div>
5. 相对熵、KL散度、KL距离、Kullback-Leibler divergence
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/9LzsTS.png' alt='9LzsTS'/></div>
   2. 注意这个前面没有负号
6. 交叉熵
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/S86Ck0.png' alt='S86Ck0'/></div>
   2. $交叉熵=相对熵（KL散度）+熵$
7. 困惑度
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hrvNb3.png' alt='hrvNb3'/></div>
8. 互信息
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/nTSQpH.png' alt='nTSQpH'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/EIAbKt.png' alt='EIAbKt'/></div>
9. 双字耦合度
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/leizbn.png' alt='leizbn'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/oyMOUX.png' alt='oyMOUX'/></div>
10. 噪声信道模型
    1. 在信号传输的过程中都要进行双重性处理：一方面要 通过压缩消除所有的冗余，另一方面又要通过增加一定的可控冗余以保障输入信号经过噪声信道后可以很好地恢复原状。信息编码时要尽量占用少量的空间，但又必须保持 足够的冗余以便能够检测和校验错误。接收到的信号需要 被解码使其尽量恢复到原始的输入信号
    2. 噪声信道模型的目标就是优化噪声信道中信号传输的 吞吐量和准确率，其基本假设是一个信道的输出以一定的 概率依赖于输入。
    3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/lfYUEI.png' alt='lfYUEI'/></div>
    4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/VdpCaw.png' alt='VdpCaw'/></div>
11. 编辑距离
    1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/zsqOCq.png' alt='zsqOCq'/></div>

## 形式语言与自动机

### 形式语言定义

1. 形式语言定义
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/A8sYHd.png' alt='A8sYHd'/></div>
2. 形式语言推导
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/0GJf8D.png' alt='0GJf8D'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Iqc5mG.png' alt='Iqc5mG'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/L1ZdXY.png' alt='L1ZdXY'/></div>
   4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/aue0Ob.png' alt='aue0Ob'/></div>
3. 文法生成语言L(G)
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3gwO1G.png' alt='3gwO1G'/></div>

### 乔姆斯基四类文法

1. 正则文法(regular grammar, RG), 或称 3型文法
2. 上下文无关文法(context-free grammar, CFG), 或称2型文法
3. 上下文有关文法(context-sensitive grammar, CSG), 或称1型文法
4. 无约束文法(unrestricted grammar)，或称0型文法

如果一种语言能由几种文法所产生，则把这种语言 称为在这几种文法中受限制最多的那种文法所产生的 语言

#### 正则文法

3型文法
1. 定义
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/0639Xc.png' alt='0639Xc'/></div>
2. 转换为正则文法
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ILPEph.png' alt='ILPEph'/></div>

#### 上下文无关文法

2型文法
1. 定义
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1Z3Z4X.png' alt='1Z3Z4X'/></div>
2. 派生树
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/oxvfYX.png' alt='oxvfYX'/></div>
3. 二义性
   1. 一个文法 G，如果存在某个句子有不只一棵分析树与之对应，那么称这个文法是二义的
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/DUvLQ1.png' alt='DUvLQ1'/></div>

#### 上下文有关文法

1型文法
1. 定义
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xz0b0e.png' alt='xz0b0e'/></div>

#### 无约束文法

0型文法
1. 定义
   1. 、<div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Ej9BQC.png' alt='Ej9BQC'/></div>

### 自动机

自动机是有穷地表示无穷语言的另一种方法。每一 个语言的句子都能被某种自动机所接受。

- 有限(状态)自动机(finite automata, FA)，用的最多，对应正则文法（3型）
- 下推自动机(push-down automata, PDA)，对应上下文无关文法（2型）
- 线性带限自动机(linear bounded automata)，对应上下文有关文法（1型）
- 图灵机(Turing Machine)，没有实际意义，一般不用，对应无约束文法（0型）

#### 有限状态自动机

1. 确定性有限自动机(deterministic finite automata, DFA)
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/0jL23K.png' alt='0jL23K'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/rLJknI.png' alt='rLJknI'/></div>
   3. 终止状态用双圈表示，起始 状态用有“开始”标记的箭头表示
   4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/baKvCp.png' alt='baKvCp'/></div>
2. 非确定性有限自动机(non-deterministic finite automata, NFA)
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/G0YXAd.png' alt='G0YXAd'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/BncGqs.png' alt='BncGqs'/></div>

### 正则文法和有限自动机关系

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/mzcl0N.png' alt='mzcl0N'/></div>
2. 文法转换为自动机
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/9coCJG.png' alt='9coCJG'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wzN1fx.png' alt='wzN1fx'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Tb6Sj7.png' alt='Tb6Sj7'/></div>
3. 自动机转文法
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/rXD7HI.png' alt='rXD7HI'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/nR45tw.png' alt='nR45tw'/></div>
   3. 转换结果
      1. $G = (V_N, V_T, P, S)$
      2. $V_N={S, A}$
      3. $V_T = {0, 1}$
      4. $P = \{ S\rightarrow 0A, A \rightarrow 1S | 0 \}$

## 语料库与语言知识库

1. 语料库定义(corpus)
   - 用于存放语言数据的文件(语言数据库)。
2. 语料库语言学(corpus linguistics)
   - 基于语料库进行的语言学研究。研究自然语言文本的采集、存储、检索、统计、 词性和句法及语义信息的标注、以及具有上述功能的 语料库在语言定量分析、词典编纂、作品风格分析和 人类语言技术等领域中的应用。
1. 类型
   1. 按内容构成和目的划分
      1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pWzZ3b.png' alt='pWzZ3b'/></div>
   2. 按语言种类划分和标注
      1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/d78IOu.png' alt='d78IOu'/></div>
   3. 平衡语料库
      1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3YS3YK.png' alt='3YS3YK'/></div>
   4. 平行语料库
      1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xRQ2cf.png' alt='xRQ2cf'/></div>
      2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/73e7ia.png' alt='73e7ia'/></div>
   5. 共时语料库和历时语料库
      1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/QnLIjp.png' alt='QnLIjp'/></div>
      2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hXgxk0.png' alt='hXgxk0'/></div>
2. 问题和现状
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/J9oFKI.png' alt='J9oFKI'/></div>

## 语言模型

### 传统语言模型

#### n元文法

1. 计算语句先验
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/mer5Zp.png' alt='mer5Zp'/></div>
2. 历史基元过多问题
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/HLHSqa.png' alt='HLHSqa'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hGyUpt.png' alt='hGyUpt'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/oOPlHf.png' alt='oOPlHf'/></div>
3. 定义
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/WEyp5V.png' alt='WEyp5V'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/77bC61.png' alt='77bC61'/></div>
4. 举例
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Fo0Hvv.png' alt='Fo0Hvv'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/2eXJGf.png' alt='2eXJGf'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/BTWGv1.png' alt='BTWGv1'/></div>

#### 参数估计

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/NLoTjl.png' alt='NLoTjl'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/iam1RN.png' alt='iam1RN'/></div>
3. 问题是，存在数据匮乏的情况，导致一些概率算出来是0， 因此要进行数据平滑

#### 数据平滑

1. 基本思想
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/OSdqEw.png' alt='OSdqEw'/></div>
2. 加一法
   1. 其实就是分子加1，分母加上所有的类（词汇表V大小）
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/tpr5BE.png' alt='tpr5BE'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Yoz8gY.png' alt='Yoz8gY'/></div>
3. 减值法/折扣法
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/zawpfF.png' alt='zawpfF'/></div>
   2. 自己查下，考试应该不考
   3. Good-Turning估计法
   4. 后退法
   5. 绝对减值法
      1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/UlFscN.png' alt='UlFscN'/></div>
      2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/vzTlnV.png' alt='vzTlnV'/></div>
      3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/CK8Vib.png' alt='CK8Vib'/></div>
   6. 线性减值法
      1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/8DZbbq.png' alt='8DZbbq'/></div>
   7. 四种减值法的比较
      1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/0S2XO2.png' alt='0S2XO2'/></div>
4. 删除插值法
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Use3hq.png' alt='Use3hq'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3sSHtC.png' alt='3sSHtC'/></div>

#### 语言模型自适应

1. 方法
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/PbjjmG.png' alt='PbjjmG'/></div>

#### n元文法应用

##### 分词

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/h4gTu2.png' alt='h4gTu2'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/GohKle.png' alt='GohKle'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/UADc8i.png' alt='UADc8i'/></div>
4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/fl3r5s.png' alt='fl3r5s'/></div>
5. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/SLFH06.png' alt='SLFH06'/></div>
6. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/4uGIp3.png' alt='4uGIp3'/></div>
7. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/SlAJ61.png' alt='SlAJ61'/></div>
8. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/8PRkKj.png' alt='8PRkKj'/></div>
9. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/QyZzEE.png' alt='QyZzEE'/></div>
10. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pGVDXG.png' alt='pGVDXG'/></div>

##### 分词与词性标注一体化方法

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/JQf7bx.png' alt='JQf7bx'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/NmF697.png' alt='NmF697'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/NyBggw.png' alt='NyBggw'/></div>
4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/RbxiwB.png' alt='RbxiwB'/></div>
5. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/SixCcQ.png' alt='SixCcQ'/></div>
6. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/XXVw6J.png' alt='XXVw6J'/></div>

### 神经语言模型

1. 传统语言模型的问题
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Nb2IDK.png' alt='Nb2IDK'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/WtXBRs.png' alt='WtXBRs'/></div>

#### 前馈神经网络语言模型

这部分省略

#### 循环神经网络语言模型

RNN和LSTM，以及Self-attention的内容，也省略

附录有BP算法的推导，可以自己看

### 文本表示

#### 基础

1. 概念
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/QD0jV4.png' alt='QD0jV4'/></div>
2. 特征项
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/8F6kjL.png' alt='8F6kjL'/></div>
3. 特征项权重：词频，TF-IDF，逆文档频率
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/5ZkYOc.png' alt='5ZkYOc'/></div>
4. 规范化Normalization
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/TPch3B.png' alt='TPch3B'/></div>

#### 分布式表示

##### 表示学习模型

1. 文本概念表示模型：以（概率）潜在语义分析和潜在狄利克雷分布为代表的主题模型，旨在挖掘 文本中的隐含主题或概念，文本将被表示为主题 的分布向量
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wHrikn.png' alt='wHrikn'/></div>
2. 深度表示学习模型：通过深度学习模型以最优化特定目标函数（例如语言模型似然度）的方式在 分布式向量空间中学习文本的低维实数向量表示

##### 词语的表示学习

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/nMq1HE.png' alt='nMq1HE'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/6zzDLR.png' alt='6zzDLR'/></div>

1. NNLM模型
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7xuDs1.png' alt='7xuDs1'/></div>
2. C&W模型
   1. 中心词进行随机替换
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/iGmEOz.png' alt='iGmEOz'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Z6lwRb.png' alt='Z6lwRb'/></div>
3. word2vec: CBOW和Skip-Gram
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/T0DfuB.png' alt='T0DfuB'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/cnuyDy.png' alt='cnuyDy'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ETseu1.png' alt='ETseu1'/></div>
   4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1kFxdA.png' alt='1kFxdA'/></div>
4. Glove模型
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/GgkdYe.png' alt='GgkdYe'/></div>
   2. 太多了，省略
5. 负采样和噪声对比估计
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/cAN7bM.png' alt='cAN7bM'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/UXoSDI.png' alt='UXoSDI'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/TIYeVg.png' alt='TIYeVg'/></div>
   4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/74GXZu.png' alt='74GXZu'/></div>
   5. 负采样技术的目标函数与噪声对比估计相同，但是不同于
噪声对比估计方法，负采样技术不对样本集合进行概率归
一化，而直接采用神经网络语言模型输出
6. 字词混合的表示学习
   1. 词语由字或字符构成，一方面词语作为不可分割的单元可以获得一个表示；另一方面词语作为字的组合，通过字的表示也可以获得一个表示；两种表示结合得到更优表示
   2. 在低维、稠密的实数向量空间中，相似的词聚集在一起， 在相同的历史上下文中具有相似的概率分布

##### 短语的表示学习

1. 词袋模型
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/dygNUx.png' alt='dygNUx'/></div>
2. 递归自编码器
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/bbp65q.png' alt='bbp65q'/></div>
3. 双语约束模型
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/RILezO.png' alt='RILezO'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ornsRQ.png' alt='ornsRQ'/></div>

##### 句子的表示学习

1. 词袋模型
   1. 平均
   2. 加权平均
2. PV-DM模型
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/aGkCb4.png' alt='aGkCb4'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/sipIq0.png' alt='sipIq0'/></div>
3. PV-DBOW模型
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/fD86KP.png' alt='fD86KP'/></div>
4. Skip-Thought模型
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/B98gwn.png' alt='B98gwn'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/FIzhtA.png' alt='FIzhtA'/></div>
5. CNN模型
   1. 对于一个句子，卷积神经网络CNN以每个词的词向量为输入，通过顺 序地对上下文窗口进行卷积（Convolution）总结局部信息，并利用池 化层（Pooling）提取全局的重要信息，再经过其他网络层（卷积池化 层、Dropout层、线性层等），得到固定维度的句子向量表达，以刻 画句子全局性的语义信息
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/IBrKZV.png' alt='IBrKZV'/></div>

##### 文档的表示学习

1. 词袋模型
2. CNN模型
3. 层次化模型
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/aYw5C1.png' alt='aYw5C1'/></div>

### 预训练模型

#### ELMo预训练模型

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/QOtrjw.png' alt='QOtrjw'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/kvikby.png' alt='kvikby'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/oZ0Bd7.png' alt='oZ0Bd7'/></div>
4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Piw4Jo.png' alt='Piw4Jo'/></div>
5. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/fcW8KX.png' alt='fcW8KX'/></div>

### GPT预训练模型

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/F9zWTP.png' alt='F9zWTP'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/oRFTOw.png' alt='oRFTOw'/></div>
3. 下游任务
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/eBgaSI.png' alt='eBgaSI'/></div>
   2. 分类，对最后一个hidden进行变换后softmax
4. 优点：
   1. 任务统一: 大部分语言处理任务形式化为语言模型任务
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hu0T7Z.png' alt='hu0T7Z'/></div>

### BERT预训练模型

参考以前的博客[BERT-预训练源码理解](https://hanielxx.com/MachineLearning/2021-02-23-bert-create-pretrain-data-analysis)和[论文笔记 | Attention Is All You Need](https://hanielxx.com/Papers/2020-08-15-attention-is-all-you-need)

## 概率图模型

概率图模型的演变

<div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/BG66M9.png' alt='BG66M9'/></div>

### 马尔科夫模型

1. 描述
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1EuLfN.png' alt='1EuLfN'/></div>
2. 假设
   1. 当前状态只和前一个状态有关
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/KomjQi.png' alt='KomjQi'/></div>
   3. 状态的转移和时间t无关
   4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7COEYj.png' alt='7COEYj'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Qdfobv.png' alt='Qdfobv'/></div>
4. 和NFA关系
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/JrlYTh.png' alt='JrlYTh'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/M4j6eH.png' alt='M4j6eH'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/L14DVB.png' alt='L14DVB'/></div>

### 隐马尔科夫模型HMM

#### 模型介绍

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/YwPViM.png' alt='YwPViM'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/HJPBvn.png' alt='HJPBvn'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/NV7Zik.png' alt='NV7Zik'/></div>
4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/DaQLAl.png' alt='DaQLAl'/></div>

#### 核心问题

##### 给定HMM求观察序列

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/awVdVr.png' alt='awVdVr'/></div>

##### 快速计算观察序列概率 $p(O|\mu)$

前向算法和后向算法

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/s0BNax.png' alt='s0BNax'/></div>
2. 上面的方法，会导致有 $N^T$个状态序列，指数级组合爆炸

前向算法

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wHyhPU.png' alt='wHyhPU'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/6jeAb4.png' alt='6jeAb4'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1h6HH5.png' alt='1h6HH5'/></div>
4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/22ywb4.png' alt='22ywb4'/></div>
5. 时间复杂度：$O(N^2T)$

后向算法

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/C3gMGt.png' alt='C3gMGt'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/K1e68X.png' alt='K1e68X'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/eaWfbe.png' alt='eaWfbe'/></div>
4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Vnn6Az.png' alt='Vnn6Az'/></div>
5. 时间复杂度：$O(N^2T)$

#### 发现最优状态序列

如何发现“最优”状态序列能够“最好地解释”观察序列？

维特比算法

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Lx7HWu.png' alt='Lx7HWu'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/UgUT9u.png' alt='UgUT9u'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xtnKV1.png' alt='xtnKV1'/></div>

#### 模型参数学习

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/62tZpf.png' alt='62tZpf'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hmouIt.png' alt='hmouIt'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Uitu5d.png' alt='Uitu5d'/></div>
4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/qVapaE.png' alt='qVapaE'/></div>
5. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/UUpMQb.png' alt='UUpMQb'/></div>

### HMM应用

1. 分词
2. 词性标注

### 条件随机场模型CRF

省略

## 形态分析、汉语分词与词性标注

1. 词性标注（消岐）中的并列鉴别规则
2. 标出命名实体，说明类型

### 英语形态分析

1. 有规律变化单词的形态还原
2. 动词、名词、形容词、副词不规则变化单词的形态还原
3. 对于表示年代、时间、百分数、货币、序数词的数字形态还原
4. 合成词的形态还原

形态分析的一般方法
1) 查词典，如果词典中有该词，直接确定该词的原形；
2) 根据不同情况查找相应规则对单词进行还原处理，如果还原后在词典中找到该词，则得到该词的原形；如果找不到相 应变换规则或者变换后词典中仍查不到该词，则作为未登 录词处理；
3) 进入未登录词处理模块。

### 汉语自动分词

1. 交集型歧义
   1. 中国/ 人为/ 了/ 实现/ 自己/ 的/ 梦想
   2. 中国人/ 为了/ 实现/ 自己/ 的/ 梦想
   3. 中/ 国人/ 为了/ 实现/ 自己/ 的/ 梦想
   4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wK9RSZ.png' alt='wK9RSZ'/></div>
2. 组合型歧义
   1. 门/ 把/ 手/ 弄/ 坏/ 了/  。
   2. 门/ 把手/ 弄/ 坏/ 了/  。
3. 分词基本原则
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wGvI8W.png' alt='wGvI8W'/></div>
4. 辅助原则
   1. div<div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/agpOW5.png' alt='agpOW5'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/nYwi51.png' alt='nYwi51'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/vDMQpc.png' alt='vDMQpc'/></div>
   4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/qweK8J.png' alt='qweK8J'/></div>
5. 评价指标
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/rKPJRt.png' alt='rKPJRt'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/F7e1Uy.png' alt='F7e1Uy'/></div>

### 汉语自动分词方法

1. 最大匹配法
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/rXR7j4.png' alt='rXR7j4'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/WekZiC.png' alt='WekZiC'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1b88gN.png' alt='1b88gN'/></div>
   4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/NBufKR.png' alt='NBufKR'/></div>
2. .最少分词法(最短路径法) 有词典切分
   1. 基本思想 <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/RiPvBo.png' alt='RiPvBo'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/AH3HXc.png' alt='AH3HXc'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/L6UtHY.png' alt='L6UtHY'/></div>
   4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/KTbTEd.png' alt='KTbTEd'/></div>
3. 基于语言模型的分词方法
   1. 基本思路 <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7BfEvd.png' alt='7BfEvd'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/okF7us.png' alt='okF7us'/></div>
4. 基于HMM的分词
5. 由字构词 (基于字标注)的分词方法
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3KhY0L.png' alt='3KhY0L'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/FRgOGF.png' alt='FRgOGF'/></div>
6. 生成式和判别式模型区别
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/aY38KN.png' alt='aY38KN'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/rgywBR.png' alt='rgywBR'/></div>
7. 生成式方法与区分式方法的结合
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/dOBzEQ.png' alt='dOBzEQ'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/gmaCw4.png' alt='gmaCw4'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/CTPScM.png' alt='CTPScM'/></div>
   4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/sJannE.png' alt='sJannE'/></div>

### 未登录词

1. 命名实体
   1. 人名、地名、机构名、时间、日期、货币、百分比，数字如果在这些类别中的话就算命名实体，如果不在，通常也有单独的数字识别任务
2. 其他新词，如专业词汇和新的普通词

### 子词切分（BPE算法）

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/rzE69U.png' alt='rzE69U'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/kx3Lz7.png' alt='kx3Lz7'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/05EFmu.png' alt='05EFmu'/></div>

例子
1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/VL7oRq.png' alt='VL7oRq'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/qc9sb9.png' alt='qc9sb9'/></div>

### 词性标注

1. 标注方法
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xfXoK9.png' alt='xfXoK9'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/sSc9uP.png' alt='sSc9uP'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/8OUrEV.png' alt='8OUrEV'/></div>
   4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/OY6Zwp.png' alt='OY6Zwp'/></div>
   5. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/o8etvc.png' alt='o8etvc'/></div>

## 句法分析

### 短语结构分析

- 基于CFG规则的分析方法
  - 线图分析法
  - CYK分析法
- 基于PCFG的分析法
- 句法分析性能评估
- 局部句法分析

#### 线图分析法

1. 三种策略
   1. 自底向上 (Bottom-up)
   2. 从上到下 (Top-down)
   3. 从上到下和从下到上结合
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/NXf5ep.png' alt='NXf5ep'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/LMmCOn.png' alt='LMmCOn'/></div>
4. 例子
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/12g1PJ.png' alt='12g1PJ'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/47efc0.png' alt='47efc0'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/CsyLZ2.png' alt='CsyLZ2'/></div>
   4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/mWyZUn.png' alt='mWyZUn'/></div>
   5. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Q0h00l.png' alt='Q0h00l'/></div>
   6. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Kaam05.png' alt='Kaam05'/></div>
   7. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/QUP4Ys.png' alt='QUP4Ys'/></div>

#### CYK分析法

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/vRxA1l.png' alt='vRxA1l'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ND8JbD.png' alt='ND8JbD'/></div>

例子
1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/NsyP3k.png' alt='NsyP3k'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/KqxED3.png' alt='KqxED3'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xhCf2y.png' alt='xhCf2y'/></div>

#### 基于PCFG的分析法

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/EHl9uU.png' alt='EHl9uU'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/sei9sx.png' alt='sei9sx'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/zJze5n.png' alt='zJze5n'/></div>
4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xSxdzT.png' alt='xSxdzT'/></div>
5. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/MxbpCo.png' alt='MxbpCo'/></div>
6. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/EVlYPo.png' alt='EVlYPo'/></div>

#### 句法分析性能评估

1. 精度precision
2. 召回recall
3. F1分数
4. 交叉括号数：（括号有重叠都算）：一棵分析树中与其他分析树中边界相交叉的成分个数的平均值。通常是对应位置的短语比较
5. 交叉准确率
6. 词性标注准确率

### 依存句法分析

在依存语法理论中，“依存”就是指词与词之间支配与被 支配的关系，这种关系不是对等的，而是有方向的。处于支配 地位的成分称为支配者(governor, regent, head)，而处于被支配 地位的成分称为从属者(modifier, subordinate, dependency)。

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/oNfzPc.png' alt='oNfzPc'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/AWOPEp.png' alt='AWOPEp'/></div>

#### 依存句法四条公理

(1) 一个句子只有一个独立的成分；即单一父结点(single headed)
(2) 句子的其他成分都从属于某一成分；即连通(connective)
(3) 任何一成分都不能依存于两个或多个成分；即无环(acyclic)
(4) 如果成分A直接从属于成分B，而成分C在句子中位于A和B
之间，那么，成分C或者从属于A，或者从属于B，或者从 属于A和B之间的某一成分；即可投射(projective)

#### 依存分析方法

目前依存句法结构描述一般采用有向图方法或依存树方法，
所采用的句法分析算法可大致归为以下4类：
- 生成式分析方法(generative parsing)
- 判别式分析方法(discriminative parsing)
- 决策式(确定性)分析方法(deterministic parsing)
- 基于约束满足的分析方法(constraint satisfaction parsing)

**Arc-eager algorithmX算法**
1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/V3tuqI.png' alt='V3tuqI'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/DHMGF0.png' alt='DHMGF0'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/H6bYpv.png' alt='H6bYpv'/></div>
4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pA2BGM.png' alt='pA2BGM'/></div>
5. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ZbW6wN.png' alt='ZbW6wN'/></div>

**评价指标**
1. 无标记依存正确率(unlabeled attachment score, UA 或 UAS)：所有词中找到其正确支配词的词所占的百分比，没有找到支配词的词(即根结点)也算在内
2. 带标记依存正确率(labeled attachment score, LA 或 LAS)：所有词中找到其正确支配词并且依存关系类型也标注正确的词所占的百分比，根结点也算在内。就是在UA里面找依存关系标注正确的
3. 依存正确率(dependency accuracy, DA)：所有非根结点词中找到其正确支配词的词所占的百分比。就是UA的分子分母同时减去1
4. 根正确率(root accuracy, RA)：有两种定义方式：
   - (1)正确根结点的个数与句子个数的比值；
   - (2)所有句子中找到正确根结点的**句子所占的百分比**。
   - 对单根结点语言或句子来说，二者是等价的。
5. 完全匹配率(complete match, CM)：所有句子中无标记依存结构完全正确的**句子所占的百分比**

#### 短语结构和依存关系转换

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3UxmT2.png' alt='3UxmT2'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/NYzV2G.png' alt='NYzV2G'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/qdY0M5.png' alt='qdY0M5'/></div>
4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/oSZaem.png' alt='oSZaem'/></div>
5. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/J70c86.png' alt='J70c86'/></div>

### 基于深度学习的句法分析

略

### 英汉句法结构特点对比

略

## 语义分析

### 语义网络的概念

语义网络通过由实体、概念或动作、状态及语义关系组成的 有向图来表达知识、描述语义。

1. 有向图：图的结点表示实体或概念，图的边表示实体或概念
之间的关系。
2. 边的类型：
   - “是一种抽象(IS-A)”: A到B的边表示“A是B的一种特例”;
   - “是一部分(PART-OF)”: A到B的边表示“A是B的一部分”;
   - “是属性(IS)”：A到B的边表示“A是B的一种属性”；
   - “拥有/占用(HAVE)”: A到B的边表示 “A拥有B”；
   - “次序在先/前(BEFORE)”: A到B的边表示 “A在B之前”
3. 事件的语义网络表示
   1. 当语义网络表示事件时，结点之间的关系可以是施事、受事、 时间等。这里所说的“事件”指某个具体的动作或状态，并非 我们日常生活中所说的事件。
4. 事件的语义关系
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/HBWoCI.png' alt='HBWoCI'/></div>

### 词义消岐

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/fT6XwB.png' alt='fT6XwB'/></div>
2. 其他略

### 语义角色标注

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/6C3LxP.png' alt='6C3LxP'/></div>
1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/P4avHL.png' alt='P4avHL'/></div>
1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/PfH4JQ.png' alt='PfH4JQ'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/dIiHO8.png' alt='dIiHO8'/></div>

## 语义的表征与解码

略

## 篇章分析

略

## 机器翻译

1. 问题和现状
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/nPIqor.png' alt='nPIqor'/></div>
2. 基本观点
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/EbHxCA.png' alt='EbHxCA'/></div>

### 基本翻译方法

1. 直接转换法
   1. 从源语言句子的表层出发，将单词、短语或句子 直接置换成目标语言译文，必要时进行简单的词序调 整。对原文句子的分析仅满足于特定译文生成的需要。 这类翻译系统一般针对某一个特定的语言对，将分析 与生成、语言数据、文法和规则与程序等都融合在一 起
2. 基于规则的翻译方法
   1. 对源语言和目标语言均进行适当描述、把翻译机制与 语法分开、用规则描述语法的实现思想，这就是基于 规则的翻译方法
   2. 步骤 <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/S3N0nO.png' alt='S3N0nO'/></div>
   3. 评价：<div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/RfUXiN.png' alt='RfUXiN'/></div>
3. 基于中间语言的翻译方法
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/OF4wtQ.png' alt='OF4wtQ'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/unrXxh.png' alt='unrXxh'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/zec67l.png' alt='zec67l'/></div>
5. 基于语料库的翻译方法
   1. 基于事例的翻译方法
      1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/rrPIMH.png' alt='rrPIMH'/></div>
      2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3cR0Wo.png' alt='3cR0Wo'/></div>
   2. 统计机器翻译
   3. 神经网络机器翻译
6. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ynUfbt.png' alt='ynUfbt'/></div>

## 统计机器翻译

1. 基本思想
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/UkZOgO.png' alt='UkZOgO'/></div>
2. 噪声信道模型
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/j02jGI.png' alt='j02jGI'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/CTcGwa.png' alt='CTcGwa'/></div>
3. 统计翻译三个关键问题
   1. 估计语言模型概率 $p(T)$；
      1. n-gram问题
      2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/fxGc1k.png' alt='fxGc1k'/></div>
   2. 估计翻译概率 $p(S|T)$；
      1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/YQd0sG.png' alt='YQd0sG'/></div>
      2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/J3h1qa.png' alt='J3h1qa'/></div>
      3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/aURzN4.png' alt='aURzN4'/></div>
      4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/aQl752.png' alt='aQl752'/></div>
      5. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/bfLzc1.png' alt='bfLzc1'/></div>
      6. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pNF8Vo.png' alt='pNF8Vo'/></div>
   3. 快速有效地搜索T  使得 $p(T) \times p(S | T)$ 最大。

## 神经网络机器翻译

1. 对比
   1. 统计机器翻译
      1. 数据稀疏（离散符号表示）
      2. 复杂结构无能为力
      3. 强烈依赖先验
      4. 可解释性高
   2. 神经机器翻译
      1. 连续稠密分布式表示
      2. 双语平行语料库 相同
      3. 可解释性低
      4. 翻译性能高，简单有效
2. 系统融合方法
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pNI7lu.png' alt='pNI7lu'/></div>
   2. 句子级系统融合
      1. 针对同一个源语言句子，利用最小贝叶斯风险解码或重打分方法比较多个机器翻译系统的译文输 出，将最优的翻译结果作为最终的一致翻译结果
   3. 短语级系统融合
      1. 利用多个翻译系统的输出结果，重新抽取短语翻译规则集合，并利用新的短语翻译规则进行重新 解码
   4. 词语级系统融合
      1. 首先将多个翻译系统的译文输出进行词语对齐，构建一个混淆网络，对混淆网络中的每个位置的 候选词进行置信度估计，最后进行混淆网络解码
   5. 深度学习的系统融合
      1. 首先将多个翻译系统的译文进行编码，然后采用层次注意机制模型预测每个时刻目标语言的输出： 底层注意机制决定该时刻更关注哪个源语言片段， 上层注意机制决定该时刻更依赖哪个系统的输出
3. 译文评估方法
   1. 句子错误率：译文与参考答案不完全相同的句子为错误句子。错误句子占全部译文的比率。
   2. 单词错误率(Multiple Word Error Rate on Multiple Reference, 记作 mWER)：分别计算译文与每个参考译文的编辑距离，以最短的为评分依据，进行归一化处理
   3. 与位置无关的单词错误率 ( Position independent mWER, 记作mPER )：不考虑单词在句子中的顺序
   4. METEOR 评测方法：对候选译文与参考译文进行词对齐，计算词汇完全 匹配、词干匹配、同义词匹配等各种情况的准确率 ( P)、召回率(R)和F平均值
   5. BLEU方法
      1. 将机器翻译产生的候选译文与人翻译的多个参考 译文相比较，越接近，候选译文的正确率越高。
      2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/NAuVJt.png' alt='NAuVJt'/></div>
      3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/AI0dxk.png' alt='AI0dxk'/></div>
      4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/8QT3Ds.png' alt='8QT3Ds'/></div>
   5. NIST
      1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/KKBq5L.png' alt='KKBq5L'/></div>
      2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/lgq0Uq.png' alt='lgq0Uq'/></div>

## 文本分类和聚类

### 传统机器学习方法

### 深度学习方法

### 文本分类性能评估

1. TP
2. FP
3. TN
4. FN
5. Precision
6. Recall
7. Acc
8. 宏平均：
   1. 分别算完，再加起来除以类别数
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/MTdEXG.png' alt='MTdEXG'/></div>
9. 微平均
   1. 整体的TP等加起来，只算一次
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/j3knan.png' alt='j3knan'/></div>
10. P-R曲线
    1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/bXBRFk.png' alt='bXBRFk'/></div>
11. ROC曲线
    1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/CLgOq6.png' alt='CLgOq6'/></div>

### 文本聚类指标

#### 两个文本对象相似度

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7JsKJC.png' alt='7JsKJC'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/m5k96o.png' alt='m5k96o'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7ykEro.png' alt='7ykEro'/></div>
4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/WvyzqJ.png' alt='WvyzqJ'/></div>

#### 两个文本集合相似度

1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/tdhVHA.png' alt='tdhVHA'/></div>
2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/tf5h15.png' alt='tf5h15'/></div>
3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/yX4Ini.png' alt='yX4Ini'/></div>
4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/LOBhoM.png' alt='LOBhoM'/></div>
5. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/m16sd6.png' alt='m16sd6'/></div>
6. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/TBlQDn.png' alt='TBlQDn'/></div>

#### 文本和簇的相似度

- 样本与簇之间的相似性通常转化为样本间的相似度或 簇间的相似度进行计算。
- 如果用均值向量来表示一个簇，那么样本与簇之间的 相似性可以转化为样本与均值向量的样本相似性。
- 如果将一个样本视为一个簇，那么就可以采用前面介 绍的簇间的相似性度量方法进行计算

#### 衡量聚类指标

1. 外部指标
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/4xa3wj.png' alt='4xa3wj'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/neYvWw.png' alt='neYvWw'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/HK182v.png' alt='HK182v'/></div>
   4. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xYojNN.png' alt='xYojNN'/></div>
   5. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/9Yp2Or.png' alt='9Yp2Or'/></div>
2. 内部标准
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/a7nkUU.png' alt='a7nkUU'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Y5kJt3.png' alt='Y5kJt3'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/fR4wEp.png' alt='fR4wEp'/></div>

## 信息抽取

- 信息抽取是从非结构化、半结构化的自然语言文本 （如网页新闻、学术文献、社交媒体等）中抽取实 体、实体属性、实体间的关系以及事件等事实信息 ，并形成结构化数据输出的一种文本数据挖掘技术 [ Sarawagi, 2008]
- 从非结构化文本到机器可读信息的一种转换技术 [Wu, 2010]

### 命名实体识别

1. 概念
   - 信息抽取的一项基础任务
   - 自动识别出文本中指定类别的实体，包括人名、地名、 机构名、日期、时间和货币等七类
   - 人名（例如“乔布斯”）、地名（例如“北京”）、组 织机构名（例如“中国科学院”）、时间（例如“10点 30分”）、日期（例如“2017年6月1日”）、货币（例 如“1000美元”）、百分比（例如“百分之五十”）
   - 由于时间、日期、货币和百分比规则性强，利用模板或 正则表达式基本可处理
   - 人名、地名和组织机构名是关注重点
2. 目标
   - 包含两个任务：实体检测和实体分类
   - 实体检测：检测出文本中哪些词串属于实体，也即发现 实体的左边界和右边界
   - 实体分类：判别检测出的实体具体属于哪个类别
3. 方法（将实体检测和分类任务联合建模）
   1. 基于规则的方法
   2. 基于有监督的机器学习方法
      1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1t3iG1.png' alt='1t3iG1'/></div>
      2. 基于HMM的NER
      3. 基于CRF的NER
      4. 基于DNN的NER
4. 评价指标
   1. 以实体为单位
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/9XCO8l.png' alt='9XCO8l'/></div>

### 实体消岐

1. 概念
   1. 一篇文档中同一实体可能有多种不同的指称（Mention） ，不同文档中相同名称的实体也可能表示不同的含义。 前者需要共指消解（Coreference Resolution）技术，后者 需要实体链接（Entity Linking）技术
   2. 共指消解就是为文本中的指称确定其具体实体的过程(一篇文档中同一实体)
      1. \[李克强\]接见\[美国总统\]\[特朗普\]，\[他\]首先对\[美国总统\]的到来表示热烈欢迎。
      2. “美国总统”和“特朗普”表示同一实体，“他”实际 指代的是“李克强”，即例子中的所有指称聚为两类
   3. 实体链接就是确定实体指称所对应的真实世界实体的过程(不同文档中相同名称)
      1. “张杰多次参加春节联欢晚会。原上海交通大学校长张杰院士出任中国科学院副院长。
      2. 在第一句话中，“张杰”指的是“歌手张杰”，而第二句话中的“张杰”指的是“中科院院士张杰”
2. 典型方法
3. 实体链接评价
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/2OZzdZ.png' alt='2OZzdZ'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/41Q0A0.png' alt='41Q0A0'/></div>

### 关系抽取

1. 概念
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hjXqkd.png' alt='hjXqkd'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/agnCE1.png' alt='agnCE1'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/gHP3rR.png' alt='gHP3rR'/></div>
2. 评价指标
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pAC5pO.png' alt='pAC5pO'/></div>

### 事件抽取

1. 概念
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/mVIfnI.png' alt='mVIfnI'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/OOjrkU.png' alt='OOjrkU'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/JJ4YwC.png' alt='JJ4YwC'/></div>
2. 实例
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7gdLg8.png' alt='7gdLg8'/></div>
   2. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/yItl9M.png' alt='yItl9M'/></div>
   3. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7Hm1dY.png' alt='7Hm1dY'/></div>
3. 典型方法
   1. 基于联合模型的事件抽取方法
   2. 基于分布式模型的事件抽取方法
4. 评价指标
   1. <div style="width:500px;word-wrap:break-word;white-space:normal"><img src='https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/jshQYf.png' alt='jshQYf'/></div>
   2. 几乎所有模型都将事件抽取任务分解为触发词识别和事件角色 分类两个步骤，其中触发词识别又可分解为触发词定位与事件 类型分类两个子任务，事件角色分类又可分解为事件元素识别 与角色分类两个子任务；因此，客观评测一般对四个子任务分 别进行测试
   3. 若模型找到了触发词的在事件描述中的具体位置， 那么触发 词定位正确；在定位正确的基础上，如果事件的类型也预测 正确，那么事件类型分类任务得到正确结果。若某候选事件 元素被正确识别为触发词的关联属性，那么事件元素识别是 正确的，如果正确识别的事件元素进一步被预测为正确的事 件角色，那么最终的事件角色分类也是正确的。根据匹配结 果，利用准确率（ Precision）、召回率（Recall）和F1值分别度量各个子任务的性能

## 人机对话

### 任务型对话系统

### 聊天型对话系统

### 问答系统

### 系统搭建方法
