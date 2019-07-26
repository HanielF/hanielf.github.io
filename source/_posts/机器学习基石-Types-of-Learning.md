---
title: 机器学习基石--Types of Learning
comments: true
mathjax: true
date: 2019-04-07 20:54:44
tags: [机器学习基石,MachineLearning]
categories: [MachineLearning,机器学习基石,]
urlname: machinelearning-types-of-learning
---

<meta name="referrer" content="no-referrer" />

{% note info %}
《机器学习基石》第三讲**Types of Learning**的笔记。主要介绍了机器学习的集中分类标准和具体分类。
{% endnote %}
<!--more-->

# 不同的输出空间

## 二元分类: binary classification
很基本的分类问题，输出只有两种. 通俗的理解就是简单的是非题，要么是，要么不是。

## 多元分类: multiclass classification
很简单的例子就是对硬币的分类，课程中使用了美元的分类，1c, 5c, 10c, 25c 这样。

输出可以不只两种，上述硬币的分类输出为四，输出可以是K种。

二元分类是特殊的多元分类，即K=2的情况。

**应用场景**:主要是视觉或听觉的辨识
- 数字识别
- 图片内容分类
- 邮件的分类

## 回归问题: Regression

$ y=R $ or $y = [lower, upper] \setminus R (bounded regression)$

**特点是**输出是一个实数

**应用**
- 股票价格
- 温度预测

## 结构化学习: Structed Learning
理解起来就是多元分类的扩展，有很多很多的类别，但是类别和类别之间有着某种潜在结构，我们要输出的就是这种结构。

比如一个句子，可以是主谓宾、主谓等等，但是不可能是谓语谓语谓语这样。我们如果对一个句子进行语法判断，输出空间就是这些结构，而不是一个个的类别。

**应用**:输出空间有着某种结构
- 蛋白质的结构
- 自然语言处理中语言的parse tree

总的可以看下图：
{% asset_img tol1.png Learning with Different Output Space %}


# 不同程度标记的样本
## 监督学习: Supervised Learning
给了一堆样本，然后还对每个样本进行了标记是什么，即每个$$ x_n $$对应一个$$ y_n $$

## 非监督学习: Unsupervised Learning
给了一对样本，但是不给样本的正确标记(without yn), 让机器自己去把样本分成几类。

**聚类**就相当于是**非监督的多分类问题**
- 把一些文章按照不同的topic分类
- 按照消费者的资料把消费者分类,针对不同的人群进行促销。

**密度预测**相当于是**Unsupervised bounded regression**
- 交通车流量分析,按照位置分

**异常监测**相当于是**Unsupervised binary classification**
- 网络流量分析

{% note info %}
聚类通常比较有用，但是评定聚类的好坏通常比较困难。
{% endnote %}

## 半监督式学习: Semi-supervised Learning
比如硬币识别，但是只给了一部分$$ y_n $$，和其他没有标记的样本混在一起

**应用**
- 人脸识别，只有少量标记的面部照片
- 药物效果预测，只有少量的药物有标签

## 强化学习
一种非常不同的，但是很自然的学习方式。不直接告诉它你要做什么，但是可以通过一定的行为反应，惩罚错误的结果，奖励正确的结果。

比如训练狗狗，做得对就奖励，做的错就惩罚，没办法直接说给他听。

**应用**
- 广告系统输入的是顾客资料，顾客点击或者不点击，推荐

## 总结
{% asset_img tol2.png "Learning with Different Data Label yn" %}

# 不同的训练方式
## batch Supervised multiclass classification
成批的将数据喂给机器学习的算法，算法从所有已知的data中学习，得到假设$$ g $$
**应用**
- 数据是email，得到邮件分类器
- 数据是cancer资料，得到cancer分类器

{% note info %}
根据数据是否一次送入模型中训练分为batch learning和online learning.

batch learning像是填鸭式，online learning像是教书，一条一条教.
{% endnote %}

## online learning 
指每次有新样本的时候就用来训练更新 hypothesis，每一轮$$ g $$会更好, 常见的比如说有垃圾邮件分类系统.

增强学习和PLA常常比较接近online learning

## active learning
希望是机器能够主动的问选择的$$ x_n $$对应的$$ y_n $$，可以通过这种方式用很少的labels来提高$$ g $$

# 不同的输入空间
根据输入的样本的特征来分也可以分为下面三类（虽然这种分类方法并不常见）：concrete features，raw features 和 abstract features。

## concrete features
指输入的样本已经标注好了各种特征，如信用卡例子中顾客的各种资料

## raw features
一般指图像或音频中的图像或声波，这些信息是原始的信号，需要进行一些转换才能使用。

比如手写数字识别，16x16的像素格，可以将16x16=256个像素变成一个256维的向量输入。

## abstract features
课程中用了KDDCup的例子，给出每个用户和他们喜欢听的音乐，要预测这个用户对一个新歌曲的评分是多少。
输入的是，用户id，歌曲的id，输出是评分数字

但是输入的特征并不是很直接，需要一方面人来提示，还有是机器自己从每个人喜欢听的歌里面得到特征，从每首歌的曲风等特征里面总结出特征，然后再用这些特征训练，得到结果。

这种按照输入样本的 features 进行分类的方法在实际中并不常用，因为输入的样本往往是各种 features交杂在一起的，不同问题需要与其相应的 features 才能得到好的效果，features 对结果的影响比较大。因此机器学习中也产生了 feature engineering 一说。

## 总结
{% asset_img tol3.png "Learning with Different input Space" %}




