---
title: 机器学习基石--Types of Learning
comments: true
mathjax: true
date: 2019-04-07 20:54:44
tags: [机器学习基石,MachineLearning]
categories: [MachineLearning,机器学习基石,]
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


# 不同的数据样本
## 监督学习: Supervised Learning
给了一堆样本，然后还对每个样本进行了标记是什么，即每个$$ x_n $$对应一个$$ y_n $$

## 非监督学习: Unsupervised Learning
给了一对样本，但是不给样本的正确标记，让机器自己去把样本分成几类。

**非监督的多分类问题**就相当于是**聚类**，比如把一些文章按照不同的topic分类，按照消费者的资料把消费者分类。

{% note info %}
聚类通常比较有用，但是评定聚类的好坏可能比较困难。
{% endnote %}

# 不同的训练方式
# 不同的输入空间

