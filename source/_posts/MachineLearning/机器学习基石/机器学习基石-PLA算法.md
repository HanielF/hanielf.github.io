---
title: 机器学习基石--PLA算法
comments: true
mathjax: true
date: 2019-04-07 12:36:48
tags: [机器学习基石,MachineLearning,PLA]
categories: [MachineLearning,机器学习基石,]
urlname: machinelearning-pla
---

<meta name="referrer" content="no-referrer" />

{% note info %}
《机器学习基石》第二讲 **Learning to Answer Yes/NO** 课程笔记。这一讲主要介绍了机器学习基本概念和感知机，以及其训练算法PLA。
{% endnote %}
<!--more-->

## 基本概念
- $$\mathcal f$$: 未知的目标函数
- $$\mathcal D$$: 训练样本，数据集
- $$\mathcal A$$: 学习算法
- $$\mathcal H$$: 假设集
- $$\mathcal g$$: 最终的假设，是\mathcalf的一个近似函数

课件上很清楚的描绘了机器学习的一个过程
![基本过程](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pla1.png)

## 感知机
感知机是神经网络的基础，与线性回归（Linear Regression），逻辑回归（Logistics Regression）等模型也非常类似，是一种非常典型的线性模型。

原始的感知机算法用于解决二分类问题，其思想如下：假设样本有 d 个特征，但是每个特征的重要性不一样，因此各个特征的权重也不一样，对其进行加权后得到的总和假如大于某个阈值则认为归为其中一类，反之归为另一类。如在信用卡的例子中，通过感知机有如下的结果
![推导过程](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pla2.png)

然后可以将threshold化为常数项作为$$w_0$$,简化为下图：
![简化过程](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pla3.png)

上面的$$w$$和$$x$$均为一个列向量，即$$w$$转置后成为行向量

## PLA
感知机要通过学习才能对样本进行正确的分类，这个学习的过程就是PLA(Perceptron Learning Algorithm).

**过程如下**：
1. 随机初始化参数$$w$$
2. 利用参数$$w$$预测每个样本点的值并与其实际的值比较，对于分类错误的样本点(xn,yn),利用公式$$w=w+ynxn$$更新参数$$w$$的值
3. 重复上面的过程直到所有的样本点都能够被参数$$w$$正确预测。

对于某个被预测错误的样本点，参数$$ w $$更新过程如下：
![w的更新](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pla4.png)

注意上面的算法的前提是所有的样本点都必须线性可分，假如样本点线性不可分，那么PLA按照上面的规则会陷入死循环中。如下是线性可分与线性不可分的例子)

![线性不可分的例子](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pla5.png)

## 收敛性证明
上面提到只有当所有的样本均为线性可分时，PLA才能将所有的样本点正确分类后再停下了，但是这仅仅是定性的说明而已，并没有严格的数学正面来支撑其收敛性，下面要讲的便是通过数学证明来说明 PLA 算法的收敛性。

课程中用两次递进的证明来说明收敛性

![简单证明](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pla6.png)

上面讲的是随着参数$$ w $$的更新,$$ w^T_fw_t+1 $$的值越来越大，也就是两者越来越相似
衡量两个向量相似性的一种方法就是考虑他们的内积，值越大，代表两者约接近，但是这里还没对向量归一化，所以证明并不严格，但是已经说明了两者具有这个趋势，下面是更严格的过程
![严格证明](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pla7.png)

上面似乎只是说明了经过 T 次的纠错，wt 的值会限制在一个范围内，但是并没有给出最终结论
$$ {w_f \over ||w_f||}{w_T \over ||w_T||} \ge \sqrt{T} * constant $$
的证明过程，因此在这里进行推导过程的描述
(注：这里的$$ w_f $$是不变的，因此$$ w_f $$与$$ w^T_f $$是一样的)

假设经过了 T 次纠错，那由第一张PPT可知
$$ w^T_fw_T \ge w_f^Tw_{T-1} + \min_{n}y_nw_f^Tx_n $$
而由第二章张ppt可知
$$ ||w_T||^2 \le ||w_{T-1}||^2 + \max_n||x_n||^2 \le T\max_n||x_n||^2 $$
即：$$ ||w_T|| \le \sqrt{T}\max_n||x_n|| $$

综合上面两个式子有
$$ {w_f^T \over ||w_f^T||}{w_T \over ||w_T||} \ge {T\min_ny_n^Tw^T_fx_n \over ||w_f^T||\sqrt{T}\max_n||x_n||} = \sqrt{T}{\min_ny_n{w_f^T \over ||w_f^T||}x_n \over \max_n||x_n||} = \sqrt{T} * constant $$

因此上面的命题得证。至此，已经可知道犯错误的次数 T 是受到某个上限的约束的。下面会给出这个具体的上限是多少。

又因为
$$ 1 \ge {w_f^T \over ||w_f^T||}{w_T\over||w_T||} \ge \sqrt{T} * constant $$
$$ {1\over constant^2 } \ge T$$
即犯错的次数上限是${1 \over constant^2}$,假设令
$$ \max_n||x||^2 = R^2, \rho = \min_ny_n{w_f^T \over ||w_f^T||}x_n $$
则有
$$ T \le {R^2 \over \rho^2} $$
这也说明了PLA会在有限步内收敛，这个证明也是后面的练习答案

## 优缺点和优化
PLA 的优点和缺点都非常明显，其中优点是简单，易于实现

缺点是假设了数据是线性可分的，然而事先并无法知道数据是否线性可分的。正如上面提到的一样，假如将PLA 用在线性不可分的数据中时，会导致PLA永远都无法对样本进行正确分类从而陷入到死循环中。

为了避免上面的情况，将 PLA 的条件放宽一点，不再要求所有的样本都能正确地分开，而是要求犯错的的样本尽可能的少，即将问题变为了
$$ arg\min_w\sum_{n=0}^N1\{y_n \neq sign(w^Tx_n)\} $$

这个最优化问题是个 NP-hard 问题，无法求得其最优解，因此只能求尽可能接近其最优解的近似解。讲义中提出的一种求解其近似解的算法`Pocket Algorithm`。  

其思想就是每次保留当前最好的$$ w $$, 当遇到错误的样本点对$$ w $$进行修正后，比较修正后的$$ w $$与原来最好的$$ w $$在整个样本点上的总体效果再决定保留哪一个，重复迭代足够多的次数后返回当前得到的最好的$$ w $$。

