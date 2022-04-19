---
title: 机器学习基石--Feasibility of Learning
comments: true
mathjax: false
date: 2019-04-08 17:38:49
tags: [机器学习基石,MachineLearning]
categories: [MachineLearning,机器学习基石,]
urlname: machinelearning-feasibility-of-learning
---

<meta name="referrer" content="no-referrer" />

{% note info %}
《机器学习基石》第四讲**Feasibility of Learning**的课程笔记。主要介绍了机器学习的可行性。
{% endnote %}
<!--more-->

机器学习很多时候，如果不加限制，常常会因为标准的不同，而有很多的不一样的结果。

并且，在训练集内得到的结果拟合的很好，但是在测试集甚至所有的可能来看，可能就是错误的，可能就是不确定的

# Learning is impossiable

## No Free Lunch
> 即: 天下没有免费的午餐
> 用于比较两种优化算法之间的关系，即如何确定一种算法比另外一种算法好

NFL定理的前提是，所有问题出现的机会相等、或所有问题都是同等重要。

而实际情形往往并不是这样。一般我们只需要关注自己正要解决的问题即可。而对于我们的解决方案在另一个问题上的表现是否同等出色，我们并不关心。

> 因此，脱离具体问题而空谈“什么算法最好”之类的讨论毫无意义. 
>  因为若考虑所有潜在的问题，那么所有的模型、算法都一样好——这也是我们通过NFL定理得出的。
> 要比较模型的相对优劣，则必须建立在与之对应的学习问题之上。

# Probability to the Rescue

## Hoeffding's Inequality
> Hoeffding 不等式

大概意思就是不知道很大的样本中的概率，但是我们可以通过很多次的抽样，得到的概率来推测真正的概率。

想到以前数学家证明硬币一面朝上的概率，通过很多很多很多次的抛硬币来统计每面朝上的概率，最后证明就是1/2，并且抛的次数越多，概率越接近。

!["Hoeffding Inequality 1/2"](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/fol1.png)

!["Hoeffding Inequality 2/2"](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/fol2.png)

# Connection to Learning

$$ E_in(h) $$代表我们抽的样本中的不一致概率

$$ E_out(h) $$代表总的样本中的不一致的概率

可以推断说N足够大时，$$ E_in(h) \sim E_out(h) $$

如果$$ E_in(h) \sim E_out(h) $$，并且$$ E_in(h) $$很小，我们就可以推断，$$ E_out(h) $$很小，并且，$$ h \sim f with respect to P $$

!["Added Componentss"](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/fol3.png)

!["The Formal Guarantee"](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/fol4.png)

{% note danger %}
**real learning** is: $$ A $$ shall **make choices $$ \in H $$** (like PLA)
rather than *being forced to pick one h*
{% endnote %}

{% note info %}
可以使用历史数据(data)来验证一个假设的表现到底好不好, 可以理解为验证集
{% endnote %}

# Connection to Real Learning
提出了问题：如果我们在一堆假设中看到了一个假设，在我们选出的样本上全对，我们要不要选择这个假设？举例子就是每个人都抛硬币，抛五次，可能有一个人会五次全都朝上，我们要不要说这个硬币会有点特殊？

Hoeffding说的是取样的和真实的大部分情况下是符合的，只有小部分是不好的

事实上当你有选择的时候，比如这里抛五次硬币实验150次的时候,150次试验里出现一次五个硬币同时朝上的概率就是$$ 1 - {(31 \over 32)^150} > 99% $$

因此不好的样本，在有选择的时候，出现的概率会恶化。
**不好的Data** == $$ A $$不可以自由做选择，可能会踩雷 == 存在$$ h $$使$$ E_out(h) $$和$$ E_in(h) $$
差得很大

Hoeffding说的是在一个Data里面，抽一堆和大部分符合

这里说的其实是一堆Data里，出现不好的Data的概率是：

!["Bound of BAD Data"](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/fol5.png)

所以如果假设数量有限，即M有限，并且每个Data样本N足够大，那么不管$$ A $$怎么选，$$ E_out(g) \sim E_in(g)$$，即可以放心选，这样就说明了有限数量的h情况下，机器学习是可行的



