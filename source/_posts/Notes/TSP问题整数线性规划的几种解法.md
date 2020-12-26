---
title: TSP问题整数线性规划的几种解法
tags:
  - TSP
  - LP
  - ILP
  - Notes
categories:
  - Notes
urlname: tsp-ilp
comments: true
mathjax: true
date: 2020-12-17 14:38:35
---

<meta name="referrer" content="no-referrer" />

{% note info %}

## 前言

本来是作业里的一道题，但是有好几种解法，mark一下。

不得不说，线性规划建模是真的让人头大，一个LP作业搞了一下午加一晚上...

{% endnote %}
<!--more-->

## 题目

TSP问题，就是旅行商问题，太经典了，就不多说了。简单概括就是N个城市，找到访问每一座城市仅一次并回到起始城市的最短回路。

这里只说线性规划解法。

## Dantzig-Fulkerson-Johnson formulation（DFJ）

![p7n8tX](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/p7n8tX.png)

- $c_{ij}$表示每个路径的距离
- $x_{ij}$表示是否访问这个路径
- $S$是路径数量

缺点：约束规模过大，无法求解大规模算例

## Miller-Tucker-Zemlin formulation（MTZ）

![LWTjcQ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/LWTjcQ.png)

- 前两个约束都是为了保证每城市只访问一次，每行每列的标志和都为1。
- 第三个约束条件 $u_i - u_j \leq |V| (1- x_{ij}) -1, \forall i ,\forall j, 1<i \neq j \leq |V|$ 是为了保证图里面没有孤立的子圈。假设存在子圈，那么$x_{ij} = x_{ji} = 1$，则有$u_i - u_j \leq -1 $且$u_j - u_i \leq -1$，则$u_i - u_j +u_j - u_i \leq -2$，即$0 \leq -2$，很显然矛盾。所以这样能够确保不存在互相孤立的子圈。同时。$i,\ j$不取1，是因为经过所有点的“子圈”即是我们要求解的较优圈，应剔除出限制条件。
- 第四个和第五个约束条件就都是对变量范围进行约束了

## Gavish-Graves formulation（GG）

![7DT4EU](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7DT4EU.png)

- 分析：通过增加一组变量，确定每两点间的弧的前序弧数，来消除子回路

## Gouveia-Pires L3RMTZ formulation（GP）

![efOsq6](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/efOsq6.png)

1. L3RMTZ 模型要强于 Gouveia 和 Pires 提出的另外几种模型，但该模型推导的原理较为复杂
2. 该模型理论上更好，但实际求解时间更长。

## 效率对比

![xpLK48](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xpLK48.png)

## 参考文献

- Dantzig, G., Fulkerson, R., & Johnson, S. (1954). Solution of a large-scale traveling-salesman problem. Journal of the operations research society of America, 2(4), 393-410.
- Miller, C. E., Tucker, A. W., & Zemlin, R. A. (1960). Integer programming formulation of traveling salesman problems. Journal of the ACM (JACM), 7(4), 326-329.
- Gavish, B., & Graves, S. C. (1978). The travelling salesman problem and related problems.
- Gouveia L, Pires J (1999) The asymmetric travelling salesman problem and a reformulation of the Miller–Tucker–Zemlin constraints. Eur J Oper Res 112:134–146.
- Roberti, R., & Toth, P. (2012). Models and algorithms for the asymmetric traveling salesman problem: An experimental comparison. EURO Journal on Transportation and Logistics, 1(1-2), 113-133.