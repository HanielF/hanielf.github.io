---
title: UCAS算法设计与分析
comments: true
mathjax: true
date: 2020-12-24 11:42:24
tags:
  - UCAS
  - Algorithm
  - Notes
  - Math
categories: Notes
urlname: ucas-algorithm
---

<meta name="referrer" content="no-referrer" />

{% note info %}
简单知识点大纲总结。

新增：期末考试题总结。
{% endnote %}
<!--more-->

## 期末考试题

这考试，复习和不复习的结果是一样的，绝了，说好是作业题的变形，和作业几乎没关系。。保佑及格。。

- 第一题分治卡了快40分钟，[LeetCode第四题](https://leetcode-cn.com/problems/median-of-two-sorted-arrays/)
- 第二题动态规划，给个数组，算相邻两个数的差，要使得相邻的两个差值异号，并且要求是连续的，如果是0也不算。简单的dp，难得有道正常的题。
- 第三题贪心，[codeforce 1426E](https://codeforces.ml/problemset/problem/1426/E)，剪刀石头布，两个人，固定各自的出手次数为$a_1, a_2, a_3, b_1, b_2,b_3$，对应剪刀石头布三个
  - 第一问，求第一个人的最大能赢的次数，只要贪赢就行。
  - 第二问，求第一个人的最少能赢的次数，这个比较难想明白。。。一直从正面想，也卡了好久。
  - 排除输的和平局，反面思路，![j2GFK1](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/j2GFK1.png)
  - [正面思路](https://www.cnblogs.com/airisa/p/13766028.html)
- 第四问LP，设置停车场位置，固定n栋楼位置为$x_1,x_2,..,x_n$，在每个楼距离$r$范围内设置一个停车场，每个楼都要设置一个，然后要使得最大的任意两个停车场之间的距离最小化。包含去绝对值，还有去max。max不知道怎么去，写了个min max。。。
  - [优化 | 线性规划和整数规划的若干建模技巧](https://zhuanlan.zhihu.com/p/69397833)
  - ![1wN6CI](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1wN6CI.png)
  - ![BacZDj](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/BacZDj.png)
- 第五题是树状DP，题目都理解反了，把neighbor理解成了兄弟节点，没想到是节点编号的neighbor。。。也是两小问，第二问好像是贪心。
- 第六题是括号匹配，只有第三小问比较难，但是没啥时间写了。。听说是卡特兰数。

## Lec1-Introduction-and-some-representative-problems

1. TSP问题
   1. 没有环，分治
   ![uQQylV](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/uQQylV.png)
   ![Mn9REW](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Mn9REW.png)
   2. 改进策略，2-opt，只允许两条边不同
   ![o5Vm8l](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/o5Vm8l.png)
   ![dYMtdy](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/dYMtdy.png)
   3. 聪明的枚举策略，就是先按照每步的顺序，构造多步决策树，然后剪枝
   ![2WhZ9j](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/2WhZ9j.png)
   ![g4uP48](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/g4uP48.png)
   ![XlXxK6](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/XlXxK6.png)
   4. 回溯策略，考虑第一步之后，枚举所有可能的情况，不用先把每一步排好序，上面的智能枚举是枚举每一步走不走，是二叉树，这里不是。这里解的形式是$[x_1, x_2,...,x_n]$的形式
   ![hGZrt4](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hGZrt4.png)
   ![kYdsmb](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/kYdsmb.png)
   ![rfE54X](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/rfE54X.png)

## Lec5-Divide-And-Conquer

1. merge sort
   1. 分治的划分，如果是数组就是看是按照值分还是按照下标分，要选择好的划分点，不能线性规模下降，要指数规模下降
   2. ![gYEk0N](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/gYEk0N.png)
   3. ![erKmMR](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/erKmMR.png)
   4. ![EKQn9K](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/EKQn9K.png)
2. 逆序数
   1. 划分成两部分，分别计算逆序数
   2. ![ZnUNL4](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ZnUNL4.png)
   3. 合并的时候有两种策略
      1. ![15oiuY](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/15oiuY.png)
      2. ![zOHSb0](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/zOHSb0.png)
   4. ![m3twg4](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/m3twg4.png)
3. 快排
   1. ![bMQS0k](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/bMQS0k.png)
   2. 时间复杂度
   ![3lnboF](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3lnboF.png)
   3. 快排的优化
      1. pivot位置，在 $\frac{n}{4}-\frac{3n}{4}$足够好
        ![9EjVF8](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/9EjVF8.png)
      2. ![noPZWg](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/noPZWg.png)
      3. ![YypPpb](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/YypPpb.png)
      4. 就地算法
      5. ![gFN0OD](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/gFN0OD.png)
      6. ![rHxjYo](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/rHxjYo.png)
4. 第k大数
   1. ![ouEAYN](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ouEAYN.png)
   2. 和快排差不多，只要看返回的pivot是哪个位置，如果整好是k，就返回这个位置，否则只要再递归找其中一边的就好了
   3. 优化
   4. 对于pivot位置进行限制，选接近中心点的位置作为pivot，如何获取这个pivot，有三种方法
      1. ![HQP3Ph](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/HQP3Ph.png)
      2. 分组，然后取中点的中点来近似全局的median
      3. ![yZ78ME](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/yZ78ME.png)
      4. inplace的方法：BFPRT算法
      5. 随机选择pivot
      6. ![8iwQyx](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/8iwQyx.png)
      7. 随机采样的方法选pivot：Lazy Select算法
      8. ![jhgoxo](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/jhgoxo.png)
      9. ![MskR14](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/MskR14.png)
5. 乘法和矩阵乘法
6. 找最邻近点对
   1. 划分成两个大小相等的子集
   2. 合并过程是在每个子集找最近点对
      1. 左边的，右边的，还有横跨中间的
      2. ![akaqzc](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/akaqzc.png)
      3. 如何加快combine？
      4. 条带法去冗余
      5. ![8aFF7E](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/8aFF7E.png)
      6. 每个点不需要探索所有的地方
      7. ![jhu9Lf](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/jhu9Lf.png)
      8. ![4AaTqF](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/4AaTqF.png)
7. FFT, DFT一系列，不考

## Lec6-Dynamic-Programming

![cifljM](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/cifljM.png)

![HTalBV](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/HTalBV.png)

1. 导弹分配问题
   1. ![K8ag9u](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/K8ag9u.png)
   2. ![e6adxN](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/e6adxN.png)
2. 矩阵链乘
   1. ![hikk3v](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hikk3v.png)
   2. ![mYefSH](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/mYefSH.png)
   3. DP方法
   4. ![N9xJdb](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/N9xJdb.png)
   5. 直接递归DP，指数级别
   6. ![F4xLWq](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/F4xLWq.png)
   7. 去冗余，记忆化OPT表，多项式级别
   8. ![R1LtUq](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/R1LtUq.png)
   9. ![WcGjoj](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/WcGjoj.png)
   10. 自底向上unroll recursion tree
   11. ![hrwxY0](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hrwxY0.png)
   12. ![0FPsED](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/0FPsED.png)
3. 背包问题
   1. 0-1背包
      1. ![3kX04r](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3kX04r.png)
      2. ![yLhTPi](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/yLhTPi.png)
      3. ![joEQWM](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/joEQWM.png)
      4. ![w5sWO2](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/w5sWO2.png)
      5. ![Sn7pQE](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Sn7pQE.png)
      6. ![7VIou9](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7VIou9.png)
      7. 空间可以用滚动数组优化到$O(n)$
   2. 完全背包
      1. ![Dl6bo1](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Dl6bo1.png)
4. 顶点覆盖问题
   1. ![FOSUJU](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/FOSUJU.png)
   2. ![cuG5st](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/cuG5st.png)
   3. ![l6vTKY](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/l6vTKY.png)
5. 字符串序列对齐问题
   1. 方法和最长公共子序列差不多，回溯的方法也是，只不过LCS的权重为1
   2. 当前字符i和j，匹配、错配、不匹配且删除i、不匹配且删除j
   3. ![XABQ4B](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/XABQ4B.png)
   4. ![jjop5h](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/jjop5h.png)
   5. ![zdpaLj](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/zdpaLj.png)
   6. ![r2EuCA](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/r2EuCA.png)
   7. 回溯
   8. ![pVAjQp](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pVAjQp.png)
6. 高级动态规划Advanced DP
   1. ![vubpu6](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/vubpu6.png)
   2. 缩减空间
   ![5kI0RY](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/5kI0RY.png)
   3. 后缀对齐
   ![mIjgJr](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/mIjgJr.png)
   4. 对于只用两列数组的DP问题，回溯方法
      1. ![ch0N7m](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ch0N7m.png)
      2. ![08cIl9](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/08cIl9.png)
   5. 进一步的优化略过
   6. ![2LU133](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/2LU133.png)
7. banded dp
8. LCS问题
   1. ![3036uS](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3036uS.png)
   2. ![DdSkOu](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/DdSkOu.png)
   3. Sparse DP
   4. ![QGNdMw](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/QGNdMw.png)
9. 单源最短路径
   1. 如果是有向无环图，相当于拓扑排序
      1. ![tKgtJM](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/tKgtJM.png)
      2. ![6jp2DH](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/6jp2DH.png)
   2. 如果有环存在，那上面的就无法处理，要把子问题划分的更小
      1. ![9cW0dq](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/9cW0dq.png)
      2. 限制步数，Bellman-Ford算法
      3. ![UhEdWi](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/UhEdWi.png)
      4. ![8tHLqW](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/8tHLqW.png)
   3. 对于负圈的问题
      1. ![LsaTcC](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/LsaTcC.png)
      2. ![wIGHkF](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wIGHkF.png)
      3. ![iqmMP9](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/iqmMP9.png)

## 线性规划LP

![YejHvp](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/YejHvp.png)

1. Diet问题
2. 最大流
3. 最小化费流
4. 多物品流
5. SAT
6. 基因重排序距离问题
7. ILP整数线性规划
8. **单纯形算法**，这个比较难
9. 对偶问题

## LP和拉格朗日对偶

其他的不管了，不考，直接记一下公式和对偶怎么用拉格朗日推出来

1. 对偶公式
   1. ![z0CwhR](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/z0CwhR.png)
   2. ![9Yx4Lr](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/9Yx4Lr.png)
   3. ![OMSSfr](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/OMSSfr.png)
2. 拉格朗日对偶
   1. ![tkRLPR](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/tkRLPR.png)
   2. ![nyQPih](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/nyQPih.png)
   3. ![DJX00X](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/DJX00X.png)

## 证明方法

1. 循环不变量
   1. 分为循环不变量和递归不变量
   2. 分治里有用到，可以用来证明包含循环或者递归的算法
   3. ![eunmgn](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/eunmgn.png)
   4. ![brshyj](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/brshyj.png)
   5. ![Fnihhm](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Fnihhm.png)
   6. 分为三步骤
      1. 初始化
      2. 维持
      3. 终止
2. 数学归纳法
3. 交换论证法
4. 反证法

## 时间复杂度

对于递归问题

![IiwZoZ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/IiwZoZ.png)

1. ![UTVKGR](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/UTVKGR.png)
2. ![MEsm6O](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/MEsm6O.png)
3. ![eE3E2I](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/eE3E2I.png)
4. ![D1O3eK](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/D1O3eK.png)
5. ![lUK62z](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/lUK62z.png)

## Take Home Message

1. ![qO1Q4S](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/qO1Q4S.png)
2. ![kowSU1](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/kowSU1.png)
3. ![WKn8V3](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/WKn8V3.png)
4. ![RAUwtC](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/RAUwtC.png)
5. ![9jf8s1](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/9jf8s1.png)
6. ![Pqwd6Q](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Pqwd6Q.png)
7. ![1rndvv](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1rndvv.png)
8. ![SRxNMi](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/SRxNMi.png)
9. ![mNljoP](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/mNljoP.png)
10. ![Jt2RTM](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Jt2RTM.png)
11. ![tk0TMv](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/tk0TMv.png)
12. ![QZ6zIV](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/QZ6zIV.png)