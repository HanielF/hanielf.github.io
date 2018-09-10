---
title: 聚类和K-Means算法
comments: true
date: 2018-09-10 18:08:12
tags: [Python,MachineLearning,Clustering,K-Means,Learning]
categories: MachineLearning
mathjax: true
---

<meta name="referrer" content="no-referrer" />
# 聚类
　　聚类(Clustering)，指的是一种学习方式（操作方式），即把物理或者抽象对象的集合分组为由彼此的对象组成的多个类的分析过程。<!--more--> 
{% note info %} 聚类属于无监督机器学习，简言之就是把特征形态相同的或者近似的划分在一个概念下，聚集为一组。{% endnote %}
## 欧氏距离
　　欧氏距离是最直观的距离度量方法，通常就是学过的两点间距离，可以用在多维。

- 二维平面上点a(x1,y1)与b(x2,y2)间的欧氏距离:  
$$ d_{12} = \sqrt{(x_1-x_2)^2+(y_1-y_2)^2} $$
- 三维空间点a(x1,y1,z1)与b(x2,y2,z2)间的欧氏距离:
$$ d{12} = \sqrt{(x_1-x_2)^2+(y_1-y_2)^2+(z_1-z_2)^2} $$
- 更高维的计算类似二维三维

## 曼哈顿距离
　　不再是两点间连线的那种，是类似九宫格的走法，只能直线和直角拐弯。又叫做“城市街区距离”。

- 二维平面两点a(x1,y1)与b(x2,y2)间的曼哈顿距离：
$$ d_12 = |

  
