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
## 聚类定义
　　聚类(Clustering)，指的是一种学习方式（操作方式），即把物理或者抽象对象的集合分组为由彼此的对象组成的多个类的分析过程。<!--more--> 

{% note info %} 聚类属于无监督机器学习，简言之就是把特征形态相同的或者近似的划分在一个概念下，聚集为一组。

聚类在实际的应用中中亦是非常广泛的，如：市场细分（Market segmentation）、社交圈分析（social network analysis）、集群计算（organize computing clusters）、天体数据分析（astronomical data analysis）等
{% endnote %}

## 聚类算法分类
　　主要的聚类主要的聚类算法可以划分为如下几类：划分方法、层次方法、基于密度的方法、基于网格的方法以及基于模型的方法。

　　每一类中都存在着得到广泛应用的算法，例如：划分方法中的k-means聚类算法、层次方法中的凝聚型层次聚类算法、基于模型方法中的神经网络聚类算法等

　　但是上述的都是硬聚类，即每一个数据只能被归为一类，还有一种是模糊聚类。
   　　模糊聚类通过隶属函数来确定每个数据隶属于各个簇的程度，而不是将一个数据对象硬性地归类到某一簇中。

# 常用距离算法
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
$$ d_12 = |x_1-x_2|+|y_1-y_2| $$
- 三维和其他维类似

{% note info %}
　　除了这两种还有余弦距离和切比雪夫距离等，这里不展开说。采用不同的距离度量方法对结果有很大的影响。
{% endnote %}

# k-means算法
  思想大致是:
- 1.　先随机选k个质心
- 2.　对每个点计算其到各个质心的距离
- 3.　选距离最近的，把这个点归为这个质心的一类，形成k个簇
- 4.　然后对于每个簇，计算其中每个点到质心的平均距离
- 5.　然后把这个作为这个簇的新的质心,进行第二步
- 6.　直到簇不怎么发生变化或者达到了预设的最大迭代次数，停止
  
　主要函数如下:

```C++
//k-means聚类
vector<Cluster> k_means(vector<vector<int> >trans,int k,int counts){
  vector<Cluster> clusters(k);

  const int row = trans.size();
  const int col = trans[0].size();

  //随机初始化聚类中心
  srand((int)time(0));
  for(int i=0;i<k;i++){
    int center = rand()%trans.size();
    clusters[i].center=trans[center]; 
  }

  //迭代counts次
  for(int cnt = 0;cnt<counts;cnt++){

    //清空样本空间
    for(int i=0;i<k;i++)
      clusters[i].samples.clear();

    //计算样本属于的簇
    for(int i=0;i<row;i++){
      int tmp_center = 0;
      int minal = cal_distance(trans[i],clusters[tmp_center].center);

      for(int j=1;j<k;j++){
        int distance = cal_distance(trans[i],clusters[j].center);
        if(distance<minal){
          tmp_center = j;
          minal = distance;
        }
      }

      clusters[tmp_center].samples.push_back(i);
    }

    //重新计算簇中心
    for(int i=0;i<k;i++){
      int sum = 0;

      for(int m=0;m<trans[0].size();m++){
        for(int j=0;j<clusters[i].samples.size();j++){
          //cout<<"sum+=: "<<trans[clusters[i].samples[j]][m];
          sum+=trans[clusters[i].samples[j]][m];
        }

        clusters[i].center[m]=sum/clusters[i].samples.size();
        sum=0;
      }
    }
  }

  return clusters;
}
```

　用了C\+\+作为实现的代码，python的代码可以参考网上的，有很多。计算距离用的是欧式距离。数据可以自己构造尝试。

----------------------------------------------
