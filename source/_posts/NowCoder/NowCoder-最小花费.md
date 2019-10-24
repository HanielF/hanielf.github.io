---
title: NowCoder-最小花费
tags:
  - NowCoder
  - Algorithm
  - DP
  - Hard
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-21 23:27:27
urlname: minimum-cost
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/e6df3e3005e34e2598b9b565cfe797c9?tpId=40&tqId=21354&tPage=2&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
补之前的

在某条线路上有N个火车站，有三种距离的路程，L1，L2，L3,对应的价格为C1,C2,C3.其对应关系如下: 距离s票价 0\<S<=L1: C1; L1\<S\<=L2, C2;  L2\<S<=L3: C3 输入保证0\<L1\<L2\<L3<10^9,0\<C1\<C2\<C3\<10^9。 

每两个站之间的距离不超过L3。 当乘客要移动的两个站的距离大于L3的时候，可以选择从中间一个站下车，然后买票再上车，所以乘客整个过程中至少会买两张票。

现在给你一个 L1，L2，L3，C1，C2,C3。然后是A B的值，其分别为乘客旅程的起始站和终点站。 然后输入N，N为该线路上的总的火车站数目，然后输入N-1个整数，分别代表从该线路上的第一个站，到第2个站，第3个站，……，第N个站的距离。

根据输入，输出乘客从A到B站的最小花费。

输入格式：  
L1  L2  L3  C1  C2  C3
A  B
N
a[2]
a[3]
……
a[N]

根据输入，输出乘客从A到B站的最小花费。

### Examples:
**Input:**
> 1 2 3 1 2 3
> 1 2
> 2
> 2
 
**Output:**
> 2

{% endnote %}
<!--more-->

## Solutions
- 动态规划问题，主要过程是没每往后移动一个站点， 就利用前面的站点信息来找到此站的最小花费，在距当前站L3范围内找，因为要找直达当前站的最近站点
- 一直更新站点花费到B站点，结束
- 状态转移方程：cost[i] = min(cost[i], getCost(dis,j,i)+cost[j])


## C++ Codes

```C++
/*
 * 计算从A站点到B站点的最小花费
 * 中间有好多站，可能转车
 * 动态规划求解
 * 第i站最小花费 = min(到前面所有距i站L3以内站点j的最小花费 + j到i站的花费)
 * cost[i] = min(cost[i], tmp+cost[j]) and tmp=getCost(dis,j,i)
 * 相当于就是找个直达i的站点，也是找转车站点，在这个站点转车，直达i站点，在距离i站L3以内选
 */
#include<iostream>
#include<climits>
#include<vector>
using namespace std;

int L1, L2, L3;
int C1, C2, C3;
int A, B;
int N;

//获取不超过L3距离的花费
int getCost(vector<int> &dis, int i, int j){
  int len = dis[j]-dis[i];
  if(len>L2 && len<=L3) return C3;
  if(len>L1 && len<=L2) return C2;
  return C1;
}

int main(){
  while(cin>>L1>>L2>>L3>>C1>>C2>>C3>>A>>B>>N){
    vector<int>dis(N+1);
    dis[0]=dis[1]=0;
    for(int i=2;i<=N;i++) cin>>dis[i];
    vector<int>cost(N+1);
    cost[A]=0;
    //向后dp
    for(int i=A+1;i<=B;i++){
      //初始化这个站点要好多好多钱
      cost[i]=INT_MAX;
      //从i站前一个站点往前找站点，当然，距离在L3以内才能直达
      for(int j=i-1;j>=1;j--){
        if(dis[i]-dis[j]>L3) break;
        int tmp = getCost(dis, j, i);
        if(tmp+cost[j]<cost[i])
          cost[i]=tmp+cost[j];
      }
    }
    cout<<cost[B]<<endl;
  }
}
```

## 总结
- 主要是用部分站点来更新当前站点，还有要用给的条件计算直达站点的cost
- 题干很长，要耐心

------
