---
title: NowCoder-剩下的树
tags:
  - NowCoder
  - Algorithm
  - Array
  - Map
  - Easy
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-22 11:07:15
urlname: remaining-tree
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/f5787c69f5cf41499ba4706bc93700a2?tpId=40&tqId=21356&tPage=2&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
补之前的

有一个长度为整数L(1<=L<=10000)的马路，可以想象成数轴上长度为L的一个线段，起点是坐标原点，在每个整数坐标点有一棵树，即在0,1,2，...，L共L+1个位置上有L+1棵树。     现在要移走一些树，移走的树的区间用一对数字表示，如 100 200表示移走从100到200之间（包括端点）所有的树。     可能有M(1<=M<=100)个区间，区间之间可能有重叠。现在要求移走所有区间的树之后剩下的树的个数。

输入L和M，和M组数字

### Examples:
**Input:**
> 500 3
> 100 200
> 150 300
> 470 471
 
**Output:**
> 298 

{% endnote %}
<!--more-->

## Solutions
- 简单的开一个大数组，对每个位置标记就可以了


## C++ Codes

```C++
#include<iostream>
using namespace std;

int road[10001];

int main(){
    int L, M;
    while(cin>>L>>M){
        for(int i=0;i<=L;i++){
            road[i]=1;
        }
        int from, to;
        for(int i=0;i<M;i++){
            cin>>from>>to;
            for(int j=from;j<=to;j++){
                road[j]=0;
            }
        }
        int res = 0;
        for(int i=0;i<=L;i++){
            if(road[i]) res++;
        }
        cout<<res<<endl;
    }
}
```


------
