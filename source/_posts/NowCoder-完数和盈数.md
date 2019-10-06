---
title: NowCoder-完数和盈数
tags:
  - NowCoder
  - Algorithm
  - Math
  - Easy
  - Integer
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-21 22:42:36
urlname: perfect-num-and-excess-num
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/ccc3d1e78014486fb7eed3c50e05c99d?tpId=40&tqId=21351&tPage=1&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
一个数如果恰好等于它的各因子(该数本身除外)子和，如：6=3+2+1。则称其为“完数”；若因子之和大于该数，则称其为“盈数”。 求出2到60之间所有“完数”和“盈数”。

没有输入
{% endnote %}
<!--more-->

## Solutions
- 暴力求解 


## C++ Codes

```C++
#include<iostream>
#include<cmath>
using namespace std;

int getSum(int n){
    int sq = sqrt(n);
    int res = 1;
    for(int i=2;i<=sq;i++){
        if(n%i==0){
            res = res+i+n/i;
            if(i*i==n) res-=i;
        }
    }
    return res;
}

int main(){
    int p[60], q[60];
    int cntp=0, cntq=0;
    for(int i=2;i<=60;i++){
        int tmp = getSum(i);
        if(tmp==i){
            p[cntp++]=i;
        }else if(tmp>i){
            q[cntq++]=i;
        }
    }
    cout<<"E:";
    for(int i=0;i<cntp;i++) cout<<" "<<p[i];
    cout<<"\nG:";
    for(int i=0;i<cntq;i++) cout<<" "<<q[i];
    return 0;
}
```


------
