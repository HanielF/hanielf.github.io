---
title: NowCoder-整数拆分
tags:
  - NowCoder
  - Algorithm
  - Medium
  - DP
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-17 10:28:10
urlname: integer-split
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/376537f4609a49d296901db5139639ec?tpId=40&tqId=21339&tPage=1&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
**补之前的**
一个整数总可以拆分为2的幂的和，例如： 7=1+2+4 7=1+2+2+2 7=1+1+1+4 7=1+1+1+2+2 7=1+1+1+1+1+2 7=1+1+1+1+1+1+1 总共有六种不同的拆分方式。 再比如：4可以拆分成：4 = 4，4 = 1 + 1 + 1 + 1，4 = 2 + 2，4=1+1+2。 用f(n)表示n的不同拆分的种数，例如f(7)=6. 要求编写程序，读入n(不超过1000000)，输出f(n)%1000000000。

### Examples:
**Input:**
每组输入包括一个整数：N(1<=N<=1000000)。
**Output:**
对于每组数据，输出f(n)%1000000000。
{% endnote %}
<!--more-->

## Solutions
- 动态规划题目， 主要要能找到转移方程。
- 具体看下面的代码



## C++ Codes

```C++
/*
 * 整数拆分
 * 动态规划
 * f(2m)=f(2m-1)+f(m)
 * f(2m+1)=f(2m)
 */
#include<iostream>
#include<cstring>
#include<cmath>
using namespace std;

#define ll long long
#define MAXN 1000000

  ll N;
  ll dp[MAXN+1];
  memset(dp, 0, sizeof(dp));
  dp[1]=1;
  dp[2]=2;
  for(ll i=3;i<=MAXN;i++){
    if(i%2) dp[i]=dp[i-1]%1000000000;
    else dp[i]=(dp[i-1]+dp[i/2])%1000000000;
  }
  while(cin>>N){
    cout<<dp[N]<<endl;
  }
  return 0;
}
```

------
