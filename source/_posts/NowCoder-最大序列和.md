---
title: NowCoder-最大序列和
tags:
  - NowCoder
  - Algorithm
  - DP
  - Medium
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-21 23:16:25
urlname: max-sum-of-sequence
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/df219d60a7af4171a981ef56bd597f7b?tpId=40&tqId=21353&tPage=2&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
补之前的

给出一个整数序列S，其中有N个数，定义其中一个非空连续子序列T中所有数的和为T的“序列和”。 对于S的所有非空连续子序列T，求最大的序列和。 变量条件：N为正整数，N≤1000000，结果序列和在范围（-2^63,2^63-1）以内。

输入第一行为一个正整数N，第二行为N个整数，表示序列中的数。

输入可能包括多组数据，对于每一组输入数据，
仅输出一个数，表示最大序列和。

### Examples:
**Input:**
> 5
> 1 5 -3 2 4
 
**Output:**
> 9

{% endnote %}
<!--more-->

## Solutions
- 经典的动态规划问题，状态转移方程是dp[i] = max(A[i], dp[i-1] + A[i])

## C++ Codes

```C++
#include<iostream>
#include<vector>
using namespace std;
#define ll long long
ll  S[1000002];
ll dp[1000002];

int main(){
    int N;
    while(cin>>N){
        for(int i=1;i<=N;i++){
            cin>>S[i];
        }
        dp[0]=0;
        for(int i=1;i<=N;i++){
            if(dp[i-1]<=0){
                dp[i]=S[i];
            }else{
                dp[i]=dp[i-1]+S[i];
            }
        }
        ll maxs=dp[1];
        for(int i=2;i<=N;i++){
            if(dp[i]>maxs) maxs = dp[i];
        }
        cout<<maxs<<endl;
    }
    return 0;
}

```

## 总结
- dp[i-1]如果小于0, 那就从A[i]开始重新算序列和 
- 延伸可以看最大子矩阵和

------
