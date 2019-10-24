---
title: NowCoder-最小邮票数
tags:
  - NowCoder
  - Algorithm
  - DP
  - Medium
  - Knapsack
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-21 15:22:41
urlname: least-number-of-stamps
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/83800ae3292b4256b7349ded5f178dd1?tpId=40&tqId=21345&tPage=1&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
补之前的

有若干张邮票，要求从中选取最少的邮票张数凑成一个给定的总值。     如，有1分，3分，3分，3分，4分五张邮票，要求凑成10分，则使用3张邮票：3分、3分、4分即可。

**输入描述：**   
有多组数据，对于每组数据，首先是要求凑成的邮票总值M，M<100。然后是一个数N，N〈20，表示有N张邮票。接下来是N个正整数，分别表示这N张邮票的面值，且以升序排列。
**输出描述：**    
对于每组数据，能够凑成总值M的最少邮票张数。若无解，输出0。

### Examples:
**Input:**
10
5
1 3 3 3 4
**Output:**
3

{% endnote %}
<!--more-->

## Solutions
- 求的是最小的邮票张数，给的条件是总金额，以及每个邮票金额，所以很明显，一个背包问题，容量是总金额，体积是每个邮票面值，然后每个邮票价值是1,求最后最大容量的情况下，价值最少
- 和正常的0-1背包还有点不一样，就是这个求的是最少，一般就是求最大。但是解决上差不多。区别在与初始化的时候，数组f每个都是N+1, 然后公式是min而不是max


## C++ Codes

```C++
/*
 * 给出几张邮票，要求选几张凑成给定面值，如果凑不出输出0，凑出输出张数
 * 可以用动态规划的0-1背包试试
 */
#include<iostream>
#include<algorithm>
using namespace std;

int C[21];      //这里相当于01背包中的所占容量
int f[101];     //dp数组
                //因为权重/价值都是1，所以不设置权重数组

int main(){
  int M,N;
  while(cin>>M>>N){
    for(int i=0;i<N;i++)
      cin>>C[i];
    for(int i=0;i<101;i++) 
      f[i]=N+1;
    f[0]=0;
    for(int i=0;i<N;i++){
      //这里设置j>=C[i]是因为，总容量肯定要大于这个物品的空间才考虑放不放啊
      for(int j=M;j>=C[i];j--){
          f[j]=min(f[j], f[j-C[i]]+1);
      }
    }
    if(f[M]<=N)
      cout<<f[M]<<endl;
    else
      cout<<0<<endl;
  }
  return 0;
}
```

## 总结
- 0-1背包中求最大和最小的差别也就是初始值和动态规划方程,最大是max，最小是min
- 注意上面的代码的写法，最外面for循环是物品编号，里层是从后往前的动态转移

------
