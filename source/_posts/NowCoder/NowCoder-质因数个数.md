---
title: NowCoder-质因数个数
tags:
  - NowCoder
  - Algorithm
  - Easy
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-17 10:23:31
urlname: number-prime-factor
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/20426b85f7fc4ba8b0844cc04807fbd9?tpId=40&tqId=21338&tPage=1&rp=1&ru=/ta/kaoyan&qru=/ta/kaoyan/question-ranking)   
补之前的

求正整数N(N>1)的质因数的个数。 相同的质因数需要重复计算。如120=2*2*2*3*5，共有5个质因数。

### Examples:
**Input:**
可能有多组测试数据，每组测试数据的输入是一个正整数N，(1\<N\<10^9)。
**Output:**
对于每组数据，输出N的质因数的个数。
{% endnote %}
<!--more-->

## Solutions
- 先计算素数，然后找因数


## C++ Codes

```C++
/*
 * 素数筛
 * 计算处sqrt(n)以内的素数
 * 最后的部分是一个比sqrt(n)大的素数，个数加1即可
 */
#include<iostream>
#include<cstring>
#include<cmath>
#include<vector>
using namespace std;

#define MAXN 100000
int prime[MAXN];
bool valid[MAXN];

//素数筛，返回1-num以内素数个数并存在pr数组内
int getPrime(int *pr, int num){
  int res = 0;
  memset(valid, true, sizeof(valid));
  for(int i=2;i<=num;i++){
    if(valid[i]) pr[res++]=i;

    for(int j=0;j<res;j++){
      //更新所有素数和当前数乘积为false
      if(pr[j]*i>num)break;
      valid[pr[j]*i]=false;

      //如果是已知素数的倍数，就break；
      if(i%pr[j]==0) break;
    }
  }
  return res;
}

int main(){
  int sumCnt = getPrime(prime, MAXN);
  int n;
  while(cin>>n){
    int cnt = 0;
    int sq = sqrt(n);
    for(int i=0;i<sumCnt;i++){
      if(prime[i]>sq) break;
      while(n%prime[i]==0){
        cnt++;
        n/=prime[i];
      }
    }
    if(n>1) cnt++;
    cout<<cnt<<endl;
  }

}
```

## 总结
- 素数筛还是要会的，几种筛法 

------
