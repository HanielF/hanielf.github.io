---
title: NowCoder-约数个数
tags:
  - NowCoder
  - Algorithm
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-12 19:12:10
urlname: nowcoder-gcd-numbers
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/04c8a5ea209d41798d23b59f053fa4d6?tpId=40&tqId=21334&tPage=1&rp=1&ru=/ta/kaoyan&qru=/ta/kaoyan/question-ranking)   
输入n个整数,依次输出每个数的约数的个数

### Examples:
**Input:**   
输入的第一行为N，即数组的个数(N<=1000)
接下来的1行包括N个整数，其中每个数的范围为(1<=Num<=1000000000)
当N=0时输入结束。
**Output:**  
可能有多组输入数据，对于每组输入数据，
输出N行，其中每一行对应上面的一个数的约数的个数。
{% endnote %}
<!--more-->

## Solutions
- 直接求每个数的约数就完了

## C++ Codes

```C++
#include<iostream>
#include<cmath>
using namespace std;

int approNums(int num){
  int res = 0;
  int sq = sqrt(num);
  for(int i=1;i<=sq;i++){
    if(num/i*i==num){
      res+=2;
      if(i*i==num) res-=1;
    }
  }
  return res;
}

int main(){
  int N;
  int nums[1000];
  int res[1000];
  while(cin>>N && N!=0){
    for(int i=0;i<N;i++) cin>>nums[i];
    for(int i=0;i<N;i++){
      res[i]=approNums(nums[i]);
    }
    for(int i=0;i<N;i++)
      cout<<res[i]<<endl;
  }
  return 0;
```


------
