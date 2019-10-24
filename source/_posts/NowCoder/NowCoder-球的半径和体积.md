---
title: NowCoder-球的半径和体积
tags:
  - NowCoder
  - Algorithm
  - Math
  - Easy
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-17 10:39:12
urlname: Radius-and-volume-of-the-ball
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/4b733a850c364c32b368555c8c2ec96b?tpId=40&tqId=21341&tPage=1&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
补之前的

输入球的中心点和球上某一点的坐标，计算球的半径和体积

### Examples:
**Input:**
球的中心点和球上某一点的坐标，以如下形式输入：x0 y0 z0 x1 y1 z1
**Output:**
输入可能有多组，对于每组输入，输出球的半径和体积，并且结果保留三位小数
为避免精度问题，PI值请使用arccos(-1)。


{% endnote %}
<!--more-->

## Solutions
- 简单题，没有算法，直接算


## C++ Codes

```C++
#include<iostream>
#include<cmath>
using namespace std;

int main(){
    double x0, y0, z0, x1, y1, z1;
    double pi = acos(-1); 
    while(cin>>x0>>y0>>z0>>x1>>y1>>z1){
        double x = abs(x1-x0);
        double y = abs(y1-y0);
        double z = abs(z1-z0);
        double r = sqrt(x*x + y*y + z*z);
        double v =4.0/3.0*pi*r*r*r;
        printf("%.3f %.3f\n", r, v);
    }
    return 0;
}
```


------
