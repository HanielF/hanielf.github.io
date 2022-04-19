---
title: KMP算法
tags:
  - Algorithm
  - KMP
  - String
  - Match
categories:
  - Notes
comments: true
mathjax: false
date: 2019-10-30 12:35:59
urlname: kmp-algorithm
---

<meta name="referrer" content="no-referrer" />

{% note info %}
KMP算法又复习了一遍，写个总结，贴下自己的代码。
{% endnote %}
<!--more-->

## KMP算法介绍和原理
简单说就是字符串匹配，在主串中匹配模式串，返回匹配到的下标。

具体的可以看[详解KMP算法](https://www.cnblogs.com/yjiyjige/p/3263858.html)这篇博客，讲的挺好的，也有图解。
这里只放几张截图，不做详细介绍了。

![基本思想](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/kmp-1.png)

![j指针移动的位置](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/kmp-2.png)

!["next[j]=k，表示当T[i] != P[j]时，j指针的下一个位置"](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/kmp-3.png)

!["j=0和j=1时的情况"](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/kmp-4.png)

!["P[k]=P[j]时，next[j+1]==next[j]+1"](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/kmp-5.png)

!["P[k] != P[j]时， k = next[k]"](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/kmp-6.png)

![kmp-7](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/kmp-7.png)

## KMP算法实现

```c
i#include <cstring>
#include <iostream>
#include <string>
#include <vector>
using namespace std;

//如果使用next作为变量名会报错
const long long int SIZE = 1e5;
int myNext[SIZE];

void getNext(string p) {
  // myNext[j] = k，表示当T[i] !=
  // P[j]时，j指针的下一个位置，myNext[0]=-1表示主串后移 
  memset(myNext, -1, sizeof(myNext));
  int j = 0;
  int k = -1;
  while (j < p.length() - 1) {
    //如果两个字符相等或者k==-1
    if (k == -1 || p[j] == p[k]) {
      //当后两个字符相等时，跳转到k就没必要了，因为p[j]==p[k]，所以应该跳转到myNext[k]
      //因此要设置为myNext[k]
      if (p[++j] == p[++k]) {
        myNext[j] = myNext[k];
        //不等就正常跳转
      } else {
        myNext[j] = k;
      }
      //不等就k=myNext[k]，往前找匹配的
    } else {
      k = myNext[k];
    }
  }
}

// ts主串，ps模式串
int KMP(string ts, string ps) {
  int i = 0; //主串位置
  int j = 0; //模式串位置

  int tsLen = ts.length(), psLen = ps.length();
  if (tsLen == 0 && psLen == 0)
    return 0;
  if (tsLen == 0)
    return -1;
  if (psLen == 0)
    return 0;
  getNext(ps);

  // 这里如果直接写 i<ts.length() &&
  // j<ps.length()会出错，只进行一次循环就跳出了，很奇怪
  while (i < tsLen && j < psLen) {
    //当j为-1时，要移动的是i，当然j也要自增归0
    if (j == -1 || ts[i] == ps[j]) {
      i++;
      j++;
    } else {
      // i不需要回溯了，i=i-j+1，j回到指定位置
      j = myNext[j];
    }
  }

  //全部匹配，返回主串匹配的下标，否则返回-1
  if (j == ps.length()) {
    return i - j;
  } else {
    return -1;
  }
}

int main() {
  string t = "hello";
  string p = "ll";
  // 输出2
  cout << KMP(t, p) << endl;
  return 0;
}
```

-------
