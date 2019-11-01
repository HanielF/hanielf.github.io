---
title: LeetCode-028-Implement strStr()
tags:
  - LeetCode
  - Algorithm
  - KMP
  - Medium
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-10-30 12:30:27
urlname: LeetCode-028-Implement-strStr 
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/implement-strstr/)
Implement strStr().

Return the index of the first occurrence of needle in haystack, or -1 if needle is not part of haystack.

### Examples:
**Input:**
> haystack = "hello", needle = "ll"

**Output:**
> 2

**Input:**
> haystack = "aaaaa", needle = "bba"

**Output:**
> -1

{% endnote %}
<!--more-->

## Solutions
- 很明显是一个字符串匹配的问题，自然想到用KMP算法
- 难的是会不会KMP...
- KMP本身难点在getNext()函数，具体见代码和注释


## C++ Codes

```C++
class Solution {
public:
    //如果使用next作为变量名会报错
    int myNext[1000000];
    
    int strStr(string haystack, string needle) {
        int pos = KMP(haystack,needle);
        return pos;
    }
    
    void getNext(string p){
      // myNext[j] = k，表示当T[i] != P[j]时，j指针的下一个位置，myNext[0]=-1表示主串后移
      memset(myNext,-1,sizeof(myNext));
      int j=0;
      int k=-1;

      while(j<p.length()-1){
        //如果两个字符相等或者k==-1
        if(k==-1 || p[j]==p[k]){
          //当后两个字符相等时，跳转到k就没必要了，因为p[j]==p[k]，所以应该跳转到myNext[k]
          //因此要设置为myNext[k]
          if(p[++j] == p[++k]){
            myNext[j]=myNext[k];
          //不等就正常跳转
          } else {
            myNext[j]=k;
          }
        //不等就k=myNext[k]，往前找匹配的
        } else {
          k=myNext[k];
        }
      }
    }
    
    //ts主串，ps模式串
    int KMP(string ts, string ps){
      int i=0;  //主串位置
      int j=0;  //模式串位置
        
      int tsLen = ts.length(), psLen = ps.length();
      if(tsLen==0 && psLen==0) return 0;
      if(tsLen==0) return -1;
      if(psLen==0) return 0;
  
      getNext(ps);
        
      // 这里如果直接写 i<ts.length() && j<ps.length()会出错，只进行一次循环就跳出了，很奇怪
      while(i<tsLen && j<psLen){
        // cout<<"Before: i: "<<i<<" j: "<<j<<endl;
        //当j为-1时，要移动的是i，当然j也要自增归0
        if(j==-1 || ts[i]==ps[j]){
          i++;
          j++;
        } else{
           // i不需要回溯了，i=i-j+1，j回到指定位置
          j=myNext[j];
        }
        // cout<<"Before: i: "<<i<<" j: "<<j<<endl;
      }
        
      //全部匹配，返回主串匹配的下标，否则返回-1
      if(j==ps.length()){
        return i-j;
      } else {
        return -1;
      }
    }
};
```

## 总结
- 理解KMP很重要，每一步要理解为什么 


------
