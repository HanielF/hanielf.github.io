---
title: LeetCode-007-Reverse Integer
tags:
  - LeetCode
  - Algorithm
  - Easy
  - Reverse
  - BigNumber
  - String
categories:
  - LeetCode
urlname: leetcode-reverse-integer
comments: true
mathjax: false
date: 2019-04-16 22:07:29
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode.com/problems/reverse-integer/)   
Given a 32-bit signed integer, reverse digits of an integer.

### Examples:
**Input:**123
**Output:**321
**Input:**-123
**Output:**-321

{% endnote %}
<!--more-->

## Solutions
- 用字符串的方法，先将输入转换成字符串，然后判断第一个字符是不是‘-’，是就加到结果字符串上，最后逆序添加字符到结果字符串上，再将字符串转换成长整数，return之前判断是否有溢出
- 题解的办法是用取模的方法获取每一位之后再加到结果上，遍历一遍，但是这种方法在计算结果时容易溢出，要事先检查


## C++ Codes
我的字符串方法，4ms，8.3MB
```C++
class Solution {
public:
    int reverse(int x) {
        string str = to_string(x);
        string res="";
        if(str[0]=='-'){
            str = str.substr(1,str.length());
            res+='-';
        }
        for(int i =0;i<str.length();i++){
            res+=str[str.length()-1-i];
        }
        long tmp =  atol(res.c_str());
        if(tmp>pow(2,31)-1 || tmp<pow(2,31)*(-1)){
            return 0;
        }
        return (int)tmp;
    }
};
```

题解的数学方法，8ms，8.3MB
```C++
class Solution {
public:
    int reverse(int x) {
        int rev = 0;
        while (x != 0) {
            int pop = x % 10;
            x /= 10;
            if (rev > INT_MAX/10 || (rev == INT_MAX / 10 && pop > 7)) return 0;
            if (rev < INT_MIN/10 || (rev == INT_MIN / 10 && pop < -8)) return 0;
            rev = rev * 10 + pop;
        }
        return rev;
    }
};
```

## Python Codes
Python的和第一种方法一样做，差别不大

## 总结
- C++中int型最大最小可以用<limits.h>中的INT_MAX和INT_MIN表示，分别是$$ 2^31=2147483647 $$和$$ -2^31-1=-2147483648 $$
- C++中有atoi(),atol(),c_str(),to_string()这几种方法，分别是字符串转int，字符串转long，字符串转C串，整形转字符串

------
