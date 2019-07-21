---
title: LeetCode-009-Palindrome Number
tags:
  - LeetCode
  - Palindrome
  - String
  - Easy
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-04-19 00:24:57
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode.com/problems/palindrome-number/)   
Determine whether an integer is a palindrome. An integer is a palindrome when it reads the same backward as forward.

### Examples:
**Input:**121
**Output:**true
**Input:**-121
**Output:**true
{% endnote %}
<!--more-->

## Solutions
- 首先想到的肯定是转成字符串，然后前后每个数判断是否相等，遇到不等的就return，这种方法会创建一个字符串，可能会慢一点
- 另一种方法也很容易想，就是把数字翻转过来嘛，再比较，相等的就是回文数了，但是可能会出现翻转后的数字溢出的问题，所以，可以只把后半部分的数字翻转，和前半部分的数字比较，如果是奇数位，可以把多一位的那部分除以10再比较。


## C++ Codes
转成字符串的方法，36ms，8.2MB

```
```C++
class Solution {
public:
    bool isPalindrome(int x) {
        string xstr = to_string(x);
        int n = xstr.length();
        for(int i=0;i<n/2;i++){
            if(xstr[i]!=xstr[n-i-1]) return false;
        }
        return true;
    }
};
```

## Python Codes
Python的代码和这个几乎一样，就不写了

## 总结
- 因为是简单题，所以没啥思维难度，很容易想到方法，要注意的还是细节上面小心点 

------
