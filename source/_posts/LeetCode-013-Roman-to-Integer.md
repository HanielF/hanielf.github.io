---
title: LeetCode-013-Roman to Integer
tags:
  - LeetCode
  - Algorithm
  - Math
  - String
  - Integer
  - Easy
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-05-01 03:57:54
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode.com/problems/roman-to-integer/)   
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.
Symbol　　　Value
I　　　　　　 1
V　　　　　　 5
X　　　　　　 10
L　　　　　　 50
C　　　　　　 100
D　　　　　　 500
M　　　　　　 1000

For example, two is written as II in Roman numeral, just two one's added together. Twelve is written as, XII, which is simply X + II. The number twenty seven is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

- I can be placed before V (5) and X (10) to make 4 and 9. 
- X can be placed before L (50) and C (100) to make 40 and 90. 
- C can be placed before D (500) and M (1000) to make 400 and 900.

Given a roman numeral, convert it to an integer. Input is guaranteed to be within the range from 1 to 3999.

### Examples:
**Input:**"III"
**Output:**3

**Input:**"IV"
**Output:**4

{% endnote %}
<!--more-->

## Solutions
- 想到两种方法，第一种是遍历字符串，判断每个字符和右边字符代表数字的大小，如果不小于右边的，就加上这个字符代表的数字，否则减去。遍历之前要建立字符对应数字的map，时间复杂度是O(n)
- 还有一种方法是遍历每种情况，然后从左向右查找字符串中的字符，并且记录出现的次数，最后加上次数乘以数字,时间复杂度也是O(n),但是比第一种会花更多时间，因为有无用的比较


## C++ Codes
第一种，遍历字符串，40ms，10.8MB
```C++
class Solution {
public:
    int romanToInt(string s) {
        int res = 0;
        map<char,int> values;
        values['M']=1000;
        values['D']=500;
        values['C']=100;
        values['L']=50;
        values['X']=10;
        values['V']=5;
        values['I']=1;
        
        for(int i=0;i<s.length()-1;i++)
          values[s[i]]>=values[s[i+1]]?
              res+=values[s[i]]:res-=values[s[i]];
        
        //加上最后一位
        res += values[s[s.length()-1]];
        return res;
    }
};
```

第二种，遍历每种情况，160ms，8.6MB
```C++
class Solution {
public:
    int romanToInt(string s) {
        int values[]={1000,900,500,400,100,90,50,40,10,9,5,4,1};
        string chr[]={"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};
        int pos = 0, cnt = 0, res = 0;

        for(int i=0;i<13;i++){
            while(s.substr(pos,chr[i].length())==chr[i]){
                cnt++;
                pos+=chr[i].length();
            }
            res += values[i]*cnt;
            cnt = 0;
        }
        return res;
    }
};
```

## Python Codes
和C++相差不多，就不写了，比较简单

## 总结
- map真好用
- 尽量不要有没有用的比较判断之类的


------
