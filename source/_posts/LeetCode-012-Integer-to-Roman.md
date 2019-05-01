---
title: LeetCode-012-Integer to Roman
tags:
  - LeetCode
  - Map
  - Integer
  - String
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-04-30 09:28:41
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode.com/problems/integer-to-roman/)   
Roman numerals are represented by seven different symbols: I, V, X, L, C, D and M.
> Symbol　　　Value
> I　　　　　　1
> V　　　　　　5
> X　　　　　　10
> L　　　　　　50
> C　　　　　　100
> D　　　　　　500
> M　　　　　　1000

For example, two is written as II in Roman numeral, just two one's added together. Twelve is written as, XII, which is simply X + II. The number twenty seven is written as XXVII, which is XX + V + II.

Roman numerals are usually written largest to smallest from left to right. However, the numeral for four is not IIII. Instead, the number four is written as IV. Because the one is before the five we subtract it making four. The same principle applies to the number nine, which is written as IX. There are six instances where subtraction is used:

- I can be placed before V (5) and X (10) to make 4 and 9. 
- X can be placed before L (50) and C (100) to make 40 and 90. 
- C can be placed before D (500) and M (1000) to make 400 and 900.

Given an integer, convert it to a roman numeral. Input is guaranteed to be within the range from 1 to 3999.

### Examples:
**Input:** 3
**Output:** "III"
**Input:** 4
**Output:** "IV"
**Input:** 58
**Output:** "LVIII"
**Input:** 1994
**Output:** "MCMXCIV"

{% endnote %}
<!--more-->

## Solutions
- 这题，题目比较长，可能容易晕，但是很好理解，就是把一个整数转成罗马数字，给出了每个字符表示的大小
- 简单的办法就是从最大值开始，整除得到这个字母的个数，求模得到剩余的数字，然后再次对较小的整除、求模一直到“I”，就是1为止，过程中把字符加到结果字符串上
- 这里由于有对应关系，所以C++中用两个数组表示对应关系，也可以用map<string,int>表示，Python直接用字典就可以


## C++ Codes

```C++
class Solution {
public:
    string intToRoman(int num) {
        int values[]={1000,900,500,400,100,90,50,40,10,9,5,4,1};
        string chr[]={"M","CM","D","CD","C","XC","L","XL","X","IX","V","IV","I"};
        
        string res;
        for(int i=0; i<13; i++){
            for(int j=0;j<num/values[i];j++)
                res += chr[i];
            num %= values[i];
        }
        return res;
    }
};
```

## Python Codes

```python
class Solution:
    def intToRoman(self, num: int) -> str:
        values = {"M":1000,"CM":900,"D":500,"CD":400,"C":100,"XC":90,"L":50,"XL":40,"X":10,"IX":9,"V":5,"IV":4,"I":1}
        res = ""
        for key,value in values.items():
            for i in range(num//value):
                res += key
            num %= value                
        return res;
```

## 总结
- 有对应关系的时候，没必要写很多重复代码，用map或者两个列表对应起来，遍历一遍就可以 


------
