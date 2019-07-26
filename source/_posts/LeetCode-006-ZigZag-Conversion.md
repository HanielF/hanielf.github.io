---
title: LeetCode-006-ZigZag Conversion
tags:
  - LeetCode
  - Algorithm
  - Medium
  - Math
  - String
  - Split
categories:
  - LeetCode
urlname: leetcode-zigzag-conversion
comments: true
mathjax: false
date: 2019-04-15 20:47:37
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode.com/problems/zigzag-conversion/)   
The string "PAYPALISHIRING" is written in a zigzag pattern on a given number of rows like this: (you may want to display this pattern in a fixed font for better legibility)
> P   A   H   N
> A P L S I I G
> Y   I   R  

And then read line by line: **"PAHNAPLSIIGYIR"**
Write the code that will take a string and make this conversion given a number of rows:   
> string convert(string s, int numRows);
{% endnote %}
<!--more-->
{% note info %}
### Examples:
**Input:** s = "PAYPALISHIRING", numRows = 3
**Output:** "PAHNAPLSIIGYIR"

**Input:** s = "PAYPALISHIRING", numRows = 4
**Output:** "PINALSIGYAHRPI"
**Explanation:**
> P     I    N
> A   L S  I G
> Y A   H R
> P     I
{% endnote %}

## Solutions
- 想到的是用数学方法，遍历每一行，然后计算每一行字符的位置，逐个添加进去，是题解的第二种方法。每行的字符位置都满足一个数列，所以可以很简单的计算出来。时间复杂度是$$ O(n) $$.
- 题解还给了一种方法是按行排序，每一行建一个字符串，从左向右迭代源字符串，判断每个字符是属于哪一行，然后加到那一行的字符串上，最后对每行字符串拼接，原理感觉和上一种方法差不多。时间复杂度也是$$ O(n) $$  

## C++ Codes
自己想到的解法，用时12ms，内存10.2MB，超过99.51%感觉还是挺快的。
```C++
class Solution {
public:
    string convert(string s, int numRows) {
        int len = s.length();
        int num = ceil(len*1.0/numRows);
        string res = "";
        int d = 2*numRows-2;

        if(numRows==1)return s;
        for(int i=0;i<numRows;i++){   //遍历行
            for(int j=1;j<=num;j++){    //遍历每个Z
                if(i==0 && (j-1)*d<len){  //第一行
                    res+=s[(j-1)*d];
                } else if(i==(numRows-1) && ((j-1)*d+numRows-1)<len){ //最后一行
                    res+=s[(j-1)*d+numRows-1];
                } else{ //其余行
                        if((j-1)*d+i<len)
                            res+=s[(j-1)*d+i];
                        if(j*d-i<len)
                            res+=s[j*d-i];
                }
            }
        }
        return res;
    }
};
```

第二种遍历每个字符的解法:测试是20ms，12.6MB
```C++
class Solution {
public:
    string convert(string s, int numRows) {

        if (numRows == 1) return s;

        vector<string> rows(min(numRows, int(s.size())));
        int curRow = 0;
        bool goingDown = false;

        for (char c : s) {
            rows[curRow] += c;
            if (curRow == 0 || curRow == numRows - 1) goingDown = !goingDown;
            curRow += goingDown ? 1 : -1;
        }

        string ret;
        for (string row : rows) ret += row;
        return ret;
    }
};
```

## Python Codes

```python
class Solution:
    def convert(self, s: str, numRows: int) -> str:
        slen = len(s)
        num = math.ceil(slen/numRows);
        res = ""
        d = 2*numRows-2
        
        if numRows==1:
            return s
        for i in range(numRows):
            for j in range(1,num+1):
                if i==0 and (j-1)*d<slen:
                    res = res + s[(j-1)*d]
                elif i==numRows-1 and (j-1)*d+numRows-1<slen:
                    res = res + s[(j-1)*d+numRows-1]
                else :
                    if (j-1)*d+i<slen:
                        res = res + s[(j-1)*d+i]
                    if(j*d-i<slen):
                        res = res + s[j*d-i]
                        
        return res;
```

## 总结
- 很多题目可以直接从数学角度着手，会很方便
- 暴力法在很多时候都有用啊，时间复杂度不够再优化下说不定就过了


------
