---
title: LeetCode-008-String to Integer (atoi)
tags:
  - LeetCode
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-04-16 22:26:17
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem]()   

### Examples:
**Input:**
**Output:**

{% endnote %}
<!--more-->

## Solutions
- 没啥思维量，就是简单的遍历每个字符，然后设置两个flag，一个用于判断有没有到数字部分，第二个判断是不是负数，如果没到数字部分，就要分情况讨论，如果到了数字部分，判断到没到非数字，到了就退出，没到就每个数字加到结果上


## C++ Codes
每个字符遍历过去的方法，12ms，8.4MB
```C++
class Solution {
public:
    int myAtoi(string str) {
        long long tmp = 0;
        int n = str.length();
        if(n==0)return 0;
        bool flag = false;//是否到数字部分
        int flag2 = 1;//负数标志，有就变为-1

        for(int i=0;i<n;i++){
            //已经开始数字部分的情况，即flag = true
            if(tmp>INT_MAX)break;
            if(flag && (str[i]<'0' || str[i]>'9')) break;
            if(flag) tmp = tmp*10+str[i]-'0';

            //还没到数字的情况，前面那部分，分为空格、非数字非符号、数字或正负号
            if(!flag && str[i]==' ') continue;
            if(!flag && str[i]!='+' && str[i]!='-' && (str[i]<'0' || str[i]>'9') )
                return 0;
            if(!flag && (str[i]=='+' || str[i]=='-'|| (str[i]>='0'&&str[i]<='9'))){
                flag = true;
                if(str[i]=='-')flag2 = -1;
                if(str[i]=='+'||str[i]=='-')continue;
                tmp = tmp*10+str[i]-'0';
            }
        }

        tmp*=flag2;
        if(tmp>INT_MAX) return INT_MAX;
        if(tmp<INT_MIN) return INT_MIN;
        return (int)tmp;
    }
};
```

## Python Codes
借用了[HazzaCheng的代码](http://chengfeng96.com/blog/2017/03/07/LeetCode-No-8-String-to-Integer-atoi/),这道题用python解简直作弊..
```python
class Solution(object):
    def myAtoi(self, s):
        ls = list(s.strip())
        if len(ls) == 0:
            return 0

        sign = -1 if ls[0] == '-' else 1
        if ls[0] in ['-', '+']:
            del ls[0]
        res, i = 0, 0
        while i < len(ls) and ls[i].isdigit():
            res = res * 10 + ord(ls[i]) - ord('0')
            i += 1
        return max(-2 ** 31, min(sign * res, 2 ** 31 - 1))
```

## 总结
- 要注意各种情况的判断
- 就算用了long long作为中转，在循环中也要判断是否已经超过int表示范围，提前break，可能出现longlong都无法表示的测试
- 注意空串
- 注意前面的符号
- 不应该进行的操作及时continue掉
- 还是会出现'+'号的

------
