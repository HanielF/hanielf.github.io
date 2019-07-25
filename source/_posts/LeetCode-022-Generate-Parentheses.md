---
title: LeetCode-022-Generate Parentheses
tags:
  - LeetCode
  - Algorithm
  - Parentheses
  - DP
  - Medium
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-07-23 01:00:33
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/generate-parentheses/)   
Given n pairs of parentheses, write a function to generate all combinations of well-formed parentheses. 

### Examples:
**Input:** n=3
**Output:**
> [
>   "((()))",
>   "(()())",
>   "(())()",
>   "()(())",
>   "()()()"
> ]
{% endnote %}
<!--more-->

## Solutions
- 可以暴力求解, 将所有括号的全排列找出来,然后把里面括号匹配的拿出来
- 动态规划, 第n个括号总是在第n-1个括号的基础上添加的,假设最左边的左括号是添加上去的. 那么, 前面的n-1个括号便分为两部分,在新加入括号内部的, 和不在新括号内部的. 我们遍历在括号内部的括号对数0-idx-1,然后遍历在括号内的每个排列, 以及不在括号内的每个排列, 将他们拼凑到一起


## C++ Codes
四层循环, 第一层求2-n的结果, 第二层遍历在新括号内部的括号对数, 第三层和第四层, 遍历在新括号内部的括号排列, 和不在括号内部的括号排列, 加起来就是一个新的排列

```C++
class Solution {
public:
    vector<string> generateParenthesis(int n) {
        vector<vector<string> > dp(n+1);
        dp[0]={""};
        dp[1]={"()"};
        //假设最左边的左括号是第n对括号新加进来的
        //遍历在第一个左括号对应的括号内的pair数, [0, idx-1]
        for(int idx=2;idx<=n;idx++) 
            for(int i=0;i<=idx-1;i++)   //在第一个左括号内的括号数
                for(string si:dp[i])    //在第一个括号内的括号数的每个排列
                    for(string sk:dp[idx-1-i])  //不在第一个括号内的括号的每个排列
                        dp[idx].push_back("("+si+")"+sk);

        return dp[n];
    }
};
```

## Python Codes
算法同C++

```python
class Solutin:
    def generateParenthesis(self, n: int) -> List[str]:
        if n == 0:
            return []
        total_l = []
        total_l.append([None])
        total_l.append(["()"])
        for i in range(2,n+1):  # 开始计算i时的括号组合，记为l
            l = []
            for j in range(i): #遍历所有可能的括号内外组合
                now_list1 = total_l[j]
                now_list2 = total_l[i-1-j]
                for k1 in now_list1:  #开始具体取内外组合的实例
                    for k2 in now_list2:
                        if k1 == None:
                            k1 = ""
                        if k2 == None:
                            k2 = ""
                        el = "(" + k1 + ")" + k2
                        l.append(el)
            total_l.append(l)
        return total_l[n]
```

## 总结
- 动态规划往往是递归转化过来, 如果能想到递归, 那可以考虑下DP怎么做, 当然, 暴力求解通常就是递归求解


------
