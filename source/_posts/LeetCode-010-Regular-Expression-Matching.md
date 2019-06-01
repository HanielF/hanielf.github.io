---
title: LeetCode-010-Regular Expression Matching
tags:
  - LeetCode
  - DP
  - String
  - Match
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-04-27 01:43:57
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode.com/problems/regular-expression-matching/)   
Given an input string (s) and a pattern (p), implement regular expression matching with support for '.' and '\*'.

> '.' Matches any single character.
> '\*' Matches zero or more of the preceding element.

The matching should cover the entire input string (not partial).

**Note**:
> s could be empty and contains only lowercase letters a-z.
> p could be empty and contains only lowercase letters a-z, and characters like . or \*.

### Examples:
**Input:** s="aa" p="a"
**Output:** false
**Input:** s="aa" p="a\*"
**Output:** true
**Input:** s="ab" p=".\*"
**Output:** true
**Input:** s="aab" p="c\*a\*b"
**Output:** true
**Input:** s="mississippi" p="mis\*is\*p\*."
**Output:** false

佛了，一道题搞了两个多小时，还是有bug...貌似方法错了，题解说了递归和DP，我用到了递归，可还是不行，DP就直接没想到，先码着回头改

{% endnote %}
<!--more-->

## Solutions
- 递归，首字母匹配，并且后面的子串也匹配
- DP动态规划，自上而下和自下而上两种，自下而上的不用递归，时间复杂度上会好点
- 自下而上的方法，在判断前一个字符的时候，因为后面的部分已经判断过了，所以可以直接从dp表里面取数据

## C++ Codes
**递归方法**

```C++
class Solution {
public:
    bool isMatch(string s, string p) {
        if(p.empty())return s.empty();
        //第一个字符匹配或者是.匹配符
        bool first_match = (!s.empty() &&
                            (p[0]==s[0]||p[0]=='.'));

        //如果是字符加*
        if(p.length()>=2 && p[1]=='*'){
            return (isMatch(s, p.substr(2)) ||
                (first_match && isMatch(s.substr(1), p)));
        //如果没有*或者少于两个字符
        } else {
            return first_match &&
                isMatch(s.substr(1), p.substr(1));
        }
    }
};
```

## Java Codes
**自上而下的DP**

```Java
enum Result {
    TRUE, FALSE
}

class Solution {
    Result[][] memo;

    public boolean isMatch(String text, String pattern) {
        memo = new Result[text.length() + 1][pattern.length() + 1];
        return dp(0, 0, text, pattern);
    }

    public boolean dp(int i, int j, String text, String pattern) {
        if (memo[i][j] != null) {
            return memo[i][j] == Result.TRUE;
        }
        boolean ans;
        if (j == pattern.length()){
            ans = i == text.length();
        } else{
            boolean first_match = (i < text.length() &&
                                   (pattern.charAt(j) == text.charAt(i) ||
                                    pattern.charAt(j) == '.'));

            if (j + 1 < pattern.length() && pattern.charAt(j+1) == '*'){
                ans = (dp(i, j+2, text, pattern) ||
                       first_match && dp(i+1, j, text, pattern));
            } else {
                ans = first_match && dp(i+1, j+1, text, pattern);
            }
        }
        memo[i][j] = ans ? Result.TRUE : Result.FALSE;
        return ans;
    }
}
```

**自下而上的DP**

```Java
class Solution {
    public boolean isMatch(String text, String pattern) {
        boolean[][] dp = new boolean[text.length() + 1][pattern.length() + 1];
        dp[text.length()][pattern.length()] = true;

        for (int i = text.length(); i >= 0; i--){
            for (int j = pattern.length() - 1; j >= 0; j--){
                boolean first_match = (i < text.length() &&
                                       (pattern.charAt(j) == text.charAt(i) ||
                                        pattern.charAt(j) == '.'));
                if (j + 1 < pattern.length() && pattern.charAt(j+1) == '*'){
                    dp[i][j] = dp[i][j+2] || first_match && dp[i+1][j];
                } else {
                    dp[i][j] = first_match && dp[i+1][j+1];
                }
            }
        }
        return dp[0][0];
    }
}
```

## C++ Codes

暂时码着自己代码，还有bug，在s遍历完了，p没遍历完时会出错...代码太丑陋了，只会暴力吗，僵硬...

```C++
class Solution {
public:
    bool isMatch(string s, string p) {
        int si=0,pi=0;
        int sn=s.length(), pn=p.length();
        for(pi;pi<pn,si<sn;pi++){
            cout<<s.substr(si)<<"\t"<<p.substr(pi)<<endl;
            switch(p[pi]){
                case '.':        //任意单个字符
                    si++;
                    break;
                case '*':        //零个或者多个前面的元素
                    if(p[pi-1]=='.') 
                    {
                        if(pi==pn-1)re后面还要改turn true;    //如果.*是最后两个字符，直接返回true
                        while((p[pi+1]=='.'||p[pi+1]=='*')&&pi<pn){ //如果后面持续特殊字符
                            pi++;
                        }
                        if(pi==pn)return true;  //如果到最后全是特殊字符

                        //p[pi+1]是普通字符,让si++,找s[si]==p[pi+1]
                        while(s[si]!=p[pi+1] && si<sn) si++;
                        break;
                    }
                    if(p[pi-1]!=s[si])//零个
                        break;
                    
                    //一个或者多个,后面部分的需要也匹配
                    if(p[pi-1]==s[si])si++;
                    if(si==sn && pi==pn-1)return true;
                    else if(si==sn)si--;
                    while(p[pi-1]==s[si] &&(si<sn && pi+1<pn && isMatch(s.substr(si),p.substr(pi+1))==false)){
                        si++;
                    }

                    break; 
                default:
                    if(p[pi]==s[si] && si<sn){           //相等就继续遍历
                        si++;
                        break;
                    }//不等，则判断后面是否有*，如果有，这个字符可以跳过,注意这里si并没有自增
                    else if(p[pi]!=s[si] && pi+1<pn && p[pi+1]=='*'){
                        break;
                    }
                    else return false;
            }
        }
        if(pi==pn && si==sn)return true;
        return false;
    }
};
```

## Python Codes

评论区有人一行Java正则就搞定了，我觉得，python一行也够了...

```python
略
```

## 总结
- 正则匹配这里，其实应该先想到递归和DP的，但是思维太局限，训练的少，虽然会递归和DP，但是想不起来用..仿佛读了假书..
- 算法题需要训练，不光是会算法，不然都不知道题目应该用什么算法...
- 这道题来说，递归还是挺容易写的，首字母匹配，后面的也匹配就可以，再考虑下\*字符的情况


------
