---
title: LeetCode-014-Longest Common Prefix
tags:
  - LeetCode
  - Prefix
  - String
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-05-18 12:55:53
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## 前言
一道简单题，看到腾讯50题里有，还是花点时间补上了，没想到简单题方法还这么多

## [Problem](https://leetcode.com/problems/longest-common-prefix/)   
Write a function to find the longest common prefix string amongst an array of strings.

If there is no common prefix, return an empty string "".

### Examples:
**Input:**["flower","flow","flight"]
**Output:**"fl"

{% endnote %}
<!--more-->

## Solutions
- 找公共前缀这个问题，比较简单的想法就是一个个比较嘛，最先想到的也是这种方法，以第一个字符串为基准，从前往后比较每个字符是否和后面的字符全部相等，如果有不等的，那肯定前面的就是最长的公共前缀了，如果都相等，那就继续比较后一个字符
- 另一种差不多的方法是从后往前，先比较最长的，再慢慢减小
- 优化的办法是二分查找，minLen作为最短字符串的长度，二分查找这个是否为最长公共前缀
- 题解中还有分治法，这个算法的思路来自于LCP操作的结合律。 我们可以发现： LCP(S1…Sn)=LCP(LCP(S1…Sk),LCP(Sk+1…Sn))，其中LCP(S1...Sn)是字符串[S1...Sn]的最长公共前缀
- 基于二分查找，更进一步是用字典树，详细的看题解把

## C++ Codes
第一种解法，最容易想到的
```C++
class Solution {
public:
    string longestCommonPrefix(vector<string>& strs) {
        string prefix;
        int flag = 1;
        if(strs.size()==0 || strs[0].length()==0)return prefix;

        for(int i=0;i<strs[0].length();i++){
            for(int j=1;j<strs.size();j++){
                if(strs[j].length()<i+1 || strs[j][i]!=strs[0][i]){
                    flag = 0;
                    break;
                }
            }

            if(flag) prefix += strs[0][i];
            else break;
        }
        return prefix;
    }
};
```

## 总结
- emm，简单题也有这么多种做法...
- 最长公共前缀可以1.遍历前缀，2.二分查找前缀，3.分治找前缀，4.字典树找前缀


------
