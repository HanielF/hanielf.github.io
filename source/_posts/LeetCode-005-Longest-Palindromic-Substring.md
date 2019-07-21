---
title: LeetCode-005-Longest Palindromic Substring
tags:
  - LeetCode
  - Medium
  - Math
  - String
  - Palindromic
  - DP
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-04-14 21:22:07
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode.com/problems/longest-palindromic-substring/solution/)   
Given a string s, find the longest palindromic substring in s. You may assume that the maximum length of s is 1000.

### Examples:
> "babad" ---> "bab"
> "cbbd" ---> "bb"
{% endnote %}
<!--more-->

## Solutions
- **暴力法**：首先想到的当然是暴力法...暴力大法好/斜眼笑
- **遍历中心点暴力:**感觉直接暴力子串太幼稚，想了想可以遍历所有字串的中心点，于是产生了下面第一种解法，遍历每个下标，然后对子串是奇数个字符和偶数个字符分别查找最长子串，记录最长子串起始下标和长度，最后返回。时间复杂度是$$ O(n^2) $$
- **动态规划: **看了题解，里面讲到了动态规划的解法，首先是有一个二维数组P[i, j]，用于保存i-j位置这段子串是不是回文，是就是true，不是就是false，初始化P中长度为1和2的子串，P[i-1, j+1]=True相当于：**P[i, j]==true && S[i-1]==S[j+1]**
- **翻转字符串和原字符串比较：**:把字符串翻转，然后和原来的比较，找到最长的公共子串就是要的结果，这种方法有缺陷就是可能存在非回文子串的反向副本，改进的办法也很简单，就是比较找到的公共子串的索引是否和反向子串的原索引相同，不相同就可以排除了


## C++ Codes
遍历中心点法：用了32ms，8.8MB

```C++
class Solution {
public:
    string longestPalindrome(string s) {
        int maxLen = 1;
        int pos = 0;
        int n = s.length();
        for(int i=0;i<n;i++){
            //判断子串为奇数个字符时
            int tmp = 1;
            while(i-tmp>=0 && i+tmp<n && s[i-tmp]==s[i+tmp]){
                if((2*tmp+1)>maxLen){
                    maxLen = 2*tmp+1;
                    pos = i-tmp;
                }
                tmp++;
            }
            
            //断子串为偶数个字符时
            tmp = 1;
            while(i-tmp+1>=0 && i+tmp<n && s[i-tmp+1]==s[i+tmp] ){
                if(2*tmp>maxLen){
                    maxLen = 2*tmp;
                    pos = i-tmp+1;
                }
                tmp++;
            }
        }    

        return s.substr(pos,maxLen);
    }
};
```

动态规划版本：用了140ms，13.1MB

```C++
class Solution {
public:
    string longestPalindrome(string s) {
        int n = s.length();
        if(n==0)return "";
        
        int p[n][n]={0};
        int pos = 0;
        int maxLen = 1;
        
        //初始化数组边界
        for(int i=0;i<n-1;i++){
            p[i][i]=1;
            p[i][i+1]=s[i]==s[i+1]?1:0;
            if(p[i][i+1]){
                pos = i;
                maxLen = 2;
            }
        }
        p[n-1][n-1]=1;
        
        //动规循环
        for(int i=2;i<n;i++){   //长度-1
            for(int j=0;j+i<n;j++){   //起始下标
                if( p[j+1][j+i-1]==1 && s[j]==s[j+i]){
                    p[j][j+i]=1;
                    if((i+1)>maxLen){
                        maxLen = i+1;
                        pos = j;
                    }
                }
            }
        }
        return s.substr(pos,maxLen);
    }
};
```

## 总结
- 算法题要注意边界问题，遍历的时候别下标越界
- C++中数组要注意初始化，不会像java那样有默认值
- 动态规划注意初始化边界
- 暴力的时候注意子串长度还可以是偶数

------
