---
title: LeetCode-003-Longest Substring Without Repeating Characters
comments: true
mathjax: false
date: 2019-04-13 01:02:13
tags: 
  - Algorithm
  - LeetCode
  - String
  - Math
  - Medium
categories: LeetCode
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode.com/problems/longest-substring-without-repeating-characters/) 
Given a string, find the length of the **longest substring** without repeating characters.
### Examples:
> "abcabcbb" ---> 3
> "bbbbb" ---> 1
> "pwwkew" ---> 3
{% endnote %}
<!--more-->

## Solutions
- 第一反应下想到的也是暴力遍历，写一个判断有没有重复字符串的函数，然后遍历每个子串，时间复杂度是$$ O(n^3) $$，太高了，导致第一次提交没过，后面优化了下暴力法，过了。
- 除了暴力，题解上给出了**滑动窗口**的解法，利用前后两个指针i,j 不断移动来遍历字符串以及子串，i,j中间的就是没有重复字符的,如果遇到重复字符，就把i自增，一直到没有重复字符为止（就是相当于找到和j处的相同字符的为止），时间复杂度是$$ O(n) $$
- 最后是优化的滑动窗口，先记录每个字符第一次出现的位置pos，然后第二次出现的时候，将第一个指针i直接变成pos+1
- 对上面的优化还有一个技巧是利用整数数组代替Map，也可以用bitmap每一位表示一个字符数量

## C++ Codes
这是我优化过的暴力法，勉强过了，判断函数那边用了太大的数组，可以用bitmap降低空间复杂度

用了72ms,20.1MB...僵硬

```C++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        if(s.length()==0)return 0;
        int max=1;
        //i是最大长度，j是每次遍历的起始位置
        for(int j=0;j<s.length();j++){
            //这里让i直接从max开始，可以少一点没用的遍历
            for(int i=max;i+j<=s.length();i++){
                //如果有重复字符直接退出循环
                if(!judgeSubstring(s.substr(j,i))) break;
                max=i;
            }
        }
        return max;
    }

    bool judgeSubstring(string substr){
        int tmp[128]={0};
        for(int i=0;i<substr.length();i++){
            //如果不是0就肯定有重复字符
            if(tmp[int(substr[i])])return false;
            tmp[int(substr[i])]+=1;
        }
        return true;
    }
};
```

好了，这是第三种方法对应的代码，这次只用了12ms,9.1MB，话说感觉这种滑动窗口有点像快排...

```C++
class Solution {
public:
    int lengthOfLongestSubstring(string s) {
        int n=s.length();
        //用来保存每个字符的下一个位置，用于出现重复字符的时候直接赋值给i
        int cNum[128]={0};      
        int max=0;
        //i是第一个指针，j是第二个
        for(int j=0,i=0;j<n;j++){
            //如果不为0就是出现了和s[j]一样的字符，让i等于那个字符的后一个位置
            if(cNum[s[j]]){
                i = cNum[s[j]]>i?cNum[s[j]]:i;
            }
            //这里不管有没有重复字符都要更新max和cNum[s[j]]
            //很明显，如果有，i已经被更新，相当于已经变成了没有重复字符的情形，位置也要更新
            max=(j-i+1)>max?(j-i+1):max;
            cNum[s[j]]=j+1;
        }
        return max;
    }

};
```

## Python Codes
用时52ms，12.7MB

```Python
class Solution(object):
    def lengthOfLongestSubstring(self, s):
        """
        :type s: str
        :rtype: int
        """
        cNum=dict()
        res=0
        start=0
        for i in range(len(s)):
            if s[i] in cNum:
                start=max(start,cNum[s[i]])
            res=max(res,i-start+1)
            cNum[s[i]]=i+1
        return res
```

## 总结
总感觉自己想的太简单，优化还是有点难度的，这道题关键的就是一次遍历，用两个指针表示字符串，并且没有单独拿出字符串判断，而是用set或者map、dict这种方式记录当前子串的内容，用于判断是否有没有出现重复字符，优化的时候减少无用的遍历

------
