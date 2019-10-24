---
title: LeetCode-017-Letter Combinations of a Phone Number
tags:
  - LeetCode
  - Algorithm
  - Recursive
  - Map
  - Medium
categories:
  - LeetCode
urlname: leetcode-letter-combinations-of-a-phone-number
comments: true
mathjax: false
date: 2019-05-22 13:48:56
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode.com/problems/letter-combinations-of-a-phone-number/)   
Given a string containing digits from 2-9 inclusive, return all possible letter combinations that the number could represent.
2："abc", 3-"def", 4-"ghi", 5-"jkl", 6-"mno", 7-"pqrs", 8-"tuv", 9-"wxyz"

A mapping of digit to letters (just like on the telephone buttons) is given below. Note that 1 does not map to any letters.

### Examples:
**Input:**"23"
**Output:**["ad", "ae", "af", "bd", "be", "bf", "cd", "ce", "cf"].

{% endnote %}
<!--more-->

## Solutions
- 递归，当后面的全部完成了排列之后，前面的加上就是全部的排列方式，这里要记录前缀，到最后没有数字的时候，就添加前缀并返回


## C++ Codes

```C++
class Solution {
public:        
    string mapChar[10] = {"","","abc","def","ghi","jkl","mno","pqrs","tuv","wxyz"};
    vector<string> letterCombinations(string digits) {
        if(digits.length()==0)return {};
        vector<string> res;
        recurDigit(digits,"",res);
        return res;
    }
    
    void recurDigit(string digits,string prefix, vector<string>& res){
        //边界条件
        if(digits=="") {
            res.push_back(prefix);
            return;
        }
        
        string key = mapChar[digits[0]-'0'];
        int n = key.length();
        for(int i=0;i<n;i++)
            recurDigit(digits.substr(1), prefix+key[i], res);
        
    }
    
};
```

## 总结
- 递归要处理好边界条件，这里是当digits为空的时候，就返回
- 这里用了记录前缀的方式，感觉也可以使用前面的加上后缀的方式，返回的时候返回字符，应该也可以
- 这题让我想到了全排列的解法


------
