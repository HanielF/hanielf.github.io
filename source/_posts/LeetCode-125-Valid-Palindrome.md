---
title: LeetCode-125-Valid Palindrome
tags:
  - LeetCode
  - Algorithm
  - Palindrome
  - String
  - Easy
categories:
  - LeetCode
urlname: leetcode-valid-palindrome
comments: true
mathjax: false
date: 2019-06-02 00:55:08
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/valid-palindrome/)   
Given a string, determine if it is a palindrome, considering only alphanumeric characters and ignoring cases.

Note: For the purpose of this problem, we define empty string as valid palindrome.

### Examples:
**Input:**"A man, a plan, a canal: Panama"
**Output:**true

{% endnote %}
<!--more-->

## Solutions
- 如果是单纯的回文串验证，那很简单，很多种方法
- 可以直接用函数将字符串逆序作比较
- 可以用两个指针指向头和尾，逐个字符比较
- 这道题因为还有空格，标点符号，大小写的问题，需要简单的处理下


## C++ Codes
8ms，超99%

```C++
class Solution {
public:
    bool isPalindrome(string s) {
        //全部变成小写并去掉非数字和字符的
        transform(s.begin(),s.end(),s.begin(),::tolower);
        string res;
        for(int i=0;i<s.length();i++){
            if((s[i]>='0' && s[i]<='9') || (s[i]>='a' && s[i]<='z')) 
                res+=s[i];
        }
        cout<<res<<endl;
        if(res.empty()) return true;  
        int l=0, r=res.length()-1;
        while(l<res.length() && r>=0 && l<=r){
            if(res[l]==res[r]){
                l++;
                r--;
            }else{
                return false;
            }
        }
        return true;
    }
};
```

{% note info %}
C++里面没有直接对字符串进行全部转换大写或者小写的方法
可以用`transform(s.begin(),s.end(),s.begin(),::tolower)`
更多方法可以进行百度，如果懒得找也可以自己遍历字符修改大小写
{% endnote %}


## 总结
- 要注意的就是字符串的预处理，并且处理之后如果是空串要返回true 


------
