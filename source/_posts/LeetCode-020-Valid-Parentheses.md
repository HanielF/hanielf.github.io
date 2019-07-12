---
title: LeetCode-020-Valid Parentheses
tags:
  - LeetCode
  - Stack
  - Pair
  - String
  - Easy
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-05-22 13:59:04
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode.com/problems/valid-parentheses/)   
Given a string containing just the characters '(', ')', '{', '}', '[' and ']', determine if the input string is valid.

An input string is valid if:
1. Open brackets must be closed by the same type of brackets.
2. Open brackets must be closed in the correct order.

Note that an empty string is also considered valid.

### Examples:
**Input:**"()"
**Output:** true
**Input:**"[(])"
**Output:** false

{% endnote %}
<!--more-->

## Solutions
- 简单的利用栈，如果是左括号，就入栈，如果是右括号，就和栈顶匹配，相同，就让栈顶出栈，否则返回错误 ，这里用哈希表映射左右括号


## C++ Codes

```C++
class Solution {
public:
    bool isValid(string s) {
        if(s.length()==0)return true;
        
        //定义栈和哈希表
        stack<char> chstack;
        map<char,char> mp;
        mp[')']='(';
        mp[']']='[';
        mp['}']='{';

        for(int i=0;i<s.length();i++){
            //左括号入栈
            if(s[i]=='(' || s[i]=='[' || s[i]=='{'){
                chstack.push(s[i]);
            }
            //右括号匹配栈顶
            else{
                if(chstack.empty()) return false;
                if(mp[s[i]]==chstack.top()) chstack.pop();
                else return false;
            }
        }
        //如果匹配完了栈为空，就是符合要求的
        if(chstack.empty())return true;
        else return false;
    }
};
```

## Python Codes

```python
class Solution:
    def isValid(self, s: str) -> bool:
        stack = []
        lookup = {
            "(":")",
            "[":"]",
            "{":"}"
        }
        for alp in s:
            if alp in lookup:
                stack.append(alp)
                continue
            if stack and lookup[stack[-1]] == alp:
                stack.pop()
            else:
                return False
        return True if not stack else False
```

## 总结
- 做过的题，简单的利用出栈入栈进行括号匹配 


------
