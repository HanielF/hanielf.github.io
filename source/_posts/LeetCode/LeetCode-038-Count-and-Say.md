---
title: LeetCode-038-Count and Say
tags:
  - LeetCode
  - Algorithm
  - Easy
  - BruteForce
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-11-18 15:51:19
urlname: LeetCode-038-Count-and-Say
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/count-and-say/)

The count-and-say sequence is the sequence of integers with the first five terms as following:

`1` is read off as `"one 1"` or `11`.
`11` is read off as `"two 1s"` or `21`.
`21` is read off as `"one 2`, then `one 1"` or `1211`.

Given an integer n where 1 ≤ n ≤ 30, generate thenthterm of the count-and-say sequence.

Note: Each term of the sequence of integers will be represented as a string.

### Examples:
**Input:**
> 1
> 4

**Output:**
> "1"
> "1211"

{% endnote %}
<!--more-->

## Solutions
- 直接对上一次的结果进行从左往右的遍历，暴力求解
- 要注意的就是里面的细节和边界


## C++ Codes

```C++
class Solution {
public:
    string countAndSay(int n) {
        if (n == 1)
            return "1";
        int cnt = 1;
        string res = "", pre = "1";
        for (; cnt < n; cnt++) {
            int numCnt = 1;
            char numPre = pre[0];
            for (int i = 1; i < pre.length(); i++) {
            if (pre[i] == numPre)
                numCnt++;
            else {
                res += to_string(numCnt) + numPre;
                numPre = pre[i];
                numCnt = 1;
            }
            }
            res += to_string(numCnt) + numPre;
            pre = res;
            res = "";
        }
        // cout<<pre<<endl;
        return pre;
    }
};
```


------
