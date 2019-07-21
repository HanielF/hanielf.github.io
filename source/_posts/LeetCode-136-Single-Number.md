---
title: LeetCode-136-Single Number
tags:
  - LeetCode
  - Xor
  - Math
  - Linear
  - Easy
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-05-26 00:57:09
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## 前言
看到有算法面试集锦，刷点题，刚开始刷，前面都是简单题，随便看看了

## [Problem](https://leetcode-cn.com/problems/single-number/)   
Given a non-empty array of integers, every element appears twice except for one. Find that single one.

**Note:**
Your algorithm should have a linear runtime complexity. Could you implement it without using extra memory?

### Examples:
**Input:**[2,2,1]
**Output:**1

{% endnote %}
<!--more-->

## Solutions
- 刚开始就是向简单的遍历一遍，用map存出现次数，但是要$$ O(1) $$ 的空间复杂度，所以不可以
- 然后想到排序，排序完找，但是要线性时间复杂度。。。
- 最后是用异或的方法，因为题目说的是，只有一个出现一次，其他的全部出现两次
- 两个相同的数异或（XOR），结果为0，0和0异或当然还是0，然后就剩下了单独的一个数字，0和非0数异或等于非0数本身


## C++ Codes
时间复杂度$$ O(n) $$，空间复杂度$$ O(1) $$

```C++
class Solution {
public:
    int singleNumber(vector<int>& nums) {
        if(nums.size()==0)return 0;
        int res = 0;
        for(int i=0;i<nums.size();i++){
            res = res^nums[i];
        }
        return res;
    }
};
```


## 总结
- 注意题目细节，每个重复的数字都是出现两次 


------
