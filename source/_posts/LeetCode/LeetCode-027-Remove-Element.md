---
title: LeetCode-027-Remove-Element
tags:
  - LeetCode
  - Algorithm
  - Easy
  - Vector
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-10-24 09:27:06
urlname:
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/remove-element/)   
Given an array nums and a value val, remove all instances of that value in-place and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

The order of elements can be changed. It doesn't matter what you leave beyond the new length.

### Examples:
**Input:**
> Given nums = [3,2,2,3], val = 3,

**Output:**
> Your function should return length = 2, with the first two elements of nums being 2.

{% endnote %}
<!--more-->

## Solutions
- 题目要求就地，不使用额外空间，所以只能在原数组修改
- 双指针，以前的题目中用了很多。如果后一个指针内容和前一个不相等，就赋值，然后全部自增。否则就只有后指针自增。


## C++ Codes

```C++
class Solution {
public:
    int removeElement(vector<int>& nums, int val) {
        int pos = 0;
        for(int i=0;i<nums.size();i++){
            if(nums[i]!=val) nums[pos++]=nums[i];
        }
        return pos;
    }
};
```

