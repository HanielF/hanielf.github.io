---
title: LeetCode-026-Remove Duplicates from Sorted Array
tags:
  - LeetCode
  - Algorithm
  - Vector
  - Easy
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-10-24 09:19:10
urlname: Remove-Duplicates-from-Sorted-Array
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/submissions/)   
Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.
### Examples:
**Input:**
> Given nums = [1,1,2]

**Output:**
> Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively.

{% endnote %}
<!--more-->

## Solutions
- 和前一个元素比较，直接对vector进行元素删除 


## C++ Codes

```C++
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        if(nums.size()==0)return 0;
        for(int i=1;i<nums.size();){
            if(nums[i]==nums[i-1]){
                nums.erase(nums.begin()+i);
            }else{
                i++;
            }
        }
        return nums.size();
    }
};
```



------
