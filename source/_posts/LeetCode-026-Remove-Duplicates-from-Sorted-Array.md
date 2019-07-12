---
title: LeetCode-026-Remove Duplicates from Sorted Array
tags:
  - LeetCode
  - Duplicates
  - Array
  - Sort
  - Easy
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-05-26 00:56:26
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/remove-duplicates-from-sorted-array/)   
Given a sorted array nums, remove the duplicates in-place such that each element appear only once and return the new length.

Do not allocate extra space for another array, you must do this by modifying the input array in-place with O(1) extra memory.

### Examples:
**Input:**[1,1,2]
**Output:**Your function should return length = 2, with the first two elements of nums being 1 and 2 respectively. It doesn't matter what you leave beyond the returned length.

{% endnote %}
<!--more-->

## Solutions
- 简单说就是去掉**排好序的**重复元素的嘛，然后要求空间复杂度$$ O(1) $$
- 利用双指针的思想，第一个指针指向排好序的最后一个元素，第二个指针指向当前的元素
- 如果当前元素不等于第一个指针指向的元素，那第一个指针自增，并赋值当前元素值
- 还有一种也能过的本办法，已经排序好了，所以遇到一个重复的删除一个就行，不过时间会花的很多

## C++ Codes
双指针法，时间复杂度是$$ O(n) $$，花了32ms

```C++
class Solution {
public:
    int removeDuplicates(vector<int>& nums) {
        if(nums.size()==0)return 0;
        //pre指向无重复排好序的最后一个位置
        int pre=0;
        for(int i=1;i<nums.size();i++){
            if(nums[i]!=nums[pre]){
                pre++;
                nums[pre]=nums[i];
            }
                
        }
        return pre+1;
    }
};

```

## C++ Codes
暴力删除法，时间复杂度$$ O(n) $$，用时252ms
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

## 总结
- 双指针很多时候都是能用上的，特别是就地算法，或者查找、排序等算法 
- 简单题就不写Py的版本了


------
