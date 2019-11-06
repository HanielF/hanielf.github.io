---
title: LeetCode-034-Find First and Last Position of Element in Sorted Array
tags:
  - LeetCode
  - Algorithm
  - Search
  - BinarySearch
  - List
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-11-06 12:10:35
urlname: LeetCode-034-Find First and Last Position of Element in Sorted Array 
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/)
Given an array of integers nums sorted in ascending order, find the starting and ending position of a given target value.

Your algorithm's runtime complexity must be in the order of O(log n).

If the target is not found in the array, return [-1, -1].

### Examples:
**Input:**
> nums = [5,7,7,8,8,10], target = 8 
> nums = [5,7,7,8,8,10], target = 6

**Output:**
> [3,4]
> [-1,-1]

{% endnote %}
<!--more-->

## Solutions
- 题目要求$$ O(log n) $$ 级别的时间复杂度，又是查找，所以肯定是二分查找比较快
- 这里要求的是找到最左边的target和最右边的target
- 两种思路
  - 直接按照原始二分，找到一个之后，向左右遍历
  - 直接找左边界和右边界，两次查找
- 下面采用的是第一种思路，第二种思路见题解：[二分查找算法细节详解](https://leetcode-cn.com/problems/find-first-and-last-position-of-element-in-sorted-array/solution/er-fen-cha-zhao-suan-fa-xi-jie-xiang-jie-by-labula/)


## C++ Codes
找到一个target之后往左右遍历，8ms，10.2MB

```C++
class Solution {
public:
    int binarySearch(vector<int>& nums, int target){
      int mid;
      int l=0,r=nums.size()-1;
      while(l<=r){
        mid = (l+r)/2;
        if(target == nums[mid]) return mid;
        else if(target>nums[mid]) l=mid+1;
        else r=mid-1;
      }
      return -1;
    }

    vector<int> searchRange(vector<int>& nums, int target) {
      int l,r;
      int pos = binarySearch(nums,target);
      if(pos==-1) return {-1,-1};
      else{
        l = r = pos;
        while(r+1<nums.size() && nums[r+1]==target) r++;
        while(l-1>=0 && nums[l-1]==target) l--;
      }
      return {l,r};
    }
};
```


------
