---
title: LeetCode-035-Search Insert Position
tags:
  - LeetCode
  - Algorithm
  - Easy
  - BinarySearch
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-11-07 11:22:45
urlname: LeetCode-035-Search-Insert-Position
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/search-insert-position/submissions/)
target is found. If not, return the index where it would be if it were inserted in order.  
You may assume no duplicates in the array.

### Examples:
**Input:**
> [1,3,5,6], 5
> [1,3,5,6], 2
> [1,3,5,6], 7
> [1,3,5,6], 0

**Output:**
> 2
> 1
> 4
> 0

{% endnote %}
<!--more-->

## Solutions
- 简单的二分查找，然后找不到的话，就返回r+1。因为while的条件是(l>=r)，所以，出了while循环就是l>r了，因此返回r+_1就可以了


## C++ Codes

```C++
class Solution {
public:
    int searchInsert(vector<int>& nums, int target) {
      int l=0, r=nums.size()-1, mid;
      // 条件是l<=r
      while(l<=r){
        mid = (l+r)/2;
        if(target==nums[mid]) return mid;
        if(target>nums[mid]){
          l=mid+1;
        } else{
          r=mid-1;
        }
      }
      // 到这里代表没找到，l>r, 返回r+1
      return r+1;
    }

};
```


------
