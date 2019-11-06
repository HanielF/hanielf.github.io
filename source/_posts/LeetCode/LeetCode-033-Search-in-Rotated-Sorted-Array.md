---
title: LeetCode-033-Search in Rotated Sorted Array
tags:
  - LeetCode
  - Algorithm
  - List
  - Search
  - BinarySearch
  - Medium
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-11-05 12:25:45
urlname: LeetCode-033-Search-in-Rotated-Sorted-Array
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/search-in-rotated-sorted-array/submissions/)
Suppose an array sorted in ascending order is rotated at some pivot unknown to you beforehand.

(i.e., [0,1,2,4,5,6,7] might become [4,5,6,7,0,1,2]).

You are given a target value to search. If found in the array return its index, otherwise return -1.

You may assume no duplicate exists in the array.

Your algorithm's runtime complexity must be in the order of O(log n).

### Examples:
**Input:**
> nums = [4,5,6,7,0,1,2], target = 0
> nums = [4,5,6,7,0,1,2], target = 3

**Output:**
> 4
> -1

{% endnote %}
<!--more-->

## Solutions
- 题目要求时间复杂度要在$$ O(log n) $$，所以很自然要想到二分搜索
- 基础版本的二分搜索很明显不适用，因为有一个分隔点，要在此基础上修改
- 思路是：判断分隔点在前半部分还是在后半部分，以mid为分界。找到有序的部分，在这部分上应用二叉搜索查找target。如果target不在有序的部分内，则转移到无序的部分，然后继续这样找有序的部分。具体看代码注释


## C++ Codes

```C++
#include<iostream>
#include<vector>

using namespace std;

int search(vector<int>& nums, int target) {
  int l=0,r=nums.size()-1;
  int mid = (l+r)/2;

  while(l<=r){
    mid = (l+r)/2;

    // 如果nums[mid]==target，就返回
    if(target==nums[mid]) return mid;

    // 如果前半部分有序，即切割点在后半部分
    // 注意这里是大于等于，将mid=l的情况分为前半部分

    if(nums[mid]>=nums[l]){
      // 如果target在有序部分，则二分
      if(target>=nums[l] && target<nums[mid]){
        r=mid-1;
      }
      else {
        l=mid+1;
      }
    // 如果后半部分有序，即切割点在前半部分
    } else{
      if(target>nums[mid] && target<=nums[r]){
        l=mid+1;
      } else{
        r=mid-1;
      }
    }
  }
  return -1;
}

int main(){
  vector<int> nums={3,1};
  cout<<search(nums, 1);
  return 0;
}
```

## Python Codes

```python
class Solution:
    def search(self, nums, target):
        """
        :type nums: List[int]
        :type target: int
        :rtype: int
        """
        def find_rotate_index(left, right):
            if nums[left] < nums[right]:
                return 0
            
            while left <= right:
                pivot = (left + right) // 2
                if nums[pivot] > nums[pivot + 1]:
                    return pivot + 1
                else:
                    if nums[pivot] < nums[left]:
                        right = pivot - 1
                    else:
                        left = pivot + 1
                
        def search(left, right):
            """
            Binary search
            """
            while left <= right:
                pivot = (left + right) // 2
                if nums[pivot] == target:
                    return pivot
                else:
                    if target < nums[pivot]:
                        right = pivot - 1
                    else:
                        left = pivot + 1
            return -1
        
        n = len(nums)
        
        if n == 0:
            return -1
        if n == 1:
            return 0 if nums[0] == target else -1 
        
        rotate_index = find_rotate_index(0, n - 1)
        
        # if target is the smallest element
        if nums[rotate_index] == target:
            return rotate_index
        # if array is not rotated, search in the entire array
        if rotate_index == 0:
            return search(0, n - 1)
        if target < nums[0]:
            # search on the right side
            return search(rotate_index, n - 1)
        # search on the left side
        return search(0, rotate_index)
```

## 总结
-  这样的题目，还是从基础算法着手，然后将基础算法和具体题目结合起来。这题关键就是找到有序的部分，然后应用二分搜索。


------
