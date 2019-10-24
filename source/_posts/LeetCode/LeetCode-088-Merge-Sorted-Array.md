---
title: LeetCode-088-Merge Sorted Array
tags:
  - LeetCode
  - Algorithm
  - Array
  - Merge
  - Easy
categories:
  - LeetCode
urlname: leetcode-merge-sorted-array
comments: true
mathjax: false
date: 2019-06-02 00:40:13
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/merge-sorted-array/)   
Given two sorted integer arrays nums1 and nums2, merge nums2 into nums1 as one sorted array.

**Note:** 

- The number of elements initialized in nums1 and nums2 are m and n respectively.
- You may assume that nums1 has enough space (size that is greater or equal to m + n) to hold additional elements from nums2.

合并两个有序数组，合并到第一个数组里

### Examples:
**Input:**
> nums1 = [1,2,3,0,0,0], m = 3
> nums2 = [2,5,6],       n = 3
**Output:**
> [1,2,2,3,5,6]

{% endnote %}
<!--more-->

## Solutions
- 简单题，所以可以直接暴力求解
- 先把第一个数组备份到tmp，然后对tmp和第二个数组进行合并，结果放到nums1里面就行


## C++ Codes
12ms，超90%，大部分人都是这个时间

```C++
class Solution {
public:
    void merge(vector<int>& nums1, int m, vector<int>& nums2, int n) {
        vector<int> tmp;
        for(int i=0;i<m;i++)
            tmp.push_back(nums1[i]);
        
        int cnt=0, i, j;
        for(i=0,j=0;i<m && j<n;cnt++){
            if(tmp[i]<nums2[j]){
                nums1[cnt]=tmp[i++];
            }else{
                nums1[cnt]=nums2[j++];
            }
        }
        while(i<m) nums1[cnt++]=tmp[i++];
        while(j<n) nums1[cnt++]=nums2[j++];
    }
};
```

## 总结
- 如果是多个链表合并，不是这种有序数组两个合并的，可以看另一个题目，网站搜索合并多个链表


------
