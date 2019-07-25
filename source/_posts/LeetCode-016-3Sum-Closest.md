---
title: LeetCode-016-3Sum Closest
tags:
  - LeetCode
  - Algorithm
  - Closest
  - 3Sum
  - Medium
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-05-22 13:28:15
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode.com/problems/3sum-closest/)   
Given an array nums of n integers and an integer target, find three integers in nums such that the sum is closest to target. Return the sum of the three integers. You may assume that each input would have exactly one solution.

### Examples:
**Input:**Given array nums = [-1, 2, 1, -4], and target = 1.
**Output:**The sum that is closest to the target is 2. (-1 + 2 + 1 = 2).

{% endnote %}
<!--more-->

## Solutions
- 还是基于上一道题，用三指针，在将向量排序后，第一个指针从前往后移动，后面两个指针一个指向第一个指针的后一个元素，另一个指向最后一个元素。记录和target相差最小的三数之和，每次比较target和当前三数之和，如果三数之和比target大，那就减小第三个指针位置，如果比target小，就增加第二个指针位置，相等就直接返回了。


## C++ Codes

```C++
class Solution {
public:
    int threeSumClosest(vector<int>& nums, int target) {
        sort(nums.begin(),nums.end());
        if(nums.size()<3) return 0;
        
        //初始化为前三个的和与target的差
        int min_diff = nums[0]+nums[1]+nums[2];
        for(int i=0;i<nums.size()-2;i++){
            if(i>0 && nums[i]==nums[i-1]) continue;
            
            int l = i+1, r = nums.size()-1;
            while(l<r){
                //当前三个数和target的差
                int cur_diff = nums[i]+nums[l]+nums[r];
                if(target == cur_diff)return target;
                if(abs(target-cur_diff) < abs(target-min_diff))min_diff = cur_diff;
                if(target > cur_diff)l++;
                if(target < cur_diff)r--;
            }
        }
        return min_diff;
    }
};
```

## Python Codes

```python
class Solution:
    def threeSumClosest(self, nums: List[int], target: int) -> int:
        nums.sort()
        n = len(nums)
        res = float("inf")
        for i in range(n):
            if i > 0 and nums[i] == nums[i-1]:
                continue
            left = i + 1
            right = n - 1
            while left < right :
                #print(left,right)
                cur = nums[i] + nums[left] + nums[right]
                if cur == target:return target
                if abs(res-target) > abs(cur-target):
                    res = cur
                if cur > target:
                    right -= 1
                elif cur < target:
                    left += 1
        return res
```

## 总结
- 基本办法和第15题一样，中间处理有点差别


------
