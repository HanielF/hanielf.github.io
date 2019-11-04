---
title: LeetCode-031-Next Permutation
tags:
  - LeetCode
  - Algorithm
  - List
  - Permutation
  - Medium
  - In-Place
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-11-04 09:46:18
urlname: LeetCode-031-Next-Permutation 
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/next-permutation/)   
Implement next permutation, which rearranges numbers into the lexicographically next greater permutation of numbers.

If such arrangement is not possible, it must rearrange it as the lowest possible order (ie, sorted in ascending order).

The replacement must be in-place and use only constant extra memory.

Here are some examples. Inputs are in the left-hand column and its corresponding outputs are in the right-hand column.

### Examples:
**Input:**
> 1,2,3 → 1,3,2
> 3,2,1 → 1,2,3
> 1,1,5 → 1,5,1

{% endnote %}
<!--more-->

## Solutions
1. 暴力方法
  - 直接求出所有可能的排列，然后找下一个比当前序列更大的
  - 很明显不太现实，因为时间复杂度$O(n!)$
1. 正确姿势
  - 首先了解到一个规律：如果序列递减排列，那就已经是最大的了，没有更大的
  - 因为要找字典序更大的，所以应该尽量修改后面部分的数字
  - 如果后面部分的数字，都是递减的，那么只能继续往前找数字
  - 因此：从后往前找到比后一个数字小的数字i，那么就可以通过改变这个数字i，及它后面所有数字的排列找到结果
  - 我们在后面递减的序列中找最接近数字i的数字j，交换两个数字，固定数字i，这时候i后面的数字还是全部递减的，所以，将它们逆序，得到最小的。再加上i就是字典序的下一个排列
  

## C++ Codes

这次用了英文注释，后面慢慢的都用英文写注释了

```C++
class Solution {
public:
    void nextPermutation(vector<int>& nums) {
        if(nums.size()<=1) return;
        
        // Find target i, which is smaller than i+1
        int target=nums.size()-2;
        while(target>=0 && nums[target]>=nums[target+1])
            target--;
        
        // If nums is descending
        if(target<0) return reverse(nums.begin(),nums.end());
        
        // Find the bigger number
        //Pay attention to the boundary 
        int bigger=target+1;
        while(bigger<nums.size() && nums[bigger]>nums[target]){ 
            bigger++;
        }
        bigger--;
        
        //swap the bigger number and target
        int tmp = nums[target];
        nums[target] = nums[bigger];
        nums[bigger]=tmp;
        
        //reverse nums[target+1:N-1]
        reverse(nums.begin()+target+1,nums.end());
        
        return;
    }
};
```

## Python Codes

```python
class Solution:
    def nextPermutation(self, nums: List[int]) -> None:
        """
        Do not return anything, modify nums in-place instead.
        """
        firstIndex = -1
        n = len(nums)
        def reverse(nums, i, j):
            while i < j:
                nums[i],nums[j] = nums[j], nums[i]
                i += 1
                j -= 1
        for i in range(n-2, -1, -1):
            if nums[i] < nums[i+1]:
                firstIndex = i
                break
        #print(firstIndex)
        if firstIndex == -1:
            reverse(nums, 0, n-1)
            return 
        secondIndex = -1
        for i in range(n-1, firstIndex, -1):
            if nums[i] > nums[firstIndex]:
                secondIndex = i
                break
        nums[firstIndex],nums[secondIndex] = nums[secondIndex], nums[firstIndex]
        reverse(nums, firstIndex+1, n-1)
```

## 总结
- 字典序问题，不应该从前往后比较，应该从后往前，越低位的越小
- 通过极端情况，如从整个序列降序和升序的情况，来找思路


------
