---
title: LeetCode-015-3Sum
tags:
  - LeetCode
  - Algorithm
  - Medium
categories:
  - LeetCode
urlname: leetcode-3sum
comments: true
mathjax: false
date: 2019-05-19 21:22:18
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## 前言
快半月没更新了，也没怎么刷题，家里有事情耽误了，后面一周又忙着别的，以后简单难度的题就直接略过了，加快进度，么得时间咯

## [Problem](https://leetcode.com/problems/3sum/)   
Given an array nums of n integers, are there elements a, b, c in nums such that a + b + c = 0? Find all unique triplets in the array which gives the sum of zero.

**Note:**  
The solution set must not contain duplicate triplets.

### Examples:
**Input:**nums = [-1, 0, 1, 2, -1, -4]
**Output:**[ [-1, 0, 1], [-1, -1, 2] ]

{% endnote %}
<!--more-->

## Solutions
- 暴力法，但是想想不优化的花$$ n^3 $$的复杂度还是算了，就算优化估计也难过
- 想了递归的方法，先固定一个数，问题变成找两个数和为第一个数的负数，然后再固定一个数，找最后一个，但是相当于也是$$ n^3 $$，想了用类似dp那样的优化，似乎也不行，数字只能用一次
- 最后的方法是，先选第一个数，然后找剩下的连个数，让三个数和为0，剩下的两个数找的时候优化一下，排序后一个从前一个从后找，具体看代码


## C++ Codes

```C++
class Solution {
public:
    vector<vector<int>> threeSum(vector<int>& nums) {
        //先排序预处理，如果发现全都是大于0或者全小于0，或者少于三个元素，就返回
        sort(nums.begin(),nums.end());
        if(nums.size()<3||nums.front()>0||nums.back()<0) return {};

        vector<vector<int>> res;
        int len = nums.size();

        for(int i=0;i<len;i++){
            //很明显只需要找第一个数小于0的
            if(nums[i]>0)break;
            //这里记得要写i>0，边界判断，容易丢
            if(i>0 && nums[i]==nums[i-1]) continue;

            //left自增，right自减
            int left = i+1, right = len-1;
            while(left<right){
                int tmp = nums[i]+nums[left]+nums[right];
                if(tmp==0){
                    res.push_back(vector<int>{nums[i], nums[left], nums[right]});
                    
                    //如果出现数字相同跳过，注意边界条件：left < right
                    while (left < right && nums[left] == nums[left + 1]) left += 1;
                    while (left < right && nums[right] == nums[right - 1]) right -= 1;

                    left++;
                    right--;
                }
                //很好理解，大于0说明正数太大，小于0说明负数太小
                else if(tmp>0) right--;
                else if(tmp<0) left++;
            }
        }
        return res;
    }
};
```

## Python Codes

```python
class Solution:
    def threeSum(self, nums: List[int]) -> List[List[int]]:
        nums.sort()
        n = len(nums)
        res = []

        for i in range(n):
            if res[i] > 0
                break
            if i > 0 and nums[i] == nums[i-1]:
                continue

            left = i + 1
            right = n - 1
            while left < right:
                cur_sum = nums[i] + nums[left] + nums[right]
                if cur_sum == 0:
                    tmp = [nums[i],nums[left],nums[right]]
                    res.append(tmp)
                    while left < right and nums[left] == nums[left+1]:
                        left += 1
                    while left < right and nums[right] == nums[right-1]:
                        right -= 1
                    left += 1
                    right -= 1
                elif cur_sum > 0:
                    right -= 1
                else:
                    left += 1
        return res
```

## 总结
- 找符合条件的数的时候，双指针前后开始找很好用
- 边界条件判断很重要，条件语句要多想想，容易漏


------
