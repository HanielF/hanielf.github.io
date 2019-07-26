---
title: LeetCode-018-4Sum
tags:
  - LeetCode
  - Algorithm
  - Math
  - TwoPointer
  - Medium
categories:
  - LeetCode
urlname: leetcode-4sum
comments: true
mathjax: false
date: 2019-07-17 15:00:41
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/4sum/)  
Given an array nums of n integers and an integer target, are there elements a, b, c, and d in nums such that a + b + c + d = target? Find all unique quadruplets in the array which gives the sum of target.

**Note:**
The solution set must not contain duplicate quadruplets.

### Examples:
**Input:** nums = [1, 0, -1, 0, -2, 2], and target = 0
**Output:**
A solution set is:
> [
> [-1, 0, 0, 1],
> [-2, -1, 1, 2],
> [-2, 0, 0, 2]
> ]

{% endnote %}
<!--more-->

## Solutions
-
和[第15题3Sum](https://catchdream.me/2019/05/19/LeetCode-015-3Sum/)思路类似，十五题是固定一个数字，然后双指针求三数之和，这题固定两个数字，然后双指针求四数之和。其实原理和3Sum一样


## C++ Codes

```C++
class Solution {
public:
    set<vector<int> >res;
    vector<vector<int>> fourSum(vector<int>& nums, int target) {
        if(nums.size()<4)return {};
        sort(nums.begin(), nums.end());
        for(int i=0;i<nums.size()-3;i++){
            for(int j=i+1;j<nums.size()-2;j++){
                int l = j+1;
                int r = nums.size()-1;
                while(l<r){
                    int tmp = nums[i]+nums[j]+nums[l]+nums[r];
                    if(tmp==target){
                        res.insert({nums[i], nums[j], nums[l], nums[r]});
                        r--;
                        l++;
                    }
                    else if(tmp>target) r--;
                    else if(tmp<target) l++;
                }
            }
        }
        return vector<vector<int> >(res.begin(), res.end());
    }
};
```

## Python Codes
为啥我总是不想用python再写一遍，感觉很没意思…
附上题解里找到[Python版本代码](https://leetcode-cn.com/problems/4sum/solution/gu-ding-liang-ge-shu-yong-shuang-zhi-zhen-zhao-lin/)

```python
class Solution:
    def fourSum(self, nums: List[int], target: int) -> List[List[int]]:
        n = len(nums)
        if n < 4: return []
        nums.sort()
        res = []
        for i in range(n-3):
            # 防止重复 数组进入 res
            if i > 0 and nums[i] == nums[i-1]:
                continue
            # 当数组最小值和都大于target 跳出
            if nums[i] + nums[i+1] + nums[i+2] + nums[i+3] > target:
                break
            # 当数组最大值和都小于target,说明i这个数还是太小,遍历下一个
            if nums[i] + nums[n-1] + nums[n-2] + nums[n-3] < target:
                continue
            for j in range(i+1,n-2):
                # 防止重复 数组进入 res
                if j - i > 1 and nums[j] == nums[j-1]:
                    continue
                # 同理
                if nums[i] + nums[j] + nums[j+1] + nums[j+2] > target:
                    break
                # 同理
                if nums[i] + nums[j] + nums[n-1] + nums[n-2] < target:
                    continue
                # 双指针
                left = j + 1
                right = n - 1
                while left < right:
                    tmp = nums[i] + nums[j] + nums[left] + nums[right]
                    if tmp == target:
                        res.append([nums[i],nums[j],nums[left],nums[right]])
                        while left < right and nums[left] == nums[left+1]:
                            left += 1
                        while left < right and nums[right] == nums[right-1]:
                            right -= 1
                        left += 1
                        right -= 1
                    elif tmp > target:
                        right -= 1
                    else:
                        left += 1
        return res
```

## 总结
- 双指针用法之一：找目标数字，四个数可以固定两个数，另外两个数双指针求
- 差不多类型的题，，，要学会套,举一反三不成,举一反一可以把


------
