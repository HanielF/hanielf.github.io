---
title: LeetCode-169-Majority Element
tags:
  - LeetCode
  - Math
  - Map
  - Majority
  - Easy
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-05-26 00:55:31
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/majority-element/)   
Given an array of size n, find the majority element. The majority element is the element that appears more than ⌊ n/2 ⌋ times.

You may assume that the array is non-empty and the majority element always exist in the array.

### Examples:
**Input:**[3,2,3]
**Output:**3
**Input:**[2,2,1,1,1,2,2]
**Output:**2

{% endnote %}
<!--more-->

## Solutions
- 找众数嘛，只要记录下来他们每个出现的次数就好了啊
- 用map存放每个数字出现的次数，然后遇到比n/2.0大的就return


## C++ Codes
时间复杂度$$ O(n) $$,用时32ms

```C++
class Solution {
public:
    int majorityElement(vector<int>& nums) {
        map<int, int> mymap;
        int n = nums.size();
        int res = 0;
        for(int i=0;i<n;i++){
            if(mymap.count(nums[i])==1)
                mymap[nums[i]] = mymap[nums[i]]+1;
            else
                mymap[nums[i]] = 1;

            if(mymap[nums[i]]>n/2.0){
                res = nums[i];
                break;
            }

        }
        return res;
    }
};

```

## Python Codes
简单题就不写py版本了

## 总结
- map真好用。。真香
- 数据量特别大的时候，可以用hash_map
- map的查找方式是二分查找，如果是100万条记录，最多也只要20次的string.compare的比较，200万条记录，也只要用21次的比较
- 如果想要在如此大的记录量下，只用一两次的string.compare就找到记录，那就要用hash_map


------
