---
title: LeetCode-004-Median of Two Sorted Arrays
tags:
  - LeetCode
  - Algorithm
  - Math
  - Binary Search
  - Array
  - Hard
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-04-14 01:48:43
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode.com/problems/median-of-two-sorted-arrays/)   
There are two sorted arrays nums1 and nums2 of size m and n respectively.
Find the median of the two sorted arrays. The overall run time complexity should be O(log (m+n)).
You may assume nums1 and nums2 cannot be both empty
### Examples:
> a = [1, 3], b = [2] ---> 2.0
> a = [1, 2], b = [3, 4] ---> 2.5
{% endnote %}
<!--more-->

## Solutions
- 刚开始是一点思路都么得...如果没有O(log (m+n))的限制，还能用遍历的方法找到，但是加了log，应该是要用二叉的
- 题解里面给出了将两个数组，以i，j为分界，分为两部分:[0,i-1], [i,m-1], [0, j-1], [j,n-1]，使得左边的[0,i-1], [0,j-1] 全都小于右边的[i,m-1], [j,n-1]，利用中位数的意义
- 前提要**确保m<=n**，这很重要！刚开始就弄反了导致一直找不到bug..
- 同时配合二叉搜索，这时搜索的条件就变成了：如果左边的有大于右边的数，就缩小，如果右边的有小于左边的数，就扩大
- 这里需要判断几个边界条件：i=0, j=0, i=m, j=n这四种


## C++ Codes
用时是36ms, 内存9.6MB

```C++
class Solution {
public:
    double findMedianSortedArrays(vector<int>& nums1, vector<int>& nums2) {
        int m=nums1.size();
        int n=nums2.size();
        if(m>n){    //确保m<=n,不然会造成j可能是负数
            nums1.swap(nums2);
            int tmp=m; m=n; n=tmp;
        }
        
        int imin=0,imax=m,half=(m+n+1)/2;
        while(imin<=imax){
            int i=(imin+imax)/2;
            int j=half - i;
            int maxLeft = 0, minRight=0;
            
            if(i<imax && nums2[j-1]>nums1[i]){  //i太小
                imin = i+1;
            } else if(i>imin && nums1[i-1]>nums2[j]){   //i太大
                imax=i-1;
            } else{
                //maxleft = max(nums1[i-1],nums2[j-1]), i, j可能是0, i-1就可能为-1
                if(i==0){ maxLeft=nums2[j-1];}
                else if(j==0){ maxLeft = nums1[i-1];}
                else { maxLeft = nums1[i-1]>nums2[j-1]?nums1[i-1]:nums2[j-1];}
                
                //如果是奇数个
                if((m+n)%2){ return maxLeft;}
                
                //maxRight = max(nums1[i], nums2[j]), i, j可能是m和n，nums1[i]就会越界
                if(i==m){ minRight=nums2[j];}
                else if(j==n){ minRight = nums1[i];}
                else { minRight=nums1[i]<nums2[j]?nums1[i]:nums2[j];}
                
                return (minRight+maxLeft)/2.0;
            }
        }
        return 0;
    }
};

```

## Python Codes

```python
class Solution:
    def findMedianSortedArrays(self, nums1: List[int], nums2: List[int]) -> float:
        m = len(nums1)
        n = len(nums2)
        if(m>n):
            nums1,nums2,m,n = nums2,nums1,n,m
    
        # 整除
        imin, imax, half = 0, m, (m+n+1)//2
        while imin<=imax:
            i = (imin+imax)//2;
            j = half - i;
            maxLeft = 0
            minRight = 0
            print(i,j)
            if i<m and nums2[j-1]>nums1[i]:
                imin = i+1
            elif i>0 and nums1[i-1]>nums2[j]:
                imax = i-1
            else :
                if i==0: maxLeft = nums2[j-1]
                elif j==0: maxLeft = nums1[i-1]
                else:
                    maxLeft = max(nums2[j-1],nums1[i-1])
                
                if (m+n)%2==1: return maxLeft
                
                if i==m: minRight = nums2[j]
                elif j==n: minRight = nums1[i]
                else :
                    minRight = min(nums1[i],nums2[j])
                    
                return (minRight+maxLeft)/2
```

## 总结
- 对用到中位数的题目可以想想中位数的意义，搜索那个分界点，可以直接遍历分界点也可以二叉找，看时间复杂度。
- 要注意前提是m<=n，不然会出错
- 注意搜索时候变化条件，还有几个边界情况注意判断
- while循环的条件这里是imin<=imax，等于的时候就是到叶子节点了
- python这里居然只能用双斜线整除

------
