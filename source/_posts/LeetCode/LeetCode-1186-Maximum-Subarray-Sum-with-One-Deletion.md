---
title: LeetCode-1186-Maximum Subarray Sum with One Deletion
tags:
  - LeetCode
  - Algorithm
  - UCAS
  - DP
  - Medium
categories:
  - LeetCode
comments: true
mathjax: false
date: 2020-11-16 10:28:05
urlname: LeetCode-Maximum-Subarray-Sum-with-One-Deletion
---

<meta name="referrer" content="no-referrer" />

{% note info %}

## [Problem](https://leetcode-cn.com/problems/maximum-subarray-sum-with-one-deletion/)

记录下算法作业的其中一题，在LeetCode上过了之后，还整了半个多小时才在OJ上通过，不得不说UCAS的OJ，数据真的太狗了。

Given an array of integers, return the maximum sum for a non-empty subarray (contiguous elements) with at most one element deletion. In other words, you want to choose a subarray and optionally delete one element from it so that there is still at least one element left and the sum of the remaining elements is maximum possible.

**Note** that the subarray needs to be non-empty after deleting one element.

Constraints:

  - 1 <= arr.length <= 10^5
  - -10^4 <= arr[i] <= 10^4

### Examples:

**Input:**

1. arr = [1,-2,0,3]
1. arr = [1,-2,-2,3]
1. arr = [-1,-1,-1,-1]

**Output:**

1. 4
1. 3
1. -1

**Exlanation**:

1. Because we can choose [1, -2, 0, 3] and drop -2, thus the subarray [1, 0, 3] becomes the maximum value.
1. We just choose [3] and it's the maximum sum.
1. The final subarray needs to be non-empty. You can't choose [-1] and delete -1 from it, then get an empty subarray to make the sum equals to 0.

{% endnote %}
<!--more-->

## Solutions

- 这个是最大子序和的升级版，可以对子数组进行删除一个数
- 很明显，如果那个数是负数，那么删除就可以让数组和更大
- 使用$dp[i]$表示第$i$个数的时候，可以删除一个元素的最大子序和，使用$sum[i]$表示不可以删除元素的最大子序和。考虑第$i$个数，有两种情况
  - 删：`dp[i]=sum[i-1]`
  - 不删：`dp[i]=dp[i-1]+v[i]`
- 因此可以将问题转化为多步决策问题，每一步决策考虑一个数删还是不删，可以得到如下的最优子结构：$dp[i]=max(sum[i-1], dp[i-1]+v[i]$。
- 需要注意的是，题目要求最少一个元素

**遇到的坑点:**

1. 最后一步要比较一下`max_remove_one`和`max_sum`的值，取最大
2. 如果数组全负，不能输出0，要输出最大的负数，因此要用一个变量保存最大值
3. 空间复杂度必须要是`O(1)`，就是说不能开数组（对比之下，LeetCode直接开vector真的是太宽容了）
4. 数据都要使用`long long`
5. 数据会卡`long long`的下界，因此在保存数组最大值的时候，初始化要是`long long`的最小值，这个可以通过打印`LLONG_MIN`这个查看


## C++ Codes

```C++
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <cmath>

using namespace std;

#define ll long long

int main()
{
    //这里会卡long long下界，设置成longlong最小值
    ll w=0, del_sum,no_del_sum;
    scanf("%lld", &w);
    ll max_sum = w, max_remove_one = w, remove_one_sum = w;
    ll max_data = -9223372036854775808;
    max_data = max(max_data, w);

    while (scanf("%lld", &w) != EOF)
    {
        // max_data保存数组最大值，如果数组全负，则返回最大值
        max_data = max(max_data, w);

        // 最多删除一个元素的dp
        no_del_sum = remove_one_sum + w;
        del_sum = max_sum;

        remove_one_sum = max(del_sum, no_del_sum);
        max_remove_one = max(remove_one_sum, max_remove_one);

        // 不删除元素的dp
        max_sum = max(max_sum + w, w);
    }
    if(max_data<0)
        printf("%lld", max_data);
    else
        printf("%lld", max(max_remove_one, max_sum));
    return 0;
}
```

## 总结

- 要注意特殊情况，如这里的数据全负
- 自己手动模拟下运算过程，找最优子结构
- 观察问题，能分解，有最优子结构，就直接用DP，从最小的实例入手，手动模拟

------
