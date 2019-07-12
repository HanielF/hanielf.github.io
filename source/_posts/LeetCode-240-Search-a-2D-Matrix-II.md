---
title: LeetCode-240-Search a 2D Matrix II
tags:
  - LeetCode
  - Matrix
  - Search
  - Medium
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-06-02 00:29:32
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/search-a-2d-matrix-ii/)   
Write an efficient algorithm that searches for a value in an m x n matrix. This matrix has the following properties:
- Integers in each row are sorted in ascending from left to right.
- Integers in each column are sorted in ascending from top to bottom.

简单点说就是二维数组，从左向右递增，从上向下递增，然后查找输入的数组，要求高效

### Examples:
**Input:**
> [
>   [1,   4,  7, 11, 15],
>   [2,   5,  8, 12, 19],
>   [3,   6,  9, 16, 22],
>   [10, 13, 14, 17, 24],
>   [18, 21, 23, 26, 30]
> ]
**Output:**
> Given target = 5, return true.
> Given target = 20, return false.

{% endnote %}
<!--more-->

## Solutions
- 如果是$$ O(n) $$可能不行，毕竟要高效的算法，这里从数组的结构入手
- 以右上角为起点进行查找，如果比它大，就向下，如果比它小，就向左，一直找到边界位置


## C++ Codes
用时108ms

```C++
class Solution {
public:
    bool searchMatrix(vector<vector<int>>& matrix, int target) {
        if(matrix.size()==0 || matrix[0].size()==0)return false;
        int col = matrix[0].size()-1;
        int row = 0;
        int n = matrix.size();
        //边界条件就是col>=0 && row<n
        while(col>=0 && row<n){
            if(matrix[row][col]>target)
                col--;
            else if(matrix[row][col]<target)
                row++;
            else return true;
        }
        return false;
    }
};
```


## 总结
- 对于这种排序好结构很明显的，尽量从结构上入手，找特点，而不是暴力解


------
