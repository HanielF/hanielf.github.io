---
title: LeetCode-036-Valid Sudoku
tags:
  - LeetCode
  - Algorithm
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-11-15 11:34:14
urlname: LeetCode-036-Valid-Sudoku

---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem]()
Determine if a 9x9 Sudoku board is valid. Only the filled cells need to be validated according to the following rules:

- Each row must contain the digits 1-9 without repetition.
- Each column must contain the digits 1-9 without repetition.
- Each of the 9 3x3 sub-boxes of the grid must contain the digits 1-9 without repetition

<center>

!['A partially filled sudoku which is valid.'](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/2soqOa.png)
</center>

The Sudoku board could be partially filled, where empty cells are filled with the character ‘.’.

### Examples:
**Input:**
> [
> [ 5 , 3 ,”.”,”.”, 7 ,”.”,”.”,”.”,”.”],
> [ 6 ,”.”,”.”, 1 , 9 , 5 ,”.”,”.”,”.”],
> [“.”, 9 , 8 ,”.”,”.”,”.”,”.”, 6 ,”.”],
> [ 8 ,”.”,”.”,”.”, 6 ,”.”,”.”,”.”, 3 ],
> [ 4 ,”.”,”.”, 8 ,”.”, 3 ,”.”,”.”, 1 ],
> [ 7 ,”.”,”.”,”.”, 2 ,”.”,”.”,”.”, 6 ],
> [“.”, 6 ,”.”,”.”,”.”,”.”, 2 , 8 ,”.”],
> [“.”,”.”,”.”, 4 , 1 , 9 ,”.”,”.”, 5 ],
> [“.”,”.”,”.”,”.”, 8 ,”.”,”.”, 7 , 9 ]
> ] 

**Output:**
> true

{% endnote %}
<!--more-->

## Solutions
- 简单的方法就是三次遍历，暴力求解
- 优化就是一次遍历，然后使用`map`来记录，确保：
  - 行中没有重复的数字。
  - 列中没有重复的数字。
  - 3 x 3 子数独内没有重复的数字。
- 难点是，如何确定子数独里方块的位置，需要进行坐标变换，参考：
  - 外层循环行i：0-8,内层循环列j：0-8
  - 子数独的行号是：`i/3*3+j/3`
  - 子数独的列号是：`i%3*3+j%3`
- 使用到的技巧就是，坐标变换

<center>
![subgrid](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hQmDNO.png)
</center>


## C++ Codes

```C++
bool isValidSudoku(vector<vector<char>>& board) {
    for(int i=0;i<9;i++){
        map<char,bool> rowMap, colMap, subMap;
        for(int j=0;j<9;j++){
            // 判断行
            if(board[i][j]!='.'){
                if(rowMap[board[i][j]]==true) return false;
                rowMap[board[i][j]]=true;
            }
            //判断列
            if(board[j][i]!='.'){
                if(colMap[board[j][i]]==true) return false;
                colMap[board[j][i]]=true;
            }
            //通过坐标变换，判断子数独
            if(board[i/3*3+j/3][i%3*3+j%3] != '.'){
                if(subMap[board[i/3*3+j/3][i%3*3+j%3]] == true) return false;
                subMap[board[i/3*3+j/3][i%3*3+j%3]] = true;
            }
        }
    }
    return true;
}
```

## 总结
- 在矩阵中的多次迭代遍历，可以想办法通过坐标变换映射到需要的上面。可以自己画个图，把坐标列出来找规律


------
