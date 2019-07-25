---
title: LeetCode-124-Binary Tree Maximum Path Sum
tags:
  - LeetCode
  - Algorithm
  - Hard
  - BinaryTree
  - Recursive
  - DP
  - Hard
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-06-13 23:41:23
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## 前言
落了好多题没记录了，而且因为刷的是专项，没有按照顺序了，所以....顺序很乱，这阵子就先这样把，后面夏令营结束再按照顺序来，把以前的慢慢补上。

LeetCode中国还是挺方便的，加载速度比原版的快很多，而且可以中英文切换。

## [Problem](https://leetcode-cn.com/problems/binary-tree-maximum-path-sum/submissions/)   
Given a non-empty binary tree, find the maximum path sum.

For this problem, a path is defined as any sequence of nodes from some starting node to any node in the tree along the parent-child connections. The path must contain at least one node and does not need to go through the root.

求二叉树最大的路径和，任意节点到任意节点的最大路径和

### Examples:
**Input:**[1,2,3]
**Output:**6
**Input:**[-10,9,20,null,null,15,7]
**Output:**42

{% endnote %}
<!--more-->

## Solutions
- 二叉树一般就是递归，这道题也很明显用递归，找到左子树、右子树的最大路径和，一共算了下，大概有一下几种情况

1. 左子树路径和加根节点值最大
2. 右子树路径和加根节点值最大
3. 左右子树路径和都是负数，所以单独的根节点值最大
4. 左右子树路径和加上根节点值最大，$$ 如果左右子树路径和都非负，那肯定就得加上嘛 $$

这里之所以不把左子树路径和与右子树路径和单独列出来，是因为递归到左右子树的时候就相当于判断过来

按照这样的思路，就可以想到办法了，求左右子树路径和，然后找到最大的一边，加上根节点值，这是前三种情况，再和第四种情况：三者之和 比较，就可以得到最后的结果resSum，然后和全局遍历maxSum比较：$$ maxSum = max(maxSum, resSum) $$

- 改进的方法是，判断左右子树的路径和是否小于0，如果小于0，那肯定就不走这边，所以令其为0，最后就变成了一种情况，就是上面的第四种，只要把三者相加，再和全局变量maxSum比较就可以

解释下，如果某一边大于0，那肯定要加上，如果小于0，不走那边就相当于加上0，就可以

但是最后的结果发现，虽然第二种办法，将四种情况缩成了一种情况，但是似乎时间会花的更多。第一种情况是32ms，第二种是40ms+。


## C++ Codes
递归，第一种解法，判断四种情况的，32ms

```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int maxSum;
    int maxPathSum(TreeNode* root) {
        maxSum = root->val;
        int rootGain = maxGain(root);
        return maxSum;
    }
    int maxGain(TreeNode *root){
        if(root==NULL) return 0;
        int leftGain = maxGain(root->left);
        int rightGain = maxGain(root->right);
        int rootGain = max(leftGain, rightGain)+root->val;
        int newPathGain = leftGain + rightGain + root->val;
        int resGain = max(newPathGain, rootGain);
        if(resGain > maxSum)
            maxSum = resGain;
        if(rootGain<=0) return 0;
        return rootGain;
    }
};
```

## C++ Codes
第二种解法，合并成一种情况的，44ms

```C++
/**
 * Definition for a binary tree node.
 * struct TreeNode {
 *     int val;
 *     TreeNode *left;
 *     TreeNode *right;
 *     TreeNode(int x) : val(x), left(NULL), right(NULL) {}
 * };
 */
class Solution {
public:
    int maxSum;
    int maxPathSum(TreeNode* root) {
        maxSum = root->val;
        int rootGain = maxGain(root);
        return maxSum;
    }
    int maxGain(TreeNode *root){
        if(root==NULL) return 0;
        int leftGain = max(maxGain(root->left), 0);
        int rightGain = max(maxGain(root->right), 0);
        int newPathGain = leftGain + rightGain + root->val;
        maxSum = max(newPathGain, maxSum);
        return max(leftGain, rightGain)+root->val;
    }
};
```


## 总结
- 二叉树一般用递归就好了，查找还是啥操作的，一般递归+剪枝就可以，往这方向想 


------
