---
title: LeetCode-039-Combination Sum
tags:
  - LeetCode
  - Algorithm
  - Medium
  - Recursive
  - Backtracking
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-11-18 15:57:44
urlname: LeetCode-039-Combination-Sum
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/combination-sum/)
Given a set of candidate numbers (candidates) (without duplicates) and a target number (target), find all unique combinations in candidates where the candidate numbers sums to target.

The same repeated number may be chosen from candidates unlimited number of times.

### Examples:
**Input:**
> candidates = [2,3,6,7], target = 7
> candidates = [2,3,5], target = 8

**Output:**
> [ [7], [2,2,3] ]
> [ [2,2,2,2], [2,3,3], [3,5] ]

{% endnote %}
<!--more-->

## Solutions

- 刚开始还以为是完全背包问题，后来想想好像不太对，只是有点像背包问题
- 然后就想到可以直接递归做，应该也可以用DP，但是递归加剪枝加回溯就可以了
- 先从不大于target的candidates开始找，然后将这个candidate记下，递归找到所有target-candidate的组合，最后回溯，将之前记下的candidate和所有的组合拼接起来，得到结果，具体看代码

## C++ Codes

```C++
class Solution {
public:
    vector<vector<int> > combinationSum(vector<int>& candidates, int target) {
        vector<vector<int> > res;
        sort(candidates.begin(),candidates.end());

        // 递归边界
        if(candidates[0]>target || target==0) return {}; 

        for(int i=candidates.size()-1;i>=0;i--){
            // 大于target，无法组合，剪枝
            if(candidates[i]>target)continue; 

            // 当前数加入pre数组
            vector<int> pre;
            pre.push_back(candidates[i]);
            if(target-candidates[i]==0) {
                res.push_back(pre);
                continue;
            }

            // 递归找到target-candidates[i]的组合
            // 使用tmpCandidates防止结果重复
            vector<int> tmpCandidates(candidates.begin(),candidates.begin()+i+1);
            vector<vector<int> > last=combinationSum(tmpCandidates,target-candidates[i]);

            // 回溯，拼接pre和last[..]
            for(int j=0;j<last.size();j++){
                vector<int> tmp;
                tmp.insert(tmp.end(),pre.begin(),pre.end());
                tmp.insert(tmp.end(),last[j].begin(),last[j].end());
                res.push_back(tmp);
            }
        }
        return res;
    }
};
```

## 总结

- 使用递归和回溯，尽量进行一些剪枝操作，防止复杂度太高
- 递归问题要注意返回条件，这里是`candidates[0]>target || target==0`，这样就返回`{}`
- `vector`的拼接操作使用`insert`比较方便


------
