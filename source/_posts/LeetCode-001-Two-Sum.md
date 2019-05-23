---
title: LeetCode-001-Two Sum
comments: true
mathjax: false
date: 2019-04-07 23:31:44
tags: [LeetCode, Hash Map, Math]
categories: [LeetCode]
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode.com/problems/two-sum/)   
Given an array of integers, return indices of the two numbers such that they add up to a specific target. You may assume that each input would have exactly one solution. Example: Given nums = [2, 7, 11, 15], target = 9,

Because nums[0] + nums[1] = 2 + 7 = 9, return [0, 1]. UPDATE (2016/2/13): The return format had been changed to zero-based indices. Please read the above updated description carefully.
{% endnote %}
<!--more-->

## Solutions
- 刚开始第一反应就是暴力遍历，但是时间复杂度是$$ O(n^2) $$
- 题解用了hash map，时间复杂度为$$ O(n) $$
  - 第一种方式是两遍hash，第一遍将元素添加进去，第二遍遍历元素
  - 第二种方式一遍hash，一边添加元素一边判断结果是否在已添加的元素中

## C++ Codes

```C++
class Solution {
public:
    vector<int> twoSum(vector<int>& nums, int target) {
        vector<int> res;
        map<int,int> numMap;          //创建map
        map<int ,int>::iterator it;   //创建迭代器

        for(int i=0;i<nums.size();i++){
            it=numMap.find(target-nums[i]);       //find函数查找key
            if(it!=numMap.end()){                 //如果到了end就说明没找到
                res.push_back(i);
                res.push_back(numMap[target-nums[i]]);
                return res;
            }
            numMap.insert(make_pair(nums[i],i));  //插入pair
        }
        return res;
    }
};
```

{% note info %}
## C++中vector用法回忆：
- 创建：$$ vector<int> vec; $$
- 访问：int i=vec[0];
- 插入：vec.push_back(1); vec.insert(index,val)，在第i个元素后面插入
- 删除：
  - vec.pop_back(),删除最后一个
  - vec.erase(index)，删除index位置处元素
  - vec.erase(1,3)，删除[1,3)区间的元素
- 大小：vec.size();
- 清空：vec.clear();
- 翻转：reverse(vec.begin(),vec.end());将元素翻转在vecotr，要#include<algorithm>
- 排序：
  - sort(vec.begin(),vec.end());默认按照升序排列
  - sort(vec.begin(),vec.end(),cmp);定义排序比较函数将序排列
> bool cmp(const int&a,const int&b){
>   return a>b;
> }
- 迭代器访问
> vector<int>::iterator it;
> for(it=vec.begin();it!=vec.end();it++)
>   cout<<\*it<<endl;
{% endnote %}


{% note info %}
## C++中map的用法
- 创建：$$ map<string, int>mapString; $$ key类型为string，val类型为int
- 添加：
  - mapString["hello"]=1; 最常用的最简单的插入方式,**会覆盖之前的数据**
  - mapString.insert(make_pair("hello",1));  次常用,** insert()不会覆盖之前的数据**
  - mapString.insert(pair<string,int>("hello",1));
  - mapString.insert(map<string,int>::value_type("hello",1));
- 迭代：
> map<string,int>::iterator it;
> for ( it = mapString.begin( ); it != mapString.end( ); it++ )
>   cout << " " << it -> second;

- 查找：
  - **不建议!** mapString["hello"];返回"hello"对应的值，如果不存在，则添加一个元素，key为"hello",val为类型默认值,并返回这个默认值
  - mapString.count("hello"); map::count()方法返回被查找元素的个数,只有0或1
  - map::find()方法,返回的是被查找元素的位置，没有则返回map.end()
> map<string,int>::iterator it;
> it = mapString.find("hello");
> if(it==test.end()){
>   cout<<"hello not found"<<endl;
> }

- 删除:
  - mapString.erase("hello");
  - 用迭代器删除,注意在迭代期间是不能被删除的
> map<string,int>::iterator it;
> it = mapString.find("hello");
> 
> if(it==mapString.end())  cout<<"hello not found"<<endl;
> else  mapString.erase(it);

- 排序：map中元素自动按照key升序排序，不可以用sort函数
- 大小：mapString.size();
- 清除：mapString.clear();

{% endnote %}

## Python Codes

```python
class Solution:
    def twoSum(self, nums: List[int], target: int) -> List[int]:
        map = {}
        for i, num in enumerate(nums):
            if target - num in map:
                return [map[target - num], i]
            map[num] = i

        return []
```

{% note info %}
list中遍历的三种方式：
1. for items in list:  根据元素遍历
2. for index in range(len(list)):  根据索引遍历
3. enumerate(seq, [start=0])  创建枚举对象，同时列出数据下标和数据
  - seq: 一个序列、迭代器或其他支持迭代对象, 如列表、元组或字符串
  - start: 下标起始位置
  - 返回：一个列表 [(下标，数据)...], 例如[(1, 'a'), (2, 'b'), (3, 'c')]
4. iterList = iter(list)  创建迭代器遍历
  - 用法：for item in iterList: print(item)
{% endnote %}

## 总结
虽然第一题很简单，但是深入了看还是能学了不少东西的，在遍历这方面，时间复杂度优化可以用map，C++时间复杂度可以从$$ O(n) $$降到$$ O(logn) $$，python和java可以直接降到$$ O(1) $$.

而且打完代码复习了C++中vector的用法，还有pythonlist的enumerate遍历。C++ STL中的map是现学现卖了.

打算每天一题，感觉有点晚了，亡羊补牢把。

--------




