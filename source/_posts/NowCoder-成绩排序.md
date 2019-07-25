---
title: NowCoder-成绩排序
tags:
  - NowCoder
  - Algorithm
  - Sort
  - BubbleSort
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-07-26 00:56:21
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/0383714a1bb749499050d2e0610418b1?tpId=40&tqId=21333&tPage=1&rp=1&ru=/ta/kaoyan&qru=/ta/kaoyan/question-ranking)   
查找和排序
题目：输入任意（用户，成绩）序列，可以获得成绩从高到低或从低到高的排列,相同成绩
都按先录入排列在前的规则处理。 
### Examples:
**Input:** 
输入多行，先输入要排序的人的个数，然后输入排序方法0（降序）或者1（升序）再分别输入他们的名字和成绩，以一个空格隔开
**Output:**
按照指定方式输出名字和成绩，名字和成绩之间以一个空格隔开
按先录入排列在前的规则处理。
<!--more-->
示例：
> jack      70
> peter     96
> Tom       70
> smith     67
> 
> 从高到低  成绩
> peter     96
> jack      70
> Tom       70
> smith     67
> 
> 从低到高
> smith     67
> jack      70
> Tom      70
> peter     96 
{% endnote %}

## Solutions
- 冒泡排序是稳定排序, 不可以用快排, 快排不稳定
- 直接调用STL的stable_sort()函数

## C++ Codes
直接调用
```C++
#include<bits/stdc++.h>
using namespace std;
int n, bs, score[500], r[500];
bool cmp(int i,int j){
    return score[i]<score[j];
}
bool cmp1(int i,int j){
    return score[i]>score[j];
}
int main() {
    string name[500];
    int i,j,k;
  while(cin >>n>>bs){
    for(i=0;i<n;i++){
        r[i]=i;
        cin >>name[i]>>score[i];
    }
    if(bs==1)
        stable_sort(r,r+n,cmp);
    else
        stable_sort(r,r+n,cmp1);
    for(i=0;i<n;i++){
        int t = r[i];
        cout << name[t]<<' '<<score[t]<<endl;
    }
  }
return 0;
}
```

手写冒泡排序
```C++
#include<iostream>
#include<string>
#include<vector>
using namespace std;

//升序
void bubbleSort(int *A, vector<string> &name, int n){
    int tmp;
    string stmp;
    for(int i=0;i<n-1;i++){//已经排序好的个数
        bool flag = false;
        for(int j=n-1;j>i;j--){//从最后向前找最小的
            if(A[j]<A[j-1]){
                flag = true;
                tmp = A[j-1];
                A[j-1]=A[j];
                A[j]=tmp;
                stmp = name[j-1];
                name[j-1]=name[j];
                name[j]=stmp;
            }
        }
        if(!flag) break;
    }
}

//降序
void bubbleSort2(int *A, vector<string> &name, int n){
    int tmp;
    string stmp;
    for(int i=0;i<n-1;i++){//已经排序好的个数
        bool flag = false;
        for(int j=n-1;j>i;j--){//从最后向前找最小的
            if(A[j]>A[j-1]){
                flag = true;
                tmp = A[j-1];
                A[j-1]=A[j];
                A[j]=tmp;
                stmp = name[j-1];
                name[j-1]=name[j];
                name[j]=stmp;
            }
        }
        if(!flag) break;
    }
}

int main(){
    vector<string> names(1000);
    int records[1000];
    int n, tag;
    while(cin>>n>>tag){
        for(int i=0;i<n;i++){
            cin>>names[i]>>records[i];
        }

        if(tag)
            bubbleSort(records, names, n);
        else
            bubbleSort2(records, names, n);

        for(int i=0;i<n;i++) 
            cout<<names[i]<<" "<<records[i]<<endl;
    }
}
```

------
