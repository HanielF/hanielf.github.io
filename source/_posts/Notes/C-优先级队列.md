---
title: C++优先级队列
comments: true
mathjax: false
date: 2019-05-23 13:42:24
tags: [C++, Queue, PriorityQueue, Notes]
categories: Notes
urlname: C-PriorityQueue
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## 前言
写LeetCode的时候学了优先级队列，这里总结一下STL中优先级队列的使用和实现
{% endnote %}

<!--more-->

## 介绍

优先级队列也是队列的一种，FIFO结构，但是和普通的队列不同的是有一个优先级的权重

优先级队列有两种，一种是最小优先队列，值小的优先级越大，另一种是最大优先队列，值大的优先级越大

头文件： "queue.h", "functional.h"

插入和删除操作复杂度都是$$ O(lgn) $$

## 使用

### 构造函数
构造函数声明如下：
```
std::priority_queue<T> pq;
std::priority_queue<T, std::vector<T>, cmp> pq;
```

**第一种构造函数：**
- 传入一个基本类型或者自定义类，自定义类要重载$$ < $$符号

**第二种构造函数：**
参数介绍：
1. 是队列中元素的种类，可以是自定义的也可以是基本类型，默认是int
2. 第二个是容纳优先级队列的容器，只需要知道默认是vector就好，使用的时候也是用vector
3. 这个是最重要的参数，支持一个比较函数，默认是less，队列是最大优先队列。

第三个参数有三种情况介绍：
1. 自定义比较结构，使用结构体，**注意返回值！！**，最小优先队列是大于
2. 使用默认的类型时用less()或者greater()
3. 使用自定义类的时候重载$$ < $$符号

### 常用操作
1. q.empty(): 如果队列为空，则返回true，否则返回false
2. q.size():  返回队列中元素的个数
3. q.pop():   删除队首元素，但不返回其值
4. q.top():   返回具有最高优先级的元素值，最大优先队列找最大的元素，最小优先队列找最小的，但不删除该元素
5. q.push(item): 在基于优先级的适当位置插入新元素

### 使用样例
注意这不是一个完整的C++代码

```C
//包含头文件并使用std命名空间
#include<functional>
#include<queue>
#include<vector>
using namespace std;

//第一种构造函数，采用默认优先级构造队列
priority_queue<int>que;

//第二种构造函数，这里使用基本的int类型，如果看其他类型，请看LeetCode 23题代码
//自定义比较结构
struct cmp1{
    bool operator ()(int &a,int &b){
        return a>b;//最小值优先
    }
};
priority_queue<int,vector<int>,cmp1>que1;//最小值优先
priority_queue<int,vector<int>,greater<int> >que3;//注意“>>”会被认为错误，greater在functional头文件中


//自定义数据结构并重载小于操作符
struct number1{
    int x;
    bool operator < (const number1 &a) const {
        return x>a.x;//最小值优先
    }
};
priority_queue<number1>que5; //最小优先级队列

printf("采用默认优先关系:/n(priority_queue<int>que;)/n");
printf("Queue 0:/n");
while(!que.empty()){
    printf("%3d",que.top());
    que.pop();
}

```

## 实现
引用博客：[优先队列原理与实现](https://www.cnblogs.com/luoxn28/p/5616101.html)
推荐结合另一篇博客: [【STL学习】优先级队列Priority Queue详解与C++编程实现](https://blog.csdn.net/xiajun07061225/article/details/8556786)
可以看这两篇博客了解插入和删除的原理，第一篇博客有图比较清楚

```
package priorityheap;

import java.util.Arrays;

/**
 * 优先队列类（最大优先队列）
 */
public class PriorityHeap {

    // ------------------------------ Instance Variables

    private int[] arr;
    private int size;

    // ------------------------------ Constructors

    /**
     * 优先队列数组默认大小为64
     */
    public PriorityHeap() {
        this(64);
    }

    public PriorityHeap(int initSize) {
        if (initSize <= 0) {
            initSize = 64;
        }
        this.arr = new int[initSize];
        this.size = 0;
    }

    // ------------------------------ Public methods

    public int max() {
        return this.arr[0];
    }

    public int maxAndRemove() {
        int t = max();

        this.arr[0] = this.arr[--size];
        sink(0, this.arr[0]);
        return t;
    }
    public void add(int data) {
        resize(1);
        this.arr[size++] = data;
        pop(size - 1, data);
    }

    // ------------------------------ Private methods

    /**
     * key下沉方法
     */
    private void sink(int i, int key) {
        while (2 * i <= this.size - 1) {
            int child = 2 * i;
            if (child < this.size - 1 && this.arr[child] < this.arr[child + 1]) {
                child++;
            }
            if (this.arr[i] >= this.arr[child]) {
                break;
            }

            swap(i, child);
            i = child;
        }
    }

    /**
     * key上浮方法
     */
    private void pop(int i, int key) {
        while (i > 0) {
            int parent = i / 2;
            if (this.arr[i] <= this.arr[parent]) {
                break;
            }
            swap(i, parent);
            i = parent;
        }
    }

    /**
     * 重新调整数组大小
     */
    private void resize(int increaseSize) {
        if ((this.size + increaseSize) > this.arr.length) {
            int newSize = (this.size + increaseSize) > 2 * this.arr.length ? (this.size + increaseSize) : 2 * this.arr.length;
            int[] t = this.arr;

            this.arr = Arrays.copyOf(t, newSize);
        }
    }

    /**
     * Swaps arr[a] with arr[b].
     */
    private void swap(int a, int b) {
        int t = this.arr[a];
        this.arr[a] = this.arr[b];
        this.arr[b] = t;
    }
}
```

---------
