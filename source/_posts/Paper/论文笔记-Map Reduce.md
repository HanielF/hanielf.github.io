---
title: 论文笔记 | Map Reduce
tags:
  - MapReduce
  - DistributedSystem
  - PaperNotes
  - GFS
categories:
  - Papers
comments: true
mathjax: true
date: 2020-11-25 13:34:18
urlname: map-reduce
---

<meta name="referrer" content="no-referrer" />

{% note info %}

分布式经典论文Map Reduce笔记。论文原文：[MapReduce: Simplified Data Processing on Large Clusters](https://www.cs.amherst.edu/~ccmcgeoch/cs34/papers/p107-dean.pdf)

{% endnote %}

<!--more-->

## 思路

将大的任务分成小的部分，分给不同的机器上，然后得到结果在汇聚到一起。

![RQBpjq](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/RQBpjq.png)

- 数据文件
- 分为M 份
- 选一个主进程，为其他worker进程分配一个map任务或者reduce任务
- map任务从输入数据中取出key/value，使用map函数
- 中间key/value保存在内存
- 中间key/value写入磁盘，分为R 份
- 主进程将磁盘位置告诉reduce 进程
- reduce worker收到master的位置信息后，使用RPC读数据
- reduce worker将数据按照中间 key排序
- reduce worker 迭代排序后的数据，将key和对应中间value集合传给reduce方法
- reduce方法的输出被添加到最后的输出文件
- 所有的reduce结束后，返回给用户程序
- 最后应该是有R个output file，一般不用用户合并，可以直接调用另一个人mapreduce或者直接用于其他应用
- 
- M和R应该远大于worker的数量，能提升动态加载均衡，也能在worker失效时加快恢复

## Map

Map函数，由用户编写，输入键值对集，产生中间键值对集，最后会把由相同键的key/value pair合并，然后传给reduce 函数。

输入数据被保存在节点本地，这样来节省带宽

## Reduce

Reduce，接受中间键值对，一个键可能对应一个value集合。Reduce将所有的值进行merge，形成小的value集合，比如进行mean操作，产生一个value输出。
中间键值对是通过迭代器传给reduce方法的，这样就可以处理比较大的，无法全放进内存的数据。

Map操作和Reduce操作都在不同的的机器上进行，map是将数据分为M个，reduce是将中间key值分为R个
M和R应该比worker多的多，这样可以让每个worker执行多个task，能够提升动态加载的平衡，并且加速恢复fail worker

## 容错

master定期访问worker，没有回应的worker被标记为失效，然后这个worker上的map/reduce任务被重置为初始状态，等待重新分配到其他worker

MapReduce能够应对大规模的故障，在出故障的机器上：
- 已经完成的map任务也要重做，因为map的结果保存在那个机器上，无法访问了。重做的同时，会发送通知给所有的reduce worker，它们还没有读过map输出的，会在新的map worker那读数据
- 已经完成的reduce不需要重做，因为结果是存在全局文件系统

## 答辩PPT

![dhh32t](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/dhh32t.png)

![rxheIS](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/rxheIS.png)

![bNZAk5](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/bNZAk5.png)

![OzsRoX](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/OzsRoX.png)

![WzpFFZ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/WzpFFZ.png)

![MmnA0V](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/MmnA0V.png)

![CH017m](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/CH017m.png)

![gN0g3b](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/gN0g3b.png)

![KNPlHS](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/KNPlHS.png)

![6PTCOf](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/6PTCOf.png)

![xbuKgG](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xbuKgG.png)

![m4D7Vh](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/m4D7Vh.png)

![4itxXM](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/4itxXM.png)

![wR6D77](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wR6D77.png)

![n9mwJS](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/n9mwJS.png)

![2nqvtK](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/2nqvtK.png)

![zH7ORt](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/zH7ORt.png)

![7AikXP](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7AikXP.png)

![3gKh1F](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3gKh1F.png)

![7hBrZA](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7hBrZA.png)

![aXOrs6](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/aXOrs6.png)

![jwgAsc](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/jwgAsc.png)

![hDO5kz](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hDO5kz.png)

![nlvUbH](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/nlvUbH.png)

![fgsSpQ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/fgsSpQ.png)

![4J5m98](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/4J5m98.png)

![S1MvAv](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/S1MvAv.png)

![rDZZTa](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/rDZZTa.png)

