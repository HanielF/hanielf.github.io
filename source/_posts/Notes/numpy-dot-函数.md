---
title: numpy.dot()函数
comments: true
mathjax: false
date: 2018-09-11 01:08:32
tags: [Python,Numpy,MachineLearning]
categories: Notes
urlname: numpy-function-dot
---

<meta name="referrer" content="no-referrer" />

#Numpy
　　numpy(Numerical Python extensions)是一个第三方的Python包，用于科学计算。本文主要讲学习Numpy过程中遇到的一个问题，关于numpy.dot()是怎运算的。<!--more-->

##Numpy中多维数组的轴
　　多维数组可以用numpy包生成，关于多维数组的运算以及定义等自行百度,主要讲多维数组的轴,这也是我学习时不懂的地方。

{% note default %}
　　多维数组的轴(axis)和该数组的size(或者)shape元素对应 。轴数从0开始，如果是二维，０轴是竖行，１轴是横行。具体看代码
{% endnote %}

```python
x = np.random.randint(0,5,[3,2,2])
print(x)
Out:
  [[[5 2]
    [4 2]]

   [[1 3]
    [2 3]]

   [[1 1]
    [0 1]]]

x.sum(axis=0)
Out:
  array([[7, 6],
        [6, 6]])

x.sum(axis=1)
Out:
  array([[9, 4],
         [3, 6],
         [1, 2]])

x.sum(axis=2)
Out:
  array([[7, 6],
       [4, 5],
       [2, 1]])
```

{% note info %} 
　　如果将三维数组的每一个二维看做一个平面（plane，X[0, :, :], X[1, :, :], X[2, :, :]），三维数组即是这些二维平面层叠（stacked）出来的结果。则（axis=0）表示全部平面上的对应位置，（axis=1），每一个平面的每一列，（axis=2），每一个平面的每一行。
{% endnote %}

#numpy.dot()
　　*numpy.dot(a,b,out=None)*

- 如果a和b都是一维数组，则进行内积运算

```python
np.dot(3, 4)
Out:
  12

np.dot([2j, 3+3j], [2j, 3j])
Out:
  (-13+9j)
```

- 如果都是二维数组，就进行矩阵乘法，推荐`a@b`

```python
a = [[1, 0], [0, 1]]
b = [[4, 1], [2, 2]]
np.dot(a, b)
Out:
  array([[4, 1],
         [2, 2]])
```

- 如果其中一个矩阵０秩，即标量，就进行`a*b`的运算，相乘
- 如果a是N-D矩阵且b是1-D矩阵,就进行a的最后一个轴上的数据和b相乘并求和

```python
  a = array([[[ 1.,  2.,  3.,  4.],
       [ 5.,  6.,  7.,  8.],
       [ 9., 10., 11., 12.]],

      [[ 1.,  2.,  3.,  4.],
       [ 5.,  6.,  7.,  8.],
       [ 9., 10., 11., 12.]]])
  b = np.array([1,2,3,4])
  np.dot(a, b)
  Out:
    array([[ 30.,  70., 110.],
         [ 30.,  70., 110.]])
```
- 如果都是多维矩阵，则_a的最后一个轴和b的倒数第二个轴上的数据乘积累加_,类似第四种情况
  `dot(a, b)[i,j,k,m] = sum(a[i,j,:] * b[k,:,m])`

```python
  a = np.arange(3*4*5*6).reshape((3,4,5,6))
  b = np.arange(3*4*5*6)[::-1].reshape((5,4,6,3))
  np.dot(a, b)[2,3,2,1,2,2]
  Out:
    499128
  sum(a[2,3,2,:] * b[1,2,:,2])
  Out:
    499128
```

&nbsp;&nbsp;

