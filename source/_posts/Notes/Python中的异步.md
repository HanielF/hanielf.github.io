---
title: Python中的异步
comments: true
mathjax: false
date: 2019-03-17 20:28:26
tags: [Python, Async]
categories: Notes
urlname: python-async
---

<meta name="referrer" content="no-referrer" />

学Tornado的异步之前学了下Python的同步和异步机制，下面是三种实现Python中同步的方法。主要是用yield和装饰器以及threading模块实现。

<!--more-->
# python中异步的理解

## 同步

> 按部就班的依次执行
> 如果在请求中添加一个耗时操作，则必须等耗时操作结束才继续下去
> 一般不会用同步

## 异步

### 概述

对于耗时的操作，一般会交给另一个线程处理，我们继续向下执行，当别人结束耗时操作后再将结果返回给我们

### 回调函数实现异步

```python
import time
import threading

# Tornado中不用我们写这个函数
# handler获取数据（数据库、其他服务器、循环耗时）
def longIo(callback):
  def run(cb):
    print("开始耗时操作")
    time.sleep(5)
    print("结束耗时操作")
    #耗时操作结束执行回调函数
    cb(" 我是返回的数据")
  #创建一个线程，处理耗时操作threading传参args=()
  threading.Thread(target=run,args=(callback,)).start()

# 函数（回调函数）
def finish(data):
  print("开始处理回调函数")
  print("接收到longIo的相应数据:",data)
  print("结束处理回调函数")

# 一个客户的请求
def reqA():
  print("开始处理reqA")
  longIo(finish)
  print("结束处理reqA")

def reqB():
  print("开始处理reqB")
  longIo(finish)
  print("结束处理reqB")

#Tornado服务
def main():
  reqA()
  reqB()
  while 1:
    time.sleep(0.1)
    pass

if __name__=='__main__':
  main()

```

### 协程实现异步

#### 版本1
```python
import time

# 全局变量生成器
gen = None

def longIo(callback):
  def run(cb):
    print("开始耗时操作")
    time.sleep(5)
    try:
      引进全局变量并且用生成器回发数据
      global gen
      gen.send("我是返回的数据")
    except StopIteration as e:
      pass
    print("结束耗时操作")
  #创建一个线程，处理耗时操作,不会影响A和B
  threading.Thread(target=run).start()

# 一个客户的请求
def reqA():
  print("开始处理reqA")
  # 接受返回结果
  # 此处相当于挂起,执行longIo,不影响执行reqB
  res = yield longIo()
  print("接收longIo的相应数据:",res)
  print("结束处理reqA")

def reqB():
  print("开始处理reqB")
  time.sleep(2)
  print("结束处理reqB")

#Tornado服务
def main():
  # 创建一个reqA的生成器
  global gen
  gen = reqA()
  # 在这里真正执行reqA
  next(gen)

  reqB()
  while 1:
    time.sleep(0.1)
    pass

if __name__=='__main__':
  main()

```

#### 版本2
**问题**    
> 版本1中调用reqA的时候不能将其视为一个简单的函数，而是要作为生成器来对待
> 很明显要在主函数中要用三行调用reqA，只要用一行调用reqB。

**解决办法**
> 给reqA添加一个装饰器

```python
import time

# 全局变量生成器
gen

def longIo(callback):
  def run(cb):
    print("开始耗时操作")
    time.sleep(5)
    try:
      引进全局变量并且用生成器回发数据
      global gen
      gen.send("我是返回的数据")
    except StopIteration as e:
      pass
    print("结束耗时操作")
  #创建一个线程，处理耗时操作,不会影响A和B
  threading.Thread(target=run).start()

# 装饰器
def genCoroutine(func):
  def wrapper(*args, **kwargs):
    global gen
    gen = func(*args, **kwargs)
    next(gen)
  # 返回内部函数的时候不可以加括号，这里不可以加括号
  return wrapper
  

# 一个客户的请求
@genCoroutine
def reqA():
  print("开始处理reqA")
  # 接受返回结果
  # 此处相当于挂起,执行longIo,不影响执行reqB
  res = yield longIo()
  print("接收longIo的相应数据:",res)
  print("结束处理reqA")

def reqB():
  print("开始处理reqB")
  time.sleep(2)
  print("结束处理reqB")

#Tornado服务
def main():
  reqA()
  reqB()
  while 1:
    time.sleep(0.1)
    pass

if __name__=='__main__':
  main()
```

#### 版本3
**问题**
> 版本2中存在一个全局变量gen，需要消除
**解决办法**

```python
import time

# 装饰器
def genCoroutine(func):
  def wrapper(*args, **kwargs):
    #reqA的生成器
    gen1 = func()
    #longIo的生成器
    gen2 = next(gen1)
    def run(g):
      #gen2,即longIo的返回数据
      res = next(g) 
      try:
        gen1.send(res)#返回给reqA数据
      except StopIteration as e:
        pass
    threading.Thread(target=run,args=(gen2,)).start()

  # 返回内部函数的时候不可以加括号，这里不可以加括号
  return wrapper
  
#这次在这里不管线程了
def longIo(callback):
    print("开始耗时操作")
    time.sleep(5)
    print("结束耗时操作")
    #要返回数据，挂起
    yield "我是返回的数据"

# 一个客户的请求
@genCoroutine
def reqA():
  print("开始处理reqA")
  # 接受返回结果
  # 此处相当于挂起,执行longIo,不影响执行reqB
  res = yield longIo()
  print("接收longIo的相应数据:",res)
  print("结束处理reqA")

def reqB():
  print("开始处理reqB")
  time.sleep(2)
  print("结束处理reqB")

#Tornado服务
def main():
  reqA()
  reqB()
  while 1:
    time.sleep(0.1)
    pass

if __name__=='__main__':
  main()
```
-----------------------------
