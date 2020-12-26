---
title: Tornado中的异步
comments: true
mathjax: false
date: 2019-03-17 22:32:28
tags: [Python,Tornado, Async]
categories: Notes
urlname: tornado-async
---

<meta name="referrer" content="no-referrer" />


## 概述

因为epoll主要用来解决网络IO的并发问题，所以Tornado中的异步也主要体现在网络的IO异步上，即异步web
<!--more-->

## tornado.httpclient.AsyncHttpClinet
- 是Tornado提供的异步web请求客户端，用来进行异步web请求
- from tornado.httpclient import AsyncHttpClinet

## fetch(request, callback = None)
- 此函数用于执行一个Web请求，并异步响应返回一个tornado.httpclient.HttpResponse
- request可以是一个URL，也可以是一个Tornado.httpclient.HttpResponse对象,如果插入的是url，会自动生成一个request对象

## HTTPRequest
- HTTP请求类，该类的构造函数可以接收参数    
- 参数:
  - url: 字符串类型，要访问的网址，必传
  - method：字符串类型，http请求方式
  - headers：字典或者HTTPHeaders，附加的协议头
  - body: HTTP请求体
  
## HTTPResponse
- HTTP响应类
- 属性
  - code: 状态码
  - reason： 状态码的描述
  - body： 相应的数据
  - error： 是否有异常

## @tornado.web.asynchronous装饰器
- 不关闭通信的通道
- **实际操作发现无法使用这个装饰器**
-------------------

## 示例

### 回调函数实现的异步
**Handler代码**
```
class StudentsHandler(RequestHandler):
    def on_response(self, response):
        if response.error:
            self.send_error(500)
        else:
            data = json.loads(response.body)
            # 这里本身无法write,要打开通道，用asynchronous装饰器
            self.write(data)
        self.finish()

    # 不关闭通信的通道
    #  @tornado.web.asynchronous
    # 实操发现用不了这个装饰器
    def get(self, *args, **kwargs):
        url = "http://s.budejie.com/v2/topic/list/10/0-0/budejie-android-8.0.1/0-25.json?uid=&t=&market=360zhushou&client=android&appname=budejie&device=&jdk=1&ver=8.0.1&udid=&from=android"
        # 创建客户端
        client = AsyncHTTPClient()
        # on_response是回调函数,如果请求成功，就进行on_response回调函数
        client.fetch(url, self.on_response)
        #  self.write("OK") 
```

### 协程实现的异步
**Handler代码**
```
class Students2Handler(RequestHandler):
    @tornado.gen.coroutine
    def get(self, *args, **kwargs):
        url = "http://s.budejie.com/v2/topic/list/10/0-0/budejie-android-8.0.1/0-25.json?uid=&t=&market=360zhushou&client=android&appname=budejie&device=&jdk=1&ver=8.0.1&udid=&from=android"
        client = AsyncHTTPClient()
        # 耗时操作挂起
        res = yield client.fetch(url)
        if res.error:
            self.send_error(500)
        else:
            data = json.loads(res.body)
            self.write(data) 
```
### 协程异步并将异步web请求单独出来

**Handler代码**

    class Students3Handler(RequestHandler):
        # 简化get函数
        @tornado.gen.coroutine
        def get(self, *args, **kwargs):
            res = yield self.getData()
            self.write(res)
    
        # 这里也要加装饰器，这里也是耗时操作
        @tornado.gen.coroutine
        def getData(self):
            url = "http://s.budejie.com/v2/topic/list/10/0-0/budejie-android-8.0.1/0-25.json?uid=&t=&market=360zhushou&client=android&appname=budejie&device=&jdk=1&ver=8.0.1&udid=&from=android"
            client = AsyncHTTPClient()
            # 耗时操作
            res = yield client.fetch(url)
            if res.error:
                #  表示没有结果
                ret = {"ret": 0}
            else:
                ret = json.loads(res.body)
            #  相当于gen.send()函数
            raise tornado.gen.Return(ret)
