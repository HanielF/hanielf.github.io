---
title: cannot start gnome-tweaks
comments: true
mathjax: false
date: 2019-03-20 20:36:47
tags: [Daily,Linux]
categories: Daily
---

<meta name="referrer" content="no-referrer" />

今天在用Gnome-tweaks的时候发现打不开了???很莫名其妙,怀疑是滚动更新滚炸了。bing了一下在stackoverflow上找到了解决办法。
<!--more-->

# 无法打开gnome-tweaks

## 报错信息
```
 [sudo] password for root: 
 Traceback (most recent call last):
   File "/usr/bin/gnome-tweaks", line 13, in <module>
     import gi
 ModuleNotFoundError: No module named 'gi'
```

## 解决办法

参照了[stackoverflow](https://stackoverflow.com/questions/32640083/gnome-terminal-not-starting-due-to-error-in-python-script-related-to-gi)

**步骤如下**
- 在命令行进入python2、python3测试import gi，看有没有gi这个包
- 发现只有python3.7下面有这个包
- $ sudo vim /usr/bin/gnome-tweaks
- 编辑第一行，把`#!/usr/bin/env python`改成`#!/usr/bin/python`
- 我这里`python->python3.7`,如果不是的话要写成`!/usr/bin/python3.7`
- 重新运行gnome-tweaks
- 注意不要带sudo,我这里sudo出现了如下报错
```
No protocol specified
Unable to init server: Could not connect: Connection refused
```

---------
OK，问题解决
