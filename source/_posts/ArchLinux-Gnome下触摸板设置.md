---
title: ArchLinux-Gnome下触摸板设置
tags:
  - Arch
  - Linux
  - Gnome
  - Synaptics
categories:
  - Linux
urlname: ArchLinux-Gnome-Synaptics 
comments: true
mathjax: false
date: 2019-07-22 23:58:57
---

<meta name="referrer" content="no-referrer" />

{% note info %}
# 前言
Arch下触摸板默认是没有模拟鼠标单击双击功能的, 通过设置可以修改,具体参考[ArchWiki](https://wiki.archlinux.org/index.php/Touchpad_Synaptics)

这里没有使用配置文件,只是用[Synclient](https://wiki.archlinux.org/index.php/Touchpad_Synaptics#Synclient)这个命令行工具设置
{% endnote %}
<!--more-->
# 模拟鼠标点击

使用synclient查看当前触摸板设置

```
$ synclient -l
```

可以发现TapButton1, TapButton2, TapButton3, 这三个都是0
我们修改 TapButton1=1, TapButton2=3, TapButton3=2

```
$ synclient TapButton1=1 TapButton2=3 TapButton3=2
```

# 设置自然滚动
把VertScrollDelta 和 HorizScrollDelta设置成负数就可以了

```
$ synclient VertScrollDelta=-111 HorizScrollDelta=-111
```

------

其他一些配置比如环形滚动, 禁用触摸板什么的, 可以自己去wiki看说明.
