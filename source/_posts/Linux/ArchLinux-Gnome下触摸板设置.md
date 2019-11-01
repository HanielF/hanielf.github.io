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
## 前言
Arch下触摸板默认是没有模拟鼠标单击双击功能的, 通过设置可以修改,具体参考[ArchWiki](https://wiki.archlinux.org/index.php/Touchpad_Synaptics)

这里没有使用配置文件,只是用[Synclient](https://wiki.archlinux.org/index.php/Touchpad_Synaptics#Synclient)这个命令行工具设置

如果希望长期有效，还是需要使用配置文件，配置文件样例见文末
{% endnote %}
<!--more-->
## 模拟鼠标点击

使用synclient查看当前触摸板设置

```
$ synclient -l
```

可以发现TapButton1, TapButton2, TapButton3, 这三个都是0
我们修改 TapButton1=1, TapButton2=3, TapButton3=2

```
$ synclient TapButton1=1 TapButton2=3 TapButton3=2
```

## 设置自然滚动
把VertScrollDelta 和 HorizScrollDelta设置成负数就可以了

```
$ synclient VertScrollDelta=-111 HorizScrollDelta=-111
```

## 设置触摸板禁用
实际使用过程中，发现触摸板常常出现误触的情况，打字的时候很容易就跑偏了。
所以设置：在鼠标存在的情况下，禁用触摸板，没有鼠标的时候才可以用

使用的是gnome环境，根据wiki的说明，我们使用GNOME Shell扩展，TouchpadIndicator.
TouchpadIndicator主页：[TouchpadIndicator](https://www.ashessin.com/TouchpadIndicator/)

通过git安装:

```bash
$ git clone --depth = 1 “ https://github.com/user501254/TouchpadIndicator.git ” ; rm -rf TouchpadIndicator / .git
$ RM -rf 〜 /。本地 / share / gnome-shell / extensions / touchpad-indicator @ orangeshirt
$ mv TouchpadIndicator / 〜 /。本地 / share / gnome-shell / extensions / touchpad-indicator @ orangeshirt

# 重启Gnome
$ Alt+F2，r，Enter

# 在gnome-tweak-tool中启用扩展
```

## 配置文件样例

```
/etc/X11/xorg.conf.d/70-synaptics.conf
-------------------------------------------------
Section "InputClass"
      Identifier "touchpad"
      Driver "synaptics"
      MatchIsTouchpad "on"
             Option "TapButton1" "1"
             Option "TapButton2" "3"
             Option "TapButton3" "2"
             Option "VertEdgeScroll" "on"
             Option "VertTwoFingerScroll" "on"
             Option "HorizEdgeScroll" "on"
             Option "HorizTwoFingerScroll" "on"
             Option "CircularScrolling" "on"
             Option "CircScrollTrigger" "2"
             Option "EmulateTwoFingerMinZ" "40"
             Option "EmulateTwoFingerMinW" "8"
             Option "FingerLow" "30"
             Option "FingerHigh" "50"
             Option "MaxTapTime" "125"
EndSection
```


------

其他一些配置比如环形滚动什么的, 可以自己去wiki看说明.
