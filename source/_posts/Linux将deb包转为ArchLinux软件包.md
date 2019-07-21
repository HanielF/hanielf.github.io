---
title: Linux将deb包转为ArchLinux软件包
tags:
  - Deb
  - Debtap
  - Daily
  - ArchLinux
categories:
  - Daily
comments: true
mathjax: false
date: 2019-07-03 18:35:48
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## 前言
有时候可能找不到arch用的包，只有deb包或者rpm包。
将deb包转换为ArchLinux的包比较方便，使用了debtap这个工具
debtap代表了 DEB T o A rch （Linux） P ackage的意思
{% endnote %}
<!--more-->

## 安装debtap并更新数据
### 安装
**[依赖关系:]** 需要提前安装好** bash， binutils ，pkgfile 和 fakeroot 包**
如果直接安装会帮你安装依赖.

```bash
yaourt -S debtap
```
遇到需要编辑的就默认enter跳过，需要安装的就确认，然后等就行了

### 创建/更新 pkgfile 和 debtap 数据库。

```bash
sudo debtap -u
```

## 转化
假如要转化name.deb

```bash
debtap name.deb
```

中间可能要输入点东西，自己看情况输入就好

**其他参数：**
- 略过除了编辑元数据之外的所有问题

```bash
debtap -q name.deb
```

- 略过所有的问题（不推荐）

```bash
debtap -Q name.deb
```

- 查看帮助

```bash
debtap -h
```

## 安装软件包
使用pacman安装软件包

```bash
sudo pacman -U <package-name>
```

---------
