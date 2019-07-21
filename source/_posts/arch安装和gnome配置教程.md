---
title: arch安装和gnome配置教程
tags:
  - Arch
  - Gnome
  - UEFI
  - GPT
  - 双系统
  - Install
categories:
  - Daily
comments: true
mathjax: false
date: 2019-07-20 21:13:37
---

<meta name="referrer" content="no-referrer" />

{% note info %}
# 前言
两天前,我的arch,被我不小心升级glibc搞坏了..很久前就想重装下系统的,因此也就懒得重新再修复了,直接重装一个也挺好.

但是!隔了这么久重装系统搞得我心态都快崩了,各种找资料.
现在终于弄好啦,还是决定记录一下,以后就不用那么担心教程靠不靠谱
{% endnote %}
<!--more-->

# 说明
- 已有系统: Win10
- 安装系统: archlinux-2019.07.1-x86_64
- 安装磁盘: 机械硬盘的后半部分(前半部分是windows的数据)
- 启动方式: UEFI
- 分区格式: GPT
- 桌面系统: Gnome3
- 参考链接: [ArchWiki](https://wiki.archlinux.org/index.php/Installation_guide_(%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87)#%E5%AE%89%E8%A3%85%E5%89%8D%E7%9A%84%E5%87%86%E5%A4%87)
- 参考链接: [Arch安装教程](http://blog.lucode.net/linux/archlinux-install-tutorial.html)
- 参考链接: [Arch安装图文教程](https://blog.csdn.net/r8l8q8/article/details/76516523)


# 准备工作
## 准备安装介质
说的简单点就是为制作启动盘做准备,安装系统当然需要这个系统的镜像文件.
可以到[Arch Download](https://www.archlinux.org/download/)这里下载镜像
一直往下可以看到CHINA的标志, 推荐选择163的源下载,[链接在这](http://mirrors.163.com/archlinux/iso/2019.07.01/)

## 制作LiveCD
如果在windows环境制作的,可以使用[USBwriter](http://sourceforge.net/p/usbwriter/wiki/Documentation/),或者其他一些工具

因为我电脑有现成的fedora media writer,所以用这个也可以制作

## 网络
身边需要可以连接的网络,有线网或者无线网

如果没有wifi可以连,当然,这个wifi不能是校园网,不然没办法直接命令行连接.
没wifi可以手机开热点啊,hhh

## 磁盘空间
从widows盘那边分出一部分给linux,我选择先压缩卷,然后新建简单卷,之后的安装就安装在这个简单卷上.

选择新建简单卷还是因为安装的时候能够看的更清楚第一个分区的开始扇区

## 设置usb启动
如果以前设置过usb启动优先就不用管了

进入BIOS,设置开机选项,调整开机顺序,让usb启动排在第一位
完事之后就可以直接插入你的U盘,开始装系统!

# 安装准备
## 联网
后面需要安装很多东西,所以最好先联网

```
# wifi-menu
```

会跳出来wifi选择的页面,自己选之前准备好的wifi连上就成

## 编辑mirrorlist
目的是为了后面下载的时候速度能快点,国外的镜像站太慢啦.

```
# vi /etc/pacman.d/mirrorlist
# 按下面描述修改mirrorlist
# pacman -Syy
```

我默认你已经会vim了,找163关键词,然后把163的镜像站复制粘贴到第一个镜像站前面.一堆的网址就是镜像站啦.

如果不放心,还可以找关键词China,把其他的ustc之类的都放前面去.
记得,保存后`pacman -Syy`

## 分区
**最重要的一步来咯**
我选择了UEFI启动,对应使用GPT分区
我的机械硬盘是/dev/sdb

### 分区规划
创建4个分区如下
- /boot　200M　/dev/sdb3
- /swap　8G　　/dev/sdb4
- /root　60G　 /dev/sdb5
- /home　150G　/dev/sdb6

别问我为什么/root设置这么大，为什么人家都推荐30G左右...
重装的其中一个原因就是/root爆了，扩容又麻烦，原本设置的是40G
/swap设置的是和我物理内存一样的大小

### 查看磁盘情况
确定你要安装的磁盘是sd几，然后使用gdisk划分磁盘

```
# fdisk -l
```
你看到的东西应该类似下面的
> Disk /dev/sdb: 931.53 GiB, 1000204886016 bytes, 1953525168 sectors
> Disk model: HGST HTS721010A9
> Units: sectors of 1 * 512 = 512 bytes
> Sector size (logical/physical): 512 bytes / 4096 bytes
> I/O size (minimum/optimal): 4096 bytes / 4096 bytes
> Disklabel type: gpt
> Disk identifier: 94717E4F-9437-4814-96AA-5CD870012F36
> 
> Device          Start        End   Sectors  Size Type
> /dev/sdb1        2048  587202559 587200512  280G Microsoft basic data
> /dev/sdb2   587202560 1468008447 880805888  420G Microsoft basic data
> /dev/sdb3  1468008448 1468418047    409600  200M EFI System
> /dev/sdb4  1468418048 1485195263  16777216    8G Linux swap
> /dev/sdb5  1485195264 1611024383 125829120   60G Linux root (x86)
> /dev/sdb6  1611024384 1925597183 314572800  150G Linux home
> 
> 
> Disk /dev/sda: 119.25 GiB, 128035676160 bytes, 250069680 sectors
> Disk model: SanDisk SD8SNAT-
> Units: sectors of 1 * 512 = 512 bytes
> Sector size (logical/physical): 512 bytes / 4096 bytes
> I/O size (minimum/optimal): 4096 bytes / 4096 bytes
> Disklabel type: gpt
> Disk identifier: A7EA8BD9-4B05-4419-AEFA-7B4F01E54CC2
> 
> Device         Start       End   Sectors   Size Type
> /dev/sda1       2048    534527    532480   260M EFI System
> /dev/sda2     534528    567295     32768    16M Microsoft reserved
> /dev/sda3     567296 246266737 245699442 117.2G Microsoft basic data
> /dev/sda4  246267904 250058751   3790848   1.8G Windows recovery environment

通过容量判断我要安装的磁盘是sda还是sdb

通过Type可以看到sdb1和sdb2都是Microsoft basic data,那是我的D盘和F盘

### 分区

```
gdisk /dev/sdb
```
记得这里sdb要换成你自己的sd..

gdisk的使用很简单，先使用？打印帮助，使用p打印分区表

如果你的磁盘是空的，使用o新建一个gpt分区表，如果不是空磁盘，别这么做

接着使用ｎ命令新建一个磁盘分区,然后输入分区号，默认回车就可以
然后设置开始扇区，如果没啥意外，直接回车
设置结束扇区，可以使用**+60G**这样的方式，不用自己计算，很方便
接着设置Hex code, 这个很重要,可以使用L查看所有的codes,下面列出要用的
> /boot: EF00
> /swap: 8200
> /root: 8303
> /home: 8302

### 格式化分区
格式化EFI分区/boot

```
# mkfs.fat -F32 /dev/sdb3
```

格式化/root和/home

```
# mkfs.ext4 /dev/sdb5
# mkfs.ext4 /dev/sdb6
```

开启swap分区/swap

```
# mkwsap /dev/sdb4
# swapon /dev/sdb4
```

### 挂载分区
一般是将根分区/挂载到/mnt下,然后将/boot和/home挂载到/mnt/boot和/mnt/home

```
# mount /dev/sdb5 /mnt
# mkdir /mnt/{boot, home}
# mount /dev/sdb6 /mnt/home
# mount /dev/sdb3 /mnt/boot
```

# 安装基本系统
## 部署基本系统

```
# pacstrap -i /mnt base base-devel net-tools
```
这里的net-tools提供了netstat和ifconfig等命令,可以选择不装,差别不大

## 生成fstab
fstab中记录了挂载信息,使用下面命令生成

```
# genfstab -U -p /mnt >> /mnt/etc/fstab
```
使用`cat /mnt/etc/fstab`检查

## 基本系统设置
### 切换到新系统

```
# arch-chroot /mnt /bin/bash
```
### 设置locale

```
# vim /etc/locale.gen
```
这里是配置本地语言环境,起码要中英文的UTF-8,所以把`en_US.UTF-8
UTF-8`和`zh_CN.UTF-8 UTF-8`取消注释, 然后保存退出
然后执行: 

```
# locale-gen
# # echo LANG=en_US.UTF-8 > /etc/locale.conf
```

### 设置时区
执行如下代码:

```
# ln -sf /usr/share/zoneinfo/Asia/Shanghai /etc/localtime
```

### 设置硬件时间
执行如下代码:

```
# hwclock --systohc --utc
```

### 设置主机名
建议使用小写

```
# echo 主机名 > /etc/hostname
# vim /etc/hosts
```
hosts文件中有如下内容
> 127.0.0.1	localhost
> ::1		localhost
> 127.0.1.1	myhostname.localdomain	myhostname

如果系统有一个永久的 IP 地址，请使用这个永久的 IP 地址而不是 127.0.1.1

### 生成Initramfs
注意,这步只对 LVM、 system encryption 或 RAID有效
正常情况下,在执行pacstrap的时候已经安装linux,并且mkinitcpio自动运行

```
# mkinitcpio -p linux
```

### 用户设置
修改root用户密码,并且创建一个新用户,同样修改密码
建议root密码和用户密码不要设置成一样的,平时用普通用户就可以

**如果需要用root,必须清楚自己每一步会有什么后果!**

```
# passwd
# ****你的密码****
# useradd -m -g users -s /bin/bash 用户名
# passwd 用户名
# ****用户密码****
```

### 配置UEFI引导(重点)
(这里网上看到的都有点不太一样, 导致我安装完之后电脑一直找不到Linux的引导项,
一直进不来linux, 后来在archwiki上看到了正确的方法, 记录如下)

这里使用UEFI引导,而不是BIOS,两者的区别自行百度

关于启动加载器可以自己查看[ArchWiki](https://wiki.archlinux.org/index.php/Arch_boot_process_%28%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87%29#%E5%90%AF%E5%8A%A8%E5%8A%A0%E8%BD%BD%E5%99%A8),里面讲的很清楚

我们选择GRUB [ArchWiki](https://wiki.archlinux.org/index.php/GRUB_(%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87)来引导系统,具体过程在archwiki中也讲的很清楚,这里不多赘述

安装必要软件包

```
# pacman -S dosfstools grub efibootmgr
```
**注意,我们选择的EFI系统分区是之前的/boot, 选择的启动引导器标识是GRUB,
不懂可以不管, 直接执行**

执行下面的命令来将 GRUB EFI 应用 `grubx64.efi` 安装到
`/boot/EFI/GRUB/`，并将其模块安装到 `/boot/grub/x86_64-efi/`。

```
# grub-install --target=x86_64-efi --efi-directory=/boot --bootloader-id=GRUB
```

安装完之后GRUB目录位于在/boot/grub/

然后执行下面命令生成主配置文件`grub.cfg`

```
# grub-mkconfig -o /boot/grub/grub.cfg
```

在/boot目录使用`tree -d`会有如下内容
> .
> ├── EFI
> │   └── GRUB
> ├── grub
> │   ├── fonts
> │   ├── themes
> │   │   └── starfield
> │   └── x86_64-efi
> └── syslinux

### 退出chroot重启
笔记本退出之前要先安装dialog

```
# pacman -S iw wpa_supplicant dialog
# exit
# umount /mnt/{boot, home}
# umount /mnt
# reboot
```

其实不重启也可以, 这时候还连着网, 直接把需要安装的一次性装了,
然后直接进到图形界面, 不然联网会很麻烦. 图形界面可以用networkmanager连网

# 驱动安装
## 显卡驱动
具体的驱动匹配表可以查看[ArchWiki](https://wiki.archlinux.org/index.php/Xorg_%28%E7%AE%80%E4%BD%93%E4%B8%AD%E6%96%87%29#%E5%AE%89%E8%A3%85)

再次折腾了一次Nvidia驱动, 失败告终, 难受
显卡驱动默认的vesa可以兼容大部分显卡了
我是双显卡, 安装了下面三个显卡驱动

```
# pacman -S xf86-video-vesa     通用驱动
# pacman -S xf86-video-intel    intel集显驱动
# pacman -S xf86-video-nouveau  开源的N卡驱动
```

## 触摸板驱动

```
# pacman -S xf86-input-synaptics
```

## Xorg显示服务器
xorg是其他例如xfce和gnome, kde一些桌面环境的基础, 提供图形环境基本框架

```
# pacman -S xorg
```

# 安装Gnome桌面环境
