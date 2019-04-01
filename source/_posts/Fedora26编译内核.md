---
title: Fedora 26 编译内核
comments: true
mathjax: false
date: 2019-04-01 00:09:49
tags: [Linux,Learning]
categories: Linux
---

<meta name="referrer" content="no-referrer" />
被迫很不情愿的编译内核...以前服务器上编译过，贼麻烦，现在又要编译，为了不让我的Arch出什么幺蛾子，新弄了个虚拟机练手了。    
还是Fedora26的，现在都出29了，时间过得真快。

<!--more-->
# 准备工作
## 查看自己内核版本
    uname -a
    Linux localhost.localdomain 4.11.8-300.fc26.x86_64 #1 SMP Thu Jun 29 20:09:48 UTC 2017 x86_64 x86_64 x86_64 GNU/Linux

## 下载内核并解压到/usr/src目录下
推荐一下[上海交大的网站](http://ftp.sjtu.edu.cn/sites/ftp.kernel.org/pub/linux/kernel/v5.x/)，速度挺快的。  
下载好后`tar xf linux-5.0.1.tar.gz -C /usr/src`解压到/usr/src下

## 注意事项
{% note danger %}
make编译内核起码预留10个G，编译完之后，不然存储会不够，又要重新弄。
boot分区发现200M会不够用，改成500M可以，扩容还是挺麻烦的
/ 根目录（具体来说是 /lib，没挂载/lib的话就默认是使用/目录） 要有至少4G的空余空间
{% endnote %}

# 编译安装
## make makemenuconfig
这是带有图形化界面的配置命令，在这里面可以定制很多功能。  
常见的是定制内核版本，或者开启ntfs的功能。

内核版本在general setup里面，找到local version -append to kernel release

ntfs在file-systems下面，很下面的位置，找到DOS/FAT/NT Filesystems，进入后用M键启用NTFS，用模块的方式，并在下面的NTFS write support 那里用y键启用  

然后 保存退出

如果想用默认配置，可以直接方向键选择**save**，然后**exit**。

刚开始使用这个命令可能会一直报错，解决的基本原则是提示缺少什么包

就在这个包的名字后面加上-devel，然后用dnf安装（其他版本也一样，只要换一下包管理器名）

### 缺少ncurses-devel包
    dnf install ncurses-devel

### 缺少flex包
    dnf install flex

### 缺少bison包
    dnf install bison

### 报错：You are building kernel with non-retpoline compiler.
应该升级GCC了，刚开始一直卡在这，网上也没发现non-retpoline是什么鬼，后来才发现是gcc版本太低，可能我刚装的虚拟机，没升级...      

    dnf update gcc

### 缺少libelf-dev, libelf-devel or elfutils-libelf-devel
报错：error: Cannot generate ORC metadata for CONFIG_UNWINDER_ORC=y, please install libelf-dev, libelf-devel or elfutils-libelf-devel

这个直接安装会发现源里面没有，可以选择rpm包安装,也可以：

    dnf install binutils gcc make patch libgomp glibc-headers glibc-devel kernel-headers kernel-devel dkms

这也是百度到的办法...当时都惊了，源里面居然都没有...

这里在装dkms,kernel-devel,patch的时候，安装了elfutils-libelf-devel和zlib-devel依赖

### 缺少openssl
报错:scripts/sign-file.c:25:10: fatal error: openssl/opensslv.h: No such file or directory
 #include <openssl/opensslv.h>

 这里`dnf install openssl`发现已经安装了，很奇怪，后来发现libssl-devel在redhat系这里叫**openssl-devel**

所以`dnf install openssl-devel`

 如果是**debian**系的，应该是安装**openss**和**libssl-devel**

## make mrproper 
清除编译过程中产生的所有中间文件   

假如你之前也编译过内核，而你没有用此命令去清除之前编译产生的.o文件，那么，在make的时候，可能就会产生干扰。  

清除之后要重新make menuconfig 生成.config文件

## make -j8
**-j**是代表编译时用几个线程，这里开了8个线程来编译，不然太慢了。

## make modules_install
安装内核模块，这里比较快，一会会就好了，安装完后可以看到/lib/modules目录下就会出现新的内核。

## make install
安装bzImage为/boot/vmlinuz-VERSION-RELEASE，并生成initramfs文件  
使用ls /boot就可以查看新生成的文件，注意以安装的版本结尾的文件就好

查看grub.cfg, `ls /boot/grub2`, 应该就有grub.cfg引导文件了

## 重启
这时候应该就可以在启动项发现新的内核了

## 删除旧内核
- 删除/lib/modules/目录下不需要的内核库文件

- 删除/usr/src/linux/目录下不需要的内核源码

- 删除/boot目录下启动的内核和内核映像文件

- 更改grub的配置文件，删除不需要的内核启动列表

------------------
这时候内核编译就结束了，还是挺耗时间的，特别是编译的时候。

