---
title: Arch安装YCM
tags:
  - Arch
  - Linux
  - Vim
  - YCM
categories:
  - Linux
comments: true
mathjax: false
date: 2019-11-10 11:26:57
urlname: arch-install-ycm
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## 前言
安装几次YCM都有问题，记录下安装过程。其实挺简单的。
{% endnote %}
<!--more-->

## 前提
安装clang，boost，llvm。我用的是Arch，其他系统自己装llvm，clang，libclang，libboost，cmake

```
$ sudo pacman -S clang boost llvm-libs cmake
```

## 正式安装

### 安装YCM

看你用的是vundle还是什么。因为我是用的SpaceVim，所以直接在配置文件中enable这个选项就可以了，然后添加一些配置。再次打开vim后会自动安装至`~/.cache/vimfiles/repos/github.com/Valloric/YouCompleteMe`。

或者可以通过git安装

```
$ git clone https://github.com/ycm-core/YouCompleteMe.git
$ git submodule update --init --recursive
```

### 编译

需要编译一下，直接使用`install.py`或者`install.sh`

```
$ cd ~/.cache/vimfiles/repos/github.com/Valloric/YouCompleteMe
$ ./install.sh --clang-completer --java-completer
```

### 构建ycm_core库

这一步需要cmake和python3

```
$ mkdir ~/tmp/ycm_build
$ cd ~/tmp/ycm_build
$ cmake -G "Unix Makefiles" -DUSE_SYSTEM_BOOST=ON -DUSE_SYSTEM_LIBCLANG=ON . /home/hanielxx/.cache/vimfiles/repos/github.com/Valloric/YouCompleteMe/third_party/ycmd/cpp
$ cmake --build . --target ycm_core --config Releas
```

上面步骤自己检查输出是否正常，如果出现`NOT using libclang, nosemantic completion for C/C++/ObjC will be avaiable`之类的输出，那么还是有问题的，没有C家族语言的支持。

### 配置ycm_extra_conf.py

```
$ cd ~/.SpaceVim.d
$ cp /home/hanielxx/.cache/vimfiles/repos/github.com/Valloric/YouCompleteMe/third_party/ycmd/examples/.ycm_extra_conf.py ./
```

### 配置YCM
还要去修改下`~/.SpaceVim.d/init.vim`，配置YCM。下面是YCM部分的配置，注意第一行，需要加上python2的位置

```
let g:ycm_server_python_interpreter='/usr/bin/python2'
let g:spacevim_enable_ycm = 1
let g:ycm_complete_in_comments = 1
let g:ycm_confirm_extra_conf = 0
let g:ycm_seed_identifiers_with_syntax = 0
let g:ycm_error_symbol = '✗'
let g:ycm_warning_symbol = '!'
let g:ycm_global_ycm_extra_conf = '~/.SpaceVim.d/.ycm_extra_conf.py'
let g:ycm_semantic_triggers =  {
  \   'c' : ['->', '.'],
  \   'cpp,objcpp' : ['->', '.', '::'],
  \   'php' : ['->', '::'],
  \   'cs,java,javascript,typescript,d,python,perl6,scala,vb,elixir,go' : ['.'],
  \   'ruby' : ['.', '::'],
  \   'lua' : ['.', ':'],
  \ }
let g:ycm_filetype_blacklist = { }
```

---------
