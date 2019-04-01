---
title: SpaceVim中自定义Markdown相关快捷键
comments: true
mathjax: false
date: 2019-03-18 23:14:48
tags: [SpaceVim,Markdown]
categories: Learning
---

<meta name="referrer" content="no-referrer" />

  SpaceVim对Markdown的注释居然是html版本的，预览的时候还是可以显示，百度了才知道用[[//]]:#()的方法，就想着弄了快捷键，又是挖坑踩坑...
  <!--more-->
# Markdown注释方法

## html标签 
注意：需要在前面空一行
```
<div style='display: none'>
哈哈我是注释，不会在浏览器中显示。
我也是注释。
</div>
```

## html注释
```
<!--哈哈我是注释，不会在浏览器中显示。-->

<!--
哈哈我是多段
注释，
不会在浏览器中显示。
-->
```

## 利用Markdown原理
利用markdown的解析原理来实现注释的。一般有的markdown解析器不支持上面的注释方法，这个时候就可以用此方法。

```
[comment]: <> (哈哈我是注释，不会在浏览器中显示。)
[comment]: <> (哈哈我是注释，不会在浏览器中显示。)
[comment]: <> (哈哈我是注释，不会在浏览器中显示。)
[//]: <> (哈哈我是注释，不会在浏览器中显示。)
[//]: # (哈哈我是注释，不会在浏览器中显示。)
```

其中，这种方法最稳定，适用性最强：
```
[//]: # (哈哈我是注释，不会在浏览器中显示。)
```

还看到这种最可爱，超级无敌萌的：
```
[^_^]: # (哈哈我是注释，不会在浏览器中显示。)
```

# SpaceVim中自定义快捷键
**下面是[SpaceVim官网](https://spacevim.org/cn)的说明**
> 启动函数   
> 由于 toml 配置的局限性，SpaceVim 提供了两种启动函数 bootstrap_before 和 bootstrap_after，在该函数内可以使用 Vim script。 
> 可通过 ~/.SpaceVim.d/init.toml 的 [options] 片段中的这两个选项 bootstrap_before 和 bootstrap_after 来指定函数名称，例如：
> 
> [options]  
> 　　bootstrap_before = "myspacevim#before"  
>  　　bootstrap_after  = "myspacevim#after"  
> 启动函数文件应放置在 Vim &runtimepath 的 autoload 文件夹内。例如：
> 
> 文件名：~/.SpaceVim.d/autoload/myspacevim.vim
> 
> function! myspacevim#before() abort  
> 　　let g:neomake_enabled_c_makers = ['clang']  
> 　　nnoremap jk <esc>  
> endfunction
> 
> function! myspacevim#after() abort  
> 　　iunmap jk  
> endfunction
> 函数 bootstrap_before 将在读取用户配置后执行，而函数 bootstrap_after 将在 VimEnter autocmd 之后执行。
> 
> 如果你需要添加自定义以 SPC 为前缀的快捷键，你需要使用 bootstrap function，在其中加入：
> 
> function! myspacevim#before() abort  
> 　　call SpaceVim#custom#SPCGroupName(['G'], '+TestGroup')  
> 　　call SpaceVim#custom#SPC('nore', ['G', 't'], 'echom 1', 'echomessage 1', 1)  
> endfunction

# vim中定义快捷键相关说明

## autocmd
- autocmd是一个十分强大的命令，在.vimrc中配置以后在用vim创建文件的时候就会自动执行一些命令

## 键盘映射
具体参照[Vim中的键映射](https://www.cnblogs.com/softwaretesting/archive/2011/09/28/2194515.html)


使用map命令，可以将键盘上的某个按键与Vim的命令绑定起来。例如使用以下命令，可以通过F5键将单词用花括号括起来：   

　　:map <F5> i{e<Esc>a}<Esc>  

其中：i{将插入字符{，然后使用Esc退回到命令状态；接着用e移到单词结尾，a}增加字符}，最后退至命令状态。

在执行以上命令之后，光标定位在一个单词上（例如amount），按下F5键，这时字符就会变成{amount}的形式。   

## 不同模式下的键盘映射
使用下表中不同形式的map命令，可以针对特定的模式设置键盘映射：


| Command  | Normal   | Visual    |Operator Pending |   插入模式  |命令行模式     |
| 	:---:  |	:---:   |	:---:     |	:---:           |	:---:       |	:---:         |
|   命令 	 | 常规模式 |可视化模式 |运算符模式       |	Insert Only | Command Line  |
| :map	   |     y	  |      y    |       	y       |             |               |    
| :nmap	   | y        |           |                 |             |               | 
| :vmap	 	 |          |       y   |                 |             |               |           
| :omap	 	 |          |           |        y        |             |               |  
| :map!	 	 |          |           |                 |       	y	  |     y         |         
| :imap	 	 |          |           |                 |    	y       |               |       
| :cmap	 	 |          |           |                 |             |     	y       |   

# SpaceVim中进行自定义SPC开头的键映射
下面是将SPC-v-c定义为Markdown文本中行注释，SPC-v-u对Markdown进行行取消注释，SPC-v-p进行Markdown文件样式预览.

**~/.SpaceVim.d/autoload/myspacevim.vim**

```
function! myspacevim#before() abort
　　set wrap

    "v开头为自定义快捷键
　　call SpaceVim#custom#SPCGroupName(['v'], '+Personal Key Bindings')

    "使用v-p进行markdown网页预览
　　autocmd BufRead,BufNewFile *.{md,mdown,mkd,mkdn,markdown,mdwn} call SpaceVim#custom#SPC('nore', ['v', 'p'], ':!google-chrome-stable "%:p"', 'Markdown-Previews',1)

    "对markdown进行行注释，在行首插入[//]:#(,在行尾插入右括号),命令是'I[//]:#(<Esc>A)<Esc>' 
　　call SpaceVim#custom#SPC('nore', ['v', 'c'], 'I[//]:#(A)', 'Markdown-comment one line', 0)

    "对markdown取消行注释,在行首删除[//]:#(,在行尾删除右括号)
　　call SpaceVim#custom#SPC('nore', ['v', 'u'], '07x$x', 'Markdown-uncomment one line', 0)
endfunction

```

---------------
---------------

其实...我就想弄两个快捷键...
