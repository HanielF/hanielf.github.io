---
title: Markdown学习笔记
comments: true
mathjax: false
date: 2017-08-02 15:52:13
tags: [Learning,Markdown]
categories: Learning
---

<meta name="referrer" content="no-referrer" />

# 前言

本来博客初建，理应写点文章总结心得，说点自己的想法。  
但正好这时候又学了[Markdown](https://en.wikipedia.org/wiki/Markdowna)，不如就先写个学习笔记，供自己以后写文参考。

关于博客的搭建和以后的想法,就留着下次吧～  
<!--more-->

# 关于Markdown

## 宗旨和兼容性

[Markdown](https://en.wikipedia.org/wiki/Markdowna)目标是实现易读易写，使用其编写的文件可以直接以纯文本发布。  
[Markdown](https://en.wikipedia.org/wiki/Markdowna)兼容[HTML](https://en.wikipedia.org/wiki/HTML)，语法目标是成为一种适用于网络的书写语言。  

相比HTML，Markdown是一种书写的格式，而HTML更多是一种发布的格式。  
在Markdown文件内可以直接用HTML书写，不用额外标注。

[Markdown](https://en.wikipedia.org/wiki/Markdowna)兼容[HTML](https://en.wikipedia.org/wiki/HTML)，但是在[HTML](https://en.wikipedia.org/wiki/HTML)等区块元素，比如`<div>`,`<table>`,`<pre>`,`<p>`,等标签，必须在前后加上空行和其他内容隔开，还要求他们的开始和结尾标签，不能用制表符或者空格来缩进。

在[HTML](https://en.wikipedia.org/wiki/HTML)区块标签内的[Markdown](https://en.wikipedia.org/wiki/Markdowna)格式语法不会被处理  
但是在[HTML](https://en.wikipedia.org/wiki/HTML)区段标签内，[Markdown](https://en.wikipedia.org/wiki/Markdowna)语法是有效的。比如`<span>`,`<cite>`,`<del>`

## 特殊字符转换

在[HTML](https://en.wikipedia.org/wiki/HTML)中，< 和 & 想要显示字符原型需要用实体的形式，`&lt` 和
`&amp`  
而在[Markdown](https://en.wikipedia.org/wiki/Markdowna)中，则可以自由书写字符。

**注** :在code范围内，< 和 & 都会一定被转换成HTML实体，因此可以更方便的写出HTML code

# 区块元素

## 段落和换行

段落由连续文本行组成，且允许段落内用换行符强迫换行。

如果想插入`<br />`，需要在插入处按入两个以上空格然后回车。  
段落的前后要有一个以上空行，且普通段落不可以用空格或者制表符缩进。

## 标题

支持两种标题的语法，类Setext和atx形式

### 类Setext

类Setext用底线的形式，利用任意数量=（最高阶）和－（第二阶）

**栗子** :  

    This is an H1
    ============
    
    This is an H2
    -------------  
  
### Atx

Atx形式则是在行首插入一到六个#,对应标题一到六阶  
```
# H1

##  H2

##### H5  
```
  
可以选择闭合#，且结尾的#和开头不用一样  
```
# H1 #

## H2 ##

### H3  ###  
```
  
## 区块引用

在每行前面加上>,(可以偷个懒在整个段落的第一行加上>),并且区块引用可以嵌套，只要根据层次加上不同数量的>

**栗子** :  
``` 
> This is the fiest level of quoting 
>
>> This is nested blokquote
>
> back to first level  
```
显示为:
> This is the fiest level of quoting 
>
>> This is nested blokquote
>
> back to first level  
  
在引用区块内也可以使用其他[Markdown](https://en.wikipedia.org/wiki/Markdowna)语法， **栗如**
标题、列表、代码区块  

    > ## 这是一个标题。
    >
    > 1.   这是第一行列表项。
    > 2.   这是第二行列表项。
    >
    > 给出一些例子代码：
    >
    >     return shell_exec("echo $input | $markdown_script");  
显示为:
> ## 这是一个标题。
>
> 1.   这是第一行列表项。
> 2.   这是第二行列表项。
>
> 给出一些例子代码：
>
>     return shell_exec("echo $input | $markdown_script");  
  
## 列表

支持有序列表和无序列表两种

### 无序列表

无序列表使用 + - 作为列表标记，个人偏向－，因为不用按shift

### 有序列表

有序列表则使用数字接着一个英文句点:  

    1.  First one
    
    2.  Second one  
显示为:

1.  First one

2.  Second one  

使用不同的数字不会有问题，但是看着不舒服，不推荐，还是顺序下来。  
或者也可以只用一个数字。

**栗如**  

    1.  First one
    
    1.  Second one
    
    1.  THird one  
  
  
列表通常在最左边，但是也可以缩进，最多三个空格，标记后面一定要接上至少一个空格或者制表符  
如果列表间由空行，[Markdown](https://en.wikipedia.org/wiki/Markdowna)会用\标签将内容裹起来

**栗子** ：  
    
    *   Bird
    
    *   Magic  
  
会被转换成：  
    
    >    <ul>
    
    >    <li><p>Bird</p></li>
    
    >    <li><p>Magic</p></li>
    
    >    </ul>  

显示为:
*   Bird

*   Magic  
  
列表可以包含多个段落，但是记得每个段落都要缩进哦

如果放引用，>也要缩进  
如果放代码区块，这个区块就要缩进两次  
如果要在行首输入2017. 这种数字+句点+空白，可以在句点前面加反斜杠，即2017.

## 代码区块

终于到这啦，代码区块最简单了，只要简单的缩进4个空格或者一个tab,或者,```这个标识,

**栗子** ：  
* ____printf("hello world")
* <Tab>printf("hello world") 
* \```   
  printf("hello world")    
  \```   

显示为:   

    printf("hello world")
  
markdown 会自动在代码区块外面加上\，而且代码区块里面& < >会自动转成[HTML](https://en.wikipedia.org/wiki/HTML)实体，所以可以想怎么写code就怎么写

## 表格

表格对齐方式  
   
    1.  居左: :----
    
    2.  居中: :----: 或者　-----
    
    3.  居右: ----:  
  
## 分割线

可以在一行中用三个以上的*，-，_ 来建立一个分割线。行内不可以用其他东西，可以在* -中间插入空格。  

    * * *
    
    ****
    
    ---
    
    ___  

显示为:
* * *

****

---
___  
  
# 区段元素

## 链接

支持行内式和参考式两种，但都是用[]标记链接文字

### 行内式

在方括号后面接一个()在里面写上网址就行，如果要加title,在后面用单引号、双引号或是括弧把title文字包起来就行

**栗子** ：  

    This is [an example](http://example.com/ "Title") inline link.
    
    [This link](http://example.net/) has no title attribute.  
  
  
如果想要链接到同主机资源，用相对路径  

    See my [About](/about/) page for details.  
  
### 参考式

参考式是在后面加上另一个[],在里面写上标记  
    
    This is [an example][id] reference-style link.  
  
可以在方括号中间加空格。。。  
最后在文件的任意处，可以是段尾，可以是文件尾，把标记的链接定义出来  

    [id]: http://example.com/  "Optional Title Here"  
  
要注意的是[]后面有一个:,还有一个以上的空格，id这个标记是不区分大小写的！链接网址可以用<>包起来。

### 隐式链接

隐式链接标记功能可以让你省略号指定的链接标记，这种情况标记会被视为等同于链接文字。  
隐式链接只要在链接文字后面加上一个空的[]

**栗子** ：  
    
    [Google][]
    
    [Google]: http://google.com/  
  
参考式链接的优点是比较好读，可以将一些标记的元数据移到段落之外，可以是段尾文件尾，这样就可以不让文章的阅读感被打断

## 强调

如果你的* 和_ 两边都有空白，就只会被当成普通的* _

## 代码

如果要标记一小段行内代码，可以用反引号｀把它包起来  
    
    Use the `printf()` function.  
显示为:

Use the `printf()` function.  

如果要在代码区段内插入反引号，你可以用多个反引号来开启和结束代码区段：  

    ``There is a literal backtick (`) here.``  
显示为:

``There is a literal backtick (`) here.``  
  
代码区段的起始和结束端都可以放入一个空白，起始端后面一个，结束端前面一个，酱紫就可以在区段开始就加入一个反引号  

    A single backtick in a code span: `` ` ``
    
    A backtick-delimited string in a code span: `` `foo` ``  

显示为:

A single backtick in a code span: `` ` ``

A backtick-delimited string in a code span: `` `foo` ``  

  
## 图片

使用行内式和参考式

### 行内式
    
    ![Alt text](/path/to/img.jpg)
    
    ![Alt text](/path/to/img.jpg "Optional title")  
  
### 参考式

    ![Alt text][id]
    
    [id]: url/to/image  "Optional title attribute"  
  
这部分和链接是差不多的,但如果想要指定图片的宽高，可以使用普通的`<img>`标签.

# 其他

## 自动链接

用<>包起来的，都会被自动转成链接  
    
    <http://example.com/>  
  
会被转成  

    <a href="http://example.com/">http://example.com/</a>  
  
## 反斜杠

可以用\插入一些在语法中有含义的符号  
    
    \   反斜线      `   反引号
    
    `*   星号       _   底线`
    
    {}  花括号      []  方括号
    
    ()  括弧        #   井字号
    
    \+   加号       \-   减号
    
    .   英文句点    !   惊叹号  
  
上面就是我学[Markdown][]的一些笔记，可能会有缺少的，有看到的可以指正～。

