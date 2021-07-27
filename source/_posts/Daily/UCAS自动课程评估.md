---
title: UCAS课程评估命令
tags:
  - Daily
  - UCAS
  - Automatic
  - JS
categories:
  - Daily
comments: true
mathjax: false
date: 2021-06-08 18:07:52
urlname: ucas-course-evaluation
---
<meta name="referrer" content="no-referrer" />

{% note info %}

国科大课程一键评估，自动评估脚本。

{% endnote %}

<!--more-->

## 课程评估

chrome浏览器F12打开开发者工具，然后Console中输入下面的命令。

```js
var tds = document.getElementsByTagName('td');
for(var i = 0; i<tds.length;i++){
    var cur_input = tds[i].getElementsByTagName("input")[0];
    if (cur_input.value==5 ) cur_input.checked=true;
}

$("textarea[id='item_"+316+"']").text("第一部分的基础知识讲的非常快，后面的都很详细，最喜欢的就就是数量合理的作业，加深了对基础概念的理解")
$("textarea[id='item_"+317+"']").text("第一部分的基础知识讲的非常快，如果再详细点就更好了")
$("textarea[id='item_"+318+"']").text("我平均每周在这门课程上花费8、9个小时")
$("textarea[id='item_"+319+"']").text("课程是非常指导实践的基础课程，一直很感兴趣")
$("textarea[id='item_"+320+"']").text("本课程我是满勤，良好的完成作业")

document.getElementById(322).checked=true
document.getElementById(329).checked=true
document.getElementById(331).checked=true
```

## 教师评估

```js
var tds = document.getElementsByTagName('td');
for(var i = 0; i<tds.length;i++){
    var cur_input = tds[i].getElementsByTagName("input")[0];
    if (cur_input.value==5 ) cur_input.checked=true;
}
$("textarea[id='item_"+364+"']").text("最喜欢老师能够结合理论和实践，同时能够在课堂上讲述的生动形象")
$("textarea[id='item_"+365+"']").text("希望老师能够增加和学生的互动，同时增强同学对知识的实践")
```
