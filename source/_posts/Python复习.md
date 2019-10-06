---
title: Python复习
tags:
  - Python
  - Review
  - Notes
categories:
  - Notes
comments: true
mathjax: false
date: 2019-08-28 18:20:16
urlname: python-review
---

<meta name="referrer" content="no-referrer" />

{% note info %}
系统的复习一下Python的知识点, 做一些笔记
{% endnote %}
<!--more-->
# CH1. Python语言基础

## 1.1 常量与变量
### 概念和注意点
1. 内存是以字节为单位的一片连续的存储空间, 变量名是内存单元的名称, 变量值是变量所对应的内存单元的内容, 变量地址是变量对应内存单元的地址.
2. 动态类型语言, 确定变量类型是在变量赋值的时候, **每次赋值**都可能改变类型
3. 是基于值的内存管理方式, 不同的值分配不同的空间.   
  因此, 变量值改变时, 改变的是变量的指向关系, 使得变量指向另一个内存空间.  
  在C语言中是改变变量内存空间的内容, 变量对应的内存空间是固定的.  
  使用** id()函数 **可以查看对象的内存地址.
3. python中的变量是对一个对象的引用, 变量和变量之间的赋值是对同一个对象的引用, 变量指向一个对象或内存空间, 内存空间的内容可以修改
4. 这里需要注意的是, 一些简单的对象, 比如较小的整型对象, Python采用对象重用的方法.   
举例: a=2, b=2, 不会分配两次内存, a和b同时指向一个对象

### 变量命名
1. 变量命名规则: 数字+字母+下划线, 不可以数字开头, 区分大小写. 单独的下划线是表示上一次运算的结果
2. 变量名不可以是,关键词:
```python
>>> import keyword
>>> print(keyword.kwlist)
['False', 'None', 'True', 'and', 'as', 'assert', 'async', 'await', 'break', 'class', 'continue', 'def', 'del', 'elif', 'else', 'except', 'finally', 'for', 'from', 'global', 'if', 'import', 'in', 'is', 'lambda', 'nonlocal', 'not', 'or', 'pass', 'raise', 'return', 'try', 'while', 'with', 'yield']
```

### 变量作用域
分局部变量和全局变量

函数外就是全局变量, 如果要在函数中修改全局变量, 使用`global`语句声明它是全局变量

如果同一程序中, 全局变量和局部变量同名, 则, 在局部变量作用范围内, 全局变量失效

如果在内层函数要修改外层函数变量:
- 使用global
- 使用nonlocal关键字, 在内层函数中使用nonlocal声明变量
- nonlocal  适用于在局部函数中的局部函数，把最内层的局部 变量设置成外层局部可用，但是还不是全局的。并且nonlocal必须绑定一个局部变量,不然报错

```python
def scope_test():
    def do_local():
        spam = "local spam"   #和外面的spam不一样
    def do_nonlocal():
        nonlocal  spam        #使用外层的spam变量
        spam = "nonlocal spam"
    def do_global():
        global spam
        spam = "global spam"
    spam = "test spam"
    do_local()
    print("After local assignmane:", spam)
    do_nonlocal()
    print("After nonlocal assignment:",spam)
    do_global()
    print("After global assignment:",spam)

scope_test()
print("In global scope:",spam)

# 输出
After local assignmane: test spam
After nonlocal assignment: nonlocal spam
After global assignment: nonlocal spam
In global scope: global spam
```

## 1.2 数据类型

### 1.2.1 数值类型
三种类型: 整型int, 浮点型float, 复数型complex

1. 整型数据:   
整数, 不带小数点, 可以有正负号.   
数据的值不固定长度, 只要内存允许, 可以**任意长度** 
四种表示:  
 - 十进制
 - 二进制, 0b或者0B开头
 - 八进制, 0o或者0O开头
 - 十六进制, 0x或者0X开头

2. 浮点类型:  
两种表示:    
 - 十进制小数: 数字+小数点, 小数点后面可以没有数字, 如: 3.23, 34.0, 0.0, 34.
 - 指数形式: 科学计数法表示的, e或者E表示10为底的指数. 两部分比如同时出现, 指数必须整数. 如4e-5, 45e-6. 提供17位**有效数字**精度, **不是17位小数!**
 - 注意可能选择题, 1234567890123456789.0 + 1 - 1234567890123456789 = 0.0, 1234567890123456789.0 - 1234567890123456789 + 1 = 1.0, 前者因为+1之后超过精度, 用科学计数法表示, +1 被忽略了, 所以结果为0. 

3. 复数类型:  
 - 表示形式: a+bJ, a是实部, b是虚部, J是-1的平方根, 是虚数单位, 也可写成j, 注意不是数学上的i
 - 用x.real和x.imag获得实部和虚部, 结果都是**浮点型**

### 1.2.2 字符串类型
1. 使用单引号, 双引号, 或者三引号定义标准字符串, 三引号的字符串可以多行
2. 使用下表访问字符串中的字符
3. 字符串中字符不可改变, 只能重新赋值
4. 使用转义字符控制输出  
注意\0: 空字符, \a:响铃, \r: 回车, 表示对当前行重叠输出, 只回车不换行
5. 字符串前面加r表示原始字符串, 不转义
6. **eval()函数**: 把字符串内容当作python语句执行, eval('1+1'), 结果为2
7. **len()函数**: 返回长度

### 1.2.3 布尔类型
1. True和False, 注意开头大写
2. 实际上分别用整型值**1和0**参与运算

### 1.2.4 复合数据类型
1. 包含多个相关联的数据元素,因此称为复合数据类型.
2. 列表, 元组和字符串是又顺序的, 称为序列.  
字典和集合是无顺序的.
3. 列表:  
 - 中括号, 逗号分隔, 类型可不同
 - 可以嵌套
 - 列表元素可以改变
4. 元组:   
 - 1,2,3 这样省略括号的默认为**元组**
 - 小括号, 逗号分隔, 类型可不同
 - 元组元素不可修改
 - () 表示空元组
 - **(9, )**只有一个元素, 要用逗号结尾
 - (9)表示整数9
5. 字典:  
 - 大括号, 逗号分隔
 - 关键字: 值, 关键字必须不可变类型且互不相同
 - {} 空字典
6. 集合:  
 - 无序, 不重复
 - 大括号或者set()函数创建
 - set() 空集合

## 1.3常用模块和函数
三种导包
```python
import math
math.sqrt(2)
====
from math import sqrt
sqrt(2)
===
from math import *
sqrt(2)
```
### 1.3.1 math模块  
常量: e, pi  
fabs(), sqrt(), pow(x, y), exp(x), ceil(x), floor(x), fmod(y, x)  
sin(x), cos(x), tan(x)...  

### 1.3.2 cmath模块
相对math支持复数运算  
polar(x): 复数笛卡儿坐标换为极坐标  
rect(r,p): 复数极坐标换为笛卡儿坐

### 1.3.3 random模块
- seed(x)随机数种子, 种子相同, 随机数相同, 默认种子是时间  
- choice([0,1,2])随机挑选一个  
- sample(seq, k): 从序列中挑k个  
- shuffle(seq): 随机排序
- random(): [0, 1)内实数
- randint(a, b): [a, b]内整数
- uniform(a,b): [a,b]内实数

### 1.3.4 time模块
- time(): 返回时间戳
- localtime(secs): 接收一个时间戳, 返回时间元组
- asctime([]): 接收一个时间元组, 返回一个日期时间字符串
- asctime(localtime(time())), 可以这样用
- ctime(time())和上面的结果相同

### 1.3.5 calendar模块
- 提供日历相关功能
- calendar.isleap(), 这个函数可以用来判断是否闰年

### 1.3.6 **内置函数**

- range([start,]end[,step]): 开始和步长可省, 默认开始0, 步长1, \[start, end),不包括end哦! 如果要使用step, 需要写start. range()函数返回的还是可迭代对象
- range()可以和list()或者tuple()结合: list(range(5)), tuple(range(5))


- ord(): 将字符转化为相应编码值
- chr(): 将整数转化为相应unicode字符


- repr(object): 返回一个对象的string格式, [菜鸟教程](https://www.runoob.com/python/python-func-repr.html)
- str() 用于将值转化为适于人阅读的形式，而repr() 转化为供解释器读取的形式
- iter(可迭代对象): 用于产生一个可迭代对象的迭代器, 可用next(it)得到迭代器下一个元素


- abs()
- pow(x,y[,z]): x^y%z
- round(x[,n]): 这个函数尽量不要用! 不是严格的四舍五入, 和浮点数精度有关, 具体可以参考[Python 中关于 round 函数的小坑](https://www.runoob.com/w3cnote/python-round-func-note.html)浮点数x, n为精确到小数点后的位数,可省默认保留整数, 结果仍浮点型
- divmod(x, y): 结果为(商,余数), divmod(7,4)==(1,3)

## 1.4 基本运算

### 1.4.1 算术运算
+, -, \*, /(返回浮点数), //(整除), %, \*\*(乘方)  
这里注意下//这个运算:
1. 如果是正数, 那就是去掉小数部分, **关键是负数, 需要向绝对值更大的方向进位**.   
举例: 4//3 == 1, -4//3 == -2, -7//3 == -3, -8//3 == -3
2. 如果a//b, a或b是浮点数, 那结果也是浮点型, 只有**两个都是整型**, 结果才是整型
举例: 4//3 == 1, 4.1//3 == 1.0  

注意取模运算的结果,只有**两个都是整型**才是整型, 例: 5%3 == 2, 5%3.0 == 2.0

### 1.4.2 计算误差
浮点数有效位有效, 17位, 所以肯定有误差.  
因此, 比较符点数的时候, == 运算符要小心使用, 应该判断是否约等于  
举例: abs(2.2-1.2-1)<1e-6, 取一个很小的误差就好  

```python
>>> 2.2-1
1.2000000000000002
>>> 2.2-1-1.2
2.220446049250313e-16
```

### 1.4.3 数据类型转换
整型+浮点型, 会自动将整型转换为浮点型  
**强制转换:**
- int(x)
- float(x)
- complex(x), 将x转换为复数, 实部为x, 虚部为0
- complex(x, y), x为实部, y为虚部

### 1.4.4 位运算
- &: 按位与, 全1为1, |: 按位或, 有1为1, ^: 按位异或,不一样为1, ~: 按位取反, <<: 左移, 低位补0, >>: 右移,高位补0

## 1.5 输入与输出
### 1.5.1 标准输入
格式:
- input([提示字符串])    

提示字符串可选, input读入一行, 并返回一个字符串.  
如果需要数值数据, 需要使用类型将字符串转化成数值   
可以给多变量赋值, 使用eval(input())

```python
# 读整数
>>> x=int(input())
12
>>> x
12

# 读多个变量
>>> x,y=eval(input())
1,2
>> x
1
>>> y
2
# 相当于x,y=(3,4)
```

### 1.5.2 标准输出
格式:
- print([输出项1,输出项2,..输出项n]\[,seq=分隔符][,end=结束符])

seq默认空格, end默认回车换行
没有输出项则输出空行. print()从左到右输出每个项

```python
>>> print(10, 20)
10 20
>>> print(10, 20, seq=',')
10,20
>>> print(10, 20, seq=',',end='*')
10,20*
#并且没换行
```

### 1.5.3 格式化输出
三种格式化输出方式:
- 用%
- 用format()内置函数
- 用字符串的format方法

#### 1.5.3.1 格式化运算符%
格式化字符串%(输出项1,输出项2...输出项n)

%的运算结果是字符串  

格式符:
- %% 百分号, %c 字符, %s 字符串, %d 带符号整数, %o 八进制, %x/X, %e/E 科学记数法, %f/F 浮点数, %g/G 根据值大小选%e还是%f
- 不管什么类型,都可以用%s

```python
>>> print('Values are %s, %s, %s'%(1,2.3,['one','two','three']))

# 表示总长度为6, 小数部分2位, 输出不足用0补齐, 小数点也占一位
# 0可以用-, +代替, -左对齐, +表在正数前加上+号,默认不加
>>> '%06.2f'%1.234
'001.24'

# 用字典输出时特殊用法, 这时候值由键确定
# '%(key)06.1f'%{'key':value}
>>> '%(name)s:%(score)06.1f'%{'score':9.5,'name':'Lucy'}
'Lucy:0009.5'

# %*.*f 运行中确定的
>>> '%0*.*f'%(6,2,2.345)
'002.35'
```

#### 1.5.3.2 format()内置函数

```
format(输出项[,格式字符串])
```

省略格式符等价与str()函数   

格式符:
- d, b(二进制), o, x, X
- f, F, e, E, g, G
- c(整数为编码的字符)
- %(输出百分号)
- <(左对齐), >(右对齐), ^(居中对齐), =(填充字符位于符号和数字之间), +(表正号)

```python
>>> print(format(65,'c'),format(3.145,'f'))
A 3.145000
>>> print(format(3.145,'6.2f'),format(3.145,'05.3'))
  3.15 03.15
>>> print(format(3.145,'0=+10'),format('test','>20'))
+00003.145                 test
```

#### 1.5.3.3 字符串的format()方法
属于字符串的类方法, 尽量用字符串的format()方法

```python
### 
'格式化字符串'.format(输出项1, 输出项2, 输出项3...输出项n)

格式字符串格式:
{[序号或者键]: 格式说明符} 冒号不能丢

格式化字符中, 普通字符原样输出, 格式说明符对应输出项

序号从0开始! 如果全部省略就按照顺序输出
### 

# 省略
>>> print('i am {}, {}'.format('Brenden','Welcome!'))
i am Brenden, Welcome!

# 序号
>>> print('first:{1}, second:{0}, third:{2}'.format(1,2,3))
first:2, second:1, third:3

# 混用序号和键
>>> print('hello, i am {name}, {0}'.format('Welcome',name="xiaoming"))
hello, i am xiaoming, Welcome

# 结合格式符
>>> print('{0:*^15}'.format(12345678))
***12345678****
>>> print('{0:<15}'.format(12345678))
12345678
>>> print('{0:>15}'.format(12345678))
       12345678
```

## 1.6 缩进和注释
### 1.6.1 缩进
我认为, Python相对其他语言, 缩进真的是一个很坑的点, 同一程序中最好是一样的缩进, 否则经常出现缩进错误   
一般是四个空格或者一个TAB   

### 1.6.2 注释
如果注释里面有中文,记得要在开头使用`#coding=utf-8`或者`#coding=gbk`

- 单行注释: #开头, #后的就是注释
- 多行注释: 三个单引号或者双引号, **多行注释不可以使用反斜杠续行!!**

## 1.7赋值语句
### 1.7.1 一般格式
变量=表达式
### 1.7.2 复合赋值语句

```
+=, -=, *=, /=, //=, **=, <<=, >>=, &=, |=, ^=
```

**注意!** `x*y+5==x*(y+5)`

### 1.7.3 多变量赋值
1. 链式赋值
 - 变量1=变量2=变量.....=变量n=表达式
 - a=b=10 等价于 a=10;b=10
 - 要注意的是, 上例, a和b同时指向一个整型对象10
2. 同步赋值
 - 变量1,变量2,变量3...=表达式1,表达式2...表达式n
 - a,b,c=1,2,3
 - 注意同步赋值不是单一赋值语句的先后执行!!!

```
>>> x,x=2,-50 
# x结果为-50

>>> x=-11
>>> x,y=12,x
# 此时x=12, 但是y=-11 !!!要小心使用

>>> a,b=1,2
>>> a,b=b,a
# 此时a, b分别为2, 1
# 可以用a,b=b,a省略一个中间变量
```

## 1.8 条件的描述
### 1.8.1 关系运算符
优先级小于算术运算符, 就是先算算术运算符, 再比较
```
<, <=, >, >=, ==, !=
```

### 1.8.2 逻辑运算

```
and(与), or(或), not(非)
```

优先级: 
- not > and > or

逻辑与: 前面的False就不算后面的
逻辑或: 前面的True就不算后面的

### 1.8.3 成员测试

```
in: 在序列中查找, 返回True或者False
not in: 么找到返回True
is: 测试两个变量是否, 指向同一个变量
```

{% cq %}
**end CH1**
{% endcq %}

# CH2. Python程序结构
## 2.1 算法及其描述
算法+数据结构=程序, 计算机解决问题的方法和步骤就是算法   
算法描述: 
1. 传统流程图
2. 结构化流程图:
    - 程序的三种基本结构: 顺序结构, 选择结构, 循环结构
    - 结构化流程图(N-S图): 顺序结构, 选择结构, 当型循环结构, 直到型循环结构, 各单元顺序执行

## 2.2 顺序结构
简单的按照出现顺序执行

## 2.3 选择结构
### 2.3.1 if-elif-else

要注意的是, if的表达式可以使任意的, 比如 if 'B':   
嵌套的时候根据对齐来配对

```python
# 单分支
if ...:
  表达式

# 双分支
if ...:
  语句
else:
  ...

# 多分支
if ...:
  ...
elif ...:
  ...
elif ...:
  ...
else:
  ...
```

### 2.3.2 条件运算

**可以作为一个运算量, 而不是一个单独的语句**

```python
表达式1 if 表达式 else 表达式2

>>> z= x if x>y else y
# 其中z=后面的就是一个条件运算

>>> i=0
>>> 'a' if i else 'A'
'A'
```

## 2.4 循环结构
### 2.4.1 while
while和if有一个相同的地方就是, 如果循环体只哟一个语句, 可以写在一行

```python
while ...:
  ....

# 可以结合else语句
# 但是只可以在while部分正常退出的时候才执行else部分
# 如果是break退出的, 不执行else部分
while ...:
  ....
else:
  ....
```

### 2.4.2 for
可结合range()函数, 看1.3.6部分

> range([start,]end[,step]): 开始和步长可省, 默认开始0, 步长1, [start, end),不包括end哦! 如果要使用step, 需要写start. range()返回可迭代对象
> for循环中会自动调用iter()函数,和next()函数获得range里的元素

```python
# 这个序列可以是: 字符串, 元组, 列表...
for 变量 in 序列:
  语句块

# 也可以用else语句
# 没碰到break就可以执行
for 变量 in 序列:
  语句块
else:
  ....

# 结合range函数
>>> for i in range(5):
>>>   print(i, end=',')
0,1,2,3,4
>>> for i in range(0,5,2):
>>>   print(i, end=',')
0,2,4
```

## 2.5 循环控制语句
有`break`和`pass`和`continue`这三个

pass语句代表空语句,空操作, 相当于占位符

{% cq %}
**end CH2**
{% endcq %}

# CH3. 字符串和正则
## 3.1 字符串
关于编码的两个函数
- 默认unicode编码, encode: 编码为, decode: 解码为unicode
- s.encode('utf-8')或者s.encode('gbk')
- s.decode('utf-8')或者s.decode('gbk')

### 索引和分片

索引分为正向和反向

|  H | e  |  l | l  |  o |
|:--:|:--:|:--:|:--:|:--:|
| s[0] | s[1] | s[2] | s[3] | s[4]|
| s[-5] | s[-4] | s[-3] | s[-2] | s[-1]|

分片使用中括号和如下格式:
- s[i:j:k]
- i: 开始位置下标, j: 结束位置下标, 不包括j位置的字符, k: 步长
- i,j,k都可以省略, i默认0或-1, j默认到最后一个字符, k默认1
- 一定要注意, 不管k是正还是负, 都不包括j位置的字符
- i和j都可以超过字符串的长度

示例:

```python
>>> s='hello'
>>> print(s[0:4:1])
hell
>>> print(s[0:5:2])
hlo

>>> print(s[-1:-5:-1])
olle
>>> print(s[-1:-6:-1])
olleh

>>> print(s[0:len(s)])
hello
>>> print(s[-len(s):-1])
hell

# 都省了
>>> print(s[:])
hello
>>> print(s[::2])
hlo
>>> print(s[::-1])
olleh
# 省略了步长
>>> print(s[:-1])
hell
```

### 字符串连接
使用`+`连接, 如果不是字符串类型, 使用str()函数或者repr()函数转换成字符串

字符串不可用下标修改, 所以可以用连接操作改某个字符

```python
>>> s='abcd'
>>> s=s[0]+9+s[2]
>>> s
'a9cd'
```

可以使用`*`进行重复连接, n\*s或者s\*n  
可以使用+=或者\*=这样的操作

如果是连加操作, +的效率很低, 可以使用s.join()函数代替, s是分隔符

```python
>>> '-'.join(['Python','and','Program'])
'Python-and-Program'
```

### 字符串比较
- 单子符就直接比较ASCII码
- 相同长度从左到右比较ASCII
- 不同长度补全空格再从左到右比较ASCII

### 字符串成员关系
用`in`和`not in`: 字符串1 [not] in 字符串2  
返回True或者False

### 字符串常用方法
字符串是一个类, 有许多类方法.这里只记常用的  

#### 大小写和对齐
- s.upper()/lower()
- s.ljust(width, [fillchar]): 输出width长度, 左对齐, 不足部分fillchar补齐, 默认空格
- s.rjust()/center(): 参数同上, 右对齐和居中

#### 字符串搜索
- s.find(sub,[start,[end]]): 返回sub的第一个字符编号, 没有返回-1
- s.index(sub,[start,[end]]): 和find相同, 不过没有时,返回一个错误
- s.rfind(sub,[start,[end]]): 从后往前找sub,第一个字符编号,没有返回-1
- s.rindex(sub,[start,[end]])
- s.count(substr,[start,[end]]): 返回出现次数
- s.startswith(prefix,[start,[end]])
- s.endswith(suffix,[start,[end]])

#### 字符串替换
- s.replace(oldstr,newstr,[count]): old替换为new
- s.strip([chars]): 默认去前后空格, s前后的chars中的字符去掉
- s.lstrip([chars])/rstrip()
- s.expandtabs([tabsize]): s中tab替换为空格, tabsize为空格的数量, 默认8个

#### 字符串拆分
- s.split([sep,[maxsplit]]): sep:分隔符,默认空格, 按sep把s拆成一个列表, maxsplit:拆分次数,默认-1无限制
- s.rsplit([sep,[maxsplit]]): 从右侧拆
- s.splitlines([keepends]): 按行拆成列表, keepends为true: 保留每行分隔符
- s.partition(sub): 拆成三个元素的元组(sub左, sub, sub右), 如果没sub, 返回(s, '', '')
- s.rpartition(sub): 从右边找sub

#### 字符串连接
- s.join(seq): seq为序列, s为分隔符
通常和list(s)结合使用, 修改string某个字符

```python
>>> s='abcd'
>>> s=list(s)
>>> s[0]='t'
>>> s="".join(s)
>>> s
'tbcd'
```

#### 字符串类型判断
- s.isalnum(): 字母和数字, 至少一个字符
- s.isalpha(): 全字母,至少一个字符
- s.isdigit(): 全数字,至少一个字符
- s.isspace(): 全空格,至少一个字符
- s.islower()/isupper()/istitle()

### 字节类型bytes
bytes和string不一样, 可以用string.encode()方法和bytes.decode()方法进行两者转换.  
bytes可以使用bytearray(by)的方式将bytes转换为bytearray对象   
bytes对象不可修改, bytearray可修改

## 3.2 正则表达式
太多了, 附上[菜鸟教程](https://www.runoob.com/python/python-reg-expressions.html)

{% cq %}
**end CH3**
{% endcq %}

# CH4. 复合数据类型
## 4.1 序列
序列包括**字符串, 列表, 元组**, 按照位置编号存取

### 4.1.1 通用操作
- 索引, 用下标访问, num[0], 支持反向索引, 从-1开始
- 分片, 注意不改变原序列, 而是产生一个新序列, 下面是常用分片用法

```python
>>> num=[1,2,3,4,5,6,7,8,9,10]
# 常见用法
>>> num[3:6]
[4, 5, 6]

# 可以从末尾计数, 负数索引, 效果相同
>>> num[7:10]
[8, 9, 10]
>>> num[-3:]
[8, 9, 10]

# 省略部分下标
>>> num[:3]
[1, 2, 3]
>>> num[:]
[1, 2, 3, 4, 5, 6, 7, 8, 9, 10]

# 设置步长
>>> num[:len(num):2]
[1, 3, 5, 7, 9]

# 开始索引可以超过列表长度, 视作从最后一个开始
>>> num[11:0:-1]
[10, 9, 8, 7, 6, 5, 4, 3, 2]
>>> num[8:0:-1]
[9, 8, 7, 6, 5, 4, 3, 2]

# 如果步长为负数, 开始索引必须大于终止索引
>>> num[0:10:-1]
[]

# 从id可以看出分片是新序列, 和赋值不一样
>>> x=[1,2,3]
>>> y=x[:]
>>> id(x),id(y)
(140158585788768, 140158585103696)
>>> z=x
>>> id(x),id(z)
(140158585788768, 140158585788768)

```

- 同类型序列可用+直接相加
- 可以用\*进行乘法, 生成新序列, n<1会返回一个空列表

```python
>>> s=[None]*10
>>> s
[None, None, None, None, None, None, None, None, None, None]
```

- 字符串部分介绍过, 可以直接比较, 同样适用于列表和元组, 但是列表和元组不可以比较
- 用in和notin判断成员资格
- 常用函数: len(), max(), min(), sum():要求元素必须为数值
- 常用函数: s.count(x), s.index(x)
- enumerate(seq): 接收可迭代对象, 返回enumerate对象, 由每个元素的索引和值组成(索引, 值)
- zip(seq1,seq2..seqn)
- sorted(iterable, key=None, reverse=False): 返回排序后的列表, 返回的是副本,不改变源. reverse=False升序, reverse=True降序
- reversed(iterable): 逆序排列, 返回新的对象
- all(seq): seq元素都是True则返回True
- any(seq): seq任一为True则返回True
- 序列拆分: 变量个数不一致时, 要用\*, \*只能有1个

```python
>>> x=[1,2,3]
>>> a,b,c=x
>>> a,b,c
(1, 2, 3)
>>> a,*b=x
>>> a,b
(1, [2, 3])
>>> *a,b=x
>>> a,b
([1, 2], 3)
```

### 4.1.2 列表专用操作
- 列表可修改, 可元素赋值
- del seq[index]/ del seq : 删除某元素或对象
- 分片赋值: 可以做到增加, 删除, 修改

```python
>>> name=list('Perl')

# 修改
>>> name[2:]=list('ar')
>>> name
['P', 'e', 'a', 'r']

# 修改并增加, 后面那部分可以比前面的长
>>> name[1:]=list('ython')
>>> name
['P', 'y', 't', 'h', 'o', 'n']

# 可以插入序列
>>> name=[1,4]
>>> name[1:1]=[2,3]
>>> name
[1, 2, 3, 4]

# 删除部分元素
>>> name[1:3]=[]
>>> name
[1, 4]

# 如果步长不等于1, 那么前后数量必须相同
>>> name[:4:2]=[2,3]
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
ValueError: attempt to assign sequence of size 2 to extended slice of size 1
```

- 列表解析, 表达式可以任意, for循环可嵌套, if可选

```python
>>> [i for i in range(4)]
[0, 1, 2, 3]
>>> [ord(x) for x in 'python']
[112, 121, 116, 104, 111, 110]
>>> [(x,y) for x in range(5) if x%2==0 for y in range(5) if y%2==1]
[(0, 1), (0, 3), (2, 1), (2, 3), (4, 1), (4, 3)]
```

- 以下方法都是在源列表上操作, 会修改原列表
 1. s.append(x): 末尾添加单个元素
 1. s.extend(seq): 在s末尾添加seq所有元素
 1. s.sort(reverse=False, key=None) 和 sorted(seq)这个函数功能一样, 但是sorted是返回副本
 1. s.reverse(): 同reversed(seq)
 1. s.pop([index]): 删除并返回index位置的元素, 默认最后一个, 超范围报错
 1. s.insert(index,x): index处插入x, index大于列表长度,则插在最后
 1. s.remove(x): 如果有多个, 只删除第一个

### 4.1.3 元组和列表
都是有序序列, 常可替换

**区别:**
1. 元组不可变
2. 元组小括号,列表中括号
3. 元组可以在字典中作为key

元组不可变, 但是可以用元组创建新元组   
如果元组内元素为可变的, 如列表或字符串, 则可以修改  
使用 list()和tuple()函数相互转换

## 4.2 字典和集合
字典主要是元素的检索, 插入和删除.   
集合主要考虑集合间的并, 交和差操作.  

### 4.2.1 字典
注意点:
- 关键是是不可变的, 如果元组里有可变类型比如列表, 就不可以作为key. python对key进行哈希, 根据计算结果决定存储地址, 所以可以用hash()函数判断是否可以作为key
- 字典是可变类型, 可以在原处增长或者缩短, 无需生成副本.
- 字典是异构的, 可以抱含任何类型数据: 元素可以是字典,列表,元组, 支持任意层次嵌套.
- {}和dict()是空字典

```python
>>> d1=dict()
>>> d1
{}

# dict参数可以是列表和元组
>>> d2=dict((['x',1],['y',2]))
>>> d2
{'x': 1, 'y': 2}

>>> d2=dict([['x',1],['y',2]])
>>> d2
{'x': 1, 'y': 2}

# 可以直接指定key和val
>>> d3=dict(name='allen',age=15)
>>> d3
{'name': 'allen', 'age': 15}
```

- 更新和添加: 字典的更新直接用dict[key]=new_val就可以, 如果没有就直接增加一个新元素
- 删除: del 字典名[key] 或者 del 字典名
- 判断key是否存在: key in / not in dict
- len(dict)
- 只支持==和！=
- list(dict)返回key

常用方法：
- fromkeys(序列[,值])
- keys(), values(), items()
- d.copy(), d.clear(), d.pop(key), d.popitem()
- d.get(key[,value]): 返回key对应的val值, 若没有, 则返回value
- d.setdefault(key,[value]): 存在则返回值, 不存在则插入key:value, value默认none
- d2.update(d1): d1合并到d2, d1不变
- has_key(key): 存在key返回true, 用in和not in就可以

### 4.2.2 集合
#### 注意点
1. 空集合是set()
2. 用大括号
3. 字符串,列表, 元组都可以转set
4. set()创建可变集合, frozenset()创建不可变集合
5. 可变集合的元素只能是数值, 字符串, 元组这些不可修改的, 可修改的列表,字典和可变集合都不可作为元素

#### 集合的运算
- 并： |, |=, s1.union(s2...sn)
- 交： & , s1.intersection(s2...sn)
- 差： -, s1.difference(s2...sn)
- 对称差： ^, s1.symmetric_difference(s2...sn)
- ==
- ！=
- s1\<s2: s1是s2真子集,  
- s1\<=s2: s1是s2子集, s1.issubset(s2)
- s1\>s2: 超集
- s1\>=s2: 相当于s1.issuperset(s2)
- s.copy()

#### 适用于可变集合的方法
- s.add(x)
- s.remove(x): 不存在会报错; s.discard(x): 不存在不报错
- s.pop(): 删除任意一个元素并返回
- s.clear()


- s.update(s1,s2...): s=s|s1|s2...sn
- s.intersection_update(s1,s2..sn): s=s&s1&s2..&sn
- s.difference_update(s1,s2..s2): s=s-s1-s2..-sn
- s.symmetric_difference_update(s1): s=s^s1

{% cq %}
**end CH4**
{% endcq %}

# CH5. 函数
和其他的语言一样, 函数先定义再使用

## 5.1 参数
参数传递的方式是"值传递"

**要注意的是:**
- 形参重新赋值重新分配对象, 指向别的对象, 不会改变实参
- 如果仅修改形参内容, 比如列表, 字典这样的可修改对象, 则会影响实参

### 参数类型
- 位置参数: 参数按照位置对应
- 关键字参数: 形参名=值

```python
def mykey(x,y):
  print(x, y)
mykey(y=10, x=1)
```

- 默认参数: 形参=默认值, 必须在参数列表右边, 在可变长度参数之前
- 可变长度参数: 参数数量不固定时, 针对元组和字典两种. 其他所有类型参数都必须在可变长度参数之前

```python
# 元组一个*
def myvarl(*t):
  print(t)
myvarl(1,2,3)

# 字典两个*
def myvarl(**t):
  print(t)
myvarl('y'=1,'x'=2,'z'=3)

```

## 5.2 匿名函数

```python
# 参数也可以使用默认值, 返回值是表达式结果
lambda [参数1[,参数2,...,参数n]]: 表达式

# 例如
lambda x,y: x+y

# 调用
>>> f=lambda x,y: x+y
>>> f(5,10)
15

# 把匿名函数作为普通函数返回值
def f():
  return lambda x,y:x+y

# 匿名函数作为序列或者字典元素
列表名=[匿名1, 匿名2,...匿名n]
# 调用
列表名[index](参数)
```

## 5.3 装饰器
这里只记录一些注意点, 比较全面的看书上或者[Python 函数装饰器|菜鸟教程](https://www.runoob.com/w3cnote/python-func-decorators.html), 这个讲的很全

- 概念: 装饰器是用来包装函数的函数, 用来为已经存在的函数添加额外功能
- 返回对象: 装饰器输入是被装饰的函数对象, 返回新的函数对象.
- 调用: 使用@调用装饰器
- 调用的时候,本质上, 是将对象名重新指向一个新返回的函数, 达到修改函数的目的
- 返回: 装饰器最里面那层, 要返回被装饰的函数
- 在装饰器中,返回值, 例如 return func, func后面没有(), 加上()表示调用.
- 外层里层函数关系: 最外面的那层函数返回值, 是里面一层函数的函数名
- 多重装饰器: 执行顺序是从下往上, 最后执行被装饰函数内容

## 5.4 \_\_name\_\_
主动调用\_\_name\_\_值为\_\_main\_\_, 被导包时值为模块名
把一个程序文件既当作模块, 有可以主动调用时, 程序入口前加if:

```python
def test():
  pass
if __name__=='__main__':
  test()
```

{% cq %}
**end CH5**
{% endcq %}

# CH6. 类

## 6.1 属性

| |格式|访问控制|添加|修改|作用域|
|:-|:-|:-|:-|:-|:-|
|私有属性|类里函数外, 以\_\_开头的属性|只能通过类方法中使用self.\_\_name使用,类外无法直接访问|只能在类里面添加|在类方法里用self修改, 不影响其他对象|单个|
|类属性|直接在类里函数外定义的非私有属性,所有对象公有|类外通过类名和实例对象(不推荐)调用|类外可以通过类名.类属性添加|类外:类名修改,类里:类方法|所有|
|实例属性|在\_\_init\_\_里面定义或者通过实例对象引用定义|类外只有实例对象可以访问,类里面可以在类方法中调用|类里面在init方法中添加,类外使用实例对象.实例属性添加|用self修改或实例对象.实例属性修改|单个|

注意点:
- 千万不要尝试实例对象创建类属性, 会自动创建一个同名实例属性,并屏蔽掉类属性
- 不可以用实例对象直接修改类属性, 要用类方法修改
- 使用del 删除实例属性和实例对象

## 6.2 方法

| |特征|调用|和属性关系|
|:-|:-|:-|:-|
|构造方法|\_\_init\_\_(self)|生成对象自动调用|里面用self定义的都是实例属性|
|析构方法|\_\_del\_\_(self)|释放对象自动调用|可以在里面进行释放资源的操作|
|类方法|@classmethod修饰器,第一个参数为cls|实例对象和类对象访问|引用的是类属性
|实例方法|第一个参数为self|只能实例对象调用|实例属性优先级高于类属性,实例方法中实例属性会屏蔽类属性|
|静态方法|@staticmethod修饰器,无需多定义参数|类名和实例对象都可以访问|不绑定类或者实例, 引用类属性需要使用类名引用|

注意点:
- 静态方法和实例方法不同的是, 静态方法不会自动绑定到类上, 类方法会自动绑定. 静态方法不需要引用类或者实例
- 静态方法很像我们在类外定义的函数，只不过静态方法可以通过类或者实例来调用而已。
- 实例方法中引用属性, 若存在同名类属性和实例属性, 则引用实例属性, 若无实例属性, 则引用类属性
- 实例方法中修改属性, 若存在同名类属性和实例属性, 则修改实例属性, 若无实例属性, 则创建新实例属性

## 6.3 继承

```python
# 可以多重继承
class 子类名(父类名1,父类名2,..):
  类体
```

注: 
- 如果子类中没有重新定义构造方法, 会执行第一个父类的构造方法
- 使用内置方法issubclass(Child, Parent)判断一个类是否是另一个类的子孙类, 返回True或者False
- 小心使用多重继承, 能不用就不用, 可以用组合
- 多重继承函数的搜索顺序是广度优先, 从左到右找, 没有就下一个
- super(Child,self).\_\_init\_\_()相当于Parent.\_\_init\_\_(self)
- 子类继承父类所有属性和方法, 如果没有覆盖掉, 则调用父类方法
- 可以使用父类名Parent.fun(self)或者super(Child,self).fun()来调用父类方法, 但是一定要传self


对于init函数:
- 如果子类没有定义自己的初始化函数，父类的初始化函数会被默认调用；但是如果要实例化子类的对象，则只能传入父类的初始化函数对应的参数，否则会出错。
- 如果子类定义了自己的初始化函数，而在子类中没有显示调用父类的初始化函数，则父类的属性不会被初始化
- 如果子类定义了自己的初始化函数，在子类中显示调用父类，子类和父类的属性都会被初始化

详细的看下面这几个链接:
　　[python中的类，对象，方法，属性初认识（一）](http://blog.csdn.net/brucewong0516/article/details/79114977)    
　　[详解类class的属性：类数据属性、实例数据属性、特殊的类属性、属性隐藏（二）](http://blog.csdn.net/brucewong0516/article/details/79118703)   
　　[详解类class的方法：实例方法、类方法、静态方法（三）](http://blog.csdn.net/brucewong0516/article/details/79119551)   
　　[详解类class的访问控制：单下划线与双下划线（四）](http://blog.csdn.net/brucewong0516/article/details/79120841)   
　　[详解类class的继承、__init__初始化、super方法](https://blog.csdn.net/brucewong0516/article/details/79121179)

# CH7. 文件操作

具体的可看[Python 文件I/O | 菜鸟教程](https://www.runoob.com/python/python-files-io.html)和[python 文件操作，最全的一个 | CSDN](https://blog.csdn.net/m0_38059843/article/details/78240835)

# CH8. 注意点
1. 整除是//不是/, 看1.4.1
2. 判断在不在用in
3. 正则的使用
4. \$是匹配结尾,如果是一行里面的不要用\$符号,匹配所有单词只需要`[a-zA-Z]+`就可以
4. read()没有参数就是读全部
3. write()写字符串的时候可以用字符串.format()方法格式化输出到文件
4. open()之后要记得close()
4. dict.items()之后要使用list()转换一下才可以用
4. 没有返回值的函数说明是要在序列内部修改, 注意不要修改引用, 只改内容
1. 文件操作要记得使用try-except
1. **大量的数据**统计要最好用字典,字典使用哈希会很快 
