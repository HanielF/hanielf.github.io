---
title: NowCoder-特殊乘法
tags:
  - NowCoder
  - Algorithm
  - Easy
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-21 15:45:06
urlname: special-multiplication
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/a5edebf0622045468436c74c3a34240f?tpId=40&tqId=21349&tPage=1&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
补之前的

写个算法，对2个小于1000000000的输入，求结果。 特殊乘法举例：123 * 45 = 1\*4 +1\*5 +2\*4 +2\*5 +3\*4+3\*5

输入可能有多组数据，对于每一组数据，输出Input中的两个数按照题目要求的方法进行运算后得到的结果。

输入两个小于1000000000的数

### Examples:
**Input:** 123 45
**Output:** 54

{% endnote %}
<!--more-->

## Solutions
- 直接读入字符串，然后计算的时候字符转数字计算


## C++ Codes

```C++
#include<iostream>
#include<string>
using namespace std;

int main(){
    string a, b;
    while(cin>>a>>b){
        int res = 0;
        for(int i=0;i<a.length();i++){
            for(int j=0;j<b.length();j++){
                res += (a[i]-'0')*(b[j]-'0');
            }
        }
        cout<<res<<endl;
    }
    return 0;
}
```

------
