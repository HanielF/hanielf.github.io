---
title: NowCoder-10进制和2进制
tags:
  - NowCoder
  - Algorithm
  - Scale
  - Binary
  - Medium
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-22 12:05:17
urlname: dec-vs-bin
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/fd972d5d5cf04dd4bb4e5f027d4fc11e?tpId=40&tqId=21357&tPage=2&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
补之前的

对于一个十进制数A，将A转换为二进制数，然后按位逆序排列，再转换为十进制数B，我们乘B为A的二进制逆序数。     例如对于十进制数173，它的二进制形式为10101101，逆序排列得到10110101，其十进制数为181，181即为173的二进制逆序数。

输入一个1000位(即10^999)以内的十进制数。
输出十进制的二进制逆序数

### Examples:
**Input:**
> 173
 
**Output:**
> 181

{% endnote %}
<!--more-->

## Solutions
- 顺序很清楚，先转换成二进制，然后逆序，然后转换成十进制
- 可以写一个通用的进制转换函数

## C++ Codes

```C++
/*
 * 进制转换
 */

#include "stdio.h"
#include "string.h"
 
//进制转换函数
void convert(int m, char* original, int n, char* conversion){
    int len = strlen(original),i,j,carry,k = 0;

    //i是最高位，从最高位开始将每一位除以2，将余数传递给下一位
    for(i = 0; i < len;){
        carry = 0;

        //这一块，数字和字符的处理没搞清楚,其中carry是参与运算的，应该作为数字。
        //而original[j]则是保存当前的运算结果，应该是字符型的
        //本质是模拟除法运算,类比手写的进制转换
        for(j = i; j < len; j++){ 
            original[j] = original[j] - '0' + carry*m;
            carry = (original[j]) % n ;
            original[j] = (original[j]) / n + '0';
        }
        //最后的余数
        conversion[k++] =(char) (carry + '0');

        //精髓，在大进制转小进制时，
        //当前的最高位orignal[i]可能无法循环一次就变成0
        //小进制转大进制最高位可能一次循环就
        //使得高几级位变成0
        while(original[i] == '0')i++; 
    }
    conversion[k] = '\0';
}
 
int main(){
    char dec[1001],bin[4000];
    int i,j;
    char temp;

    while(scanf("%s",dec) != EOF){
        convert(10,dec,2,bin);
        convert(2,bin,10,dec);

        //将字符串反序
        for(j = strlen(dec)-1,i=0;i<j;){
                temp = dec[i];
                dec[i++] = dec[j];
                dec[j--] = temp;
            }
        printf("%s",dec);
    }
    return 0;
}
```

## 总结
- 进制转换的方法大概都是模拟手写进制转换，只是写法可能有不同

------
