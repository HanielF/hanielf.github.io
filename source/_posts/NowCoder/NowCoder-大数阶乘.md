---
title: NowCoder-大数阶乘
tags:
  - NowCoder
  - Algorithm
  - Hard
  - BigNumber
  - Factorial
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-21 23:47:25
urlname: factorial-of-bignumber
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/f54d8e6de61e4efb8cce3eebfd0e0daa?tpId=40&tqId=21355&tPage=2&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
补之前的

输入一个正整数N，输出N的阶乘。0\<=N\<=1000

### Examples:
**Input:**
> 4
> 15
 
**Output:**
> 24
> 1307674368000

{% endnote %}
<!--more-->

## Solutions
- 因为数字太大可能超出long long表示范围，所以用大正数乘法做
- 乘的过程有点不太一样，看代码


## C++ Codes

```C++
#include<stdio.h>
#define width 3000

int main()
{
    //i,j: 循环变量
    int i,j;

    //k: 上一次进位,t: 最高位下标
    int k,t;
    int N;

    //存放结果，从d[0]是个位，从低位开始存
    int d[width];

    //a*b，N是a，b是已经乘好的结果
    while(scanf("%d",&N)!=EOF)
    {
        t=0;                    //t是位数-1
        //给数组初始化为零
        for(i=0;i<width;i++)    
            d[i]=0;
        d[0]=1;                 //个位初始化为1

        for(i=1;i<=N;i++)       //从1到N进行阶乘
        {
            k=0;
            //这里直接用b的每位数和整个a相乘，而不是b的每位数和a相乘后再加
            //从个位开始往高位运算
            for(j=0;j<=t;j++)
            {
                int tmp = d[j]*i+k;     //第j位乘以i加上后一位运算得到的k作为tmp
                k=tmp/10;               //tmp除以10得到k
                d[j]=tmp%10;            //tmp取余得到运算后第j位的值
            }

            //由于是直接和a相乘，所以k可能比较大，不止一位
            //k!=0说明要向高位进位
            while(k!=0){
                d[++t]=k%10;
                k=k/10;
            }
        }

        //从个位开始输出各位数字
        for(i=t;i>=0;i--)   
            printf("%d",d[i]);
        printf("\n");
    }
    return 0;
}
```


------
