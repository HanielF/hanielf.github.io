---
title: LeetCode-029-Divide Two Integers
tags:
  - BigNumber
  - ShiftOperation
  - LeetCode
  - Medium
  - Math
  - Algorithm
categories:
  - LeetCode
comments: true
mathjax: false
date: 2019-10-31 11:54:20
urlname: leetcode-028-divide-two-integers
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://leetcode-cn.com/problems/divide-two-integers/submissions/)   
Given two integers dividend and divisor, divide two integers without using multiplication, division and mod operator.

Return the quotient after dividing dividend by divisor.

The integer division should truncate toward zero.

### Examples:
**Input:**
> dividend = 10, divisor = 3
> dividend = 7, divisor = -3

**Output:**
> 3
> -2

{% endnote %}
<!--more-->

## Solutions
- 刚开始就是想到用减法，把被除数和除数都变成正数，然后一个个减。这样虽然能做，但是复杂度太高，总超时
- 题解中都用到了左移和右移这样的方法，看了之后按照自己的理解，将除数的倍数存在向量中，一直到最接近被除数。比如被除数是23，除数是5，，那么向量中就存放`5<<0=5，5<<1=10，10<<1=20`。分别对应的是`1<<0=1，1<<1=2 和 1<<2=4`。最后的结果计算是，`23>20，res+4，23-20=3, 3<10, 3<5`，都不加，结果就是`4`


## C++ Codes

```C++
class Solution {
public:
    int divide(int dividend, int divisor) {
        //边界
        if (divisor == 0 || (dividend == INT_MIN && divisor == -1)) return INT_MAX; 
        if(dividend == divisor) return 1;
        if(divisor == INT_MIN) return 0;
        
        int symbolFlag= (dividend<0)==(divisor<0);    //符号，同号为1
        int minFlag = dividend==INT_MIN;    //被除数为INT_MIN标识
        
        //如果负数为2147483648，就先加上一个除数绝对值
        if(dividend==INT_MIN) dividend+=abs(divisor); 
        
        //全部变成正数
        divisor=abs(divisor);
        dividend=abs(dividend);
        
        vector<int> nums;
        int result=0, tmp=divisor;
        
        //直到divisor大于dividend
        while(tmp<=dividend){
            nums.push_back(tmp);
            if(INT_MAX-tmp<tmp) break;  //如果右移溢出，那肯定也大于divideng
            tmp<<=1;    //除数乘以2
        }
        
        //从后往前减
        for(int i=nums.size()-1;i>=0;i--){
            if(dividend>=nums[i]){
                result += 1<<i;     //result加上对应的数量
                dividend -= nums[i];
            }
        }
        
        //处理结果
        if(minFlag){    //绝对值需加上1
            if(symbolFlag){ //同号
                result = result==INT_MAX? result : result+1;
            } else{ //异号
                result = -result-1;
            }
        } else{
            result = symbolFlag?result:-result;
        }
        
        return result;
    }
};
```

## Python Codes
题解：[小学生都会的列竖式算除法](https://leetcode-cn.com/problems/divide-two-integers/solution/xiao-xue-sheng-du-hui-de-lie-shu-shi-suan-chu-fa-b/)

```python
def divide(self, dividend: int, divisor: int) -> int:
    sign = (dividend > 0) ^ (divisor > 0)
    dividend = abs(dividend)
    divisor = abs(divisor)
    count = 0
    #把除数不断左移，直到它大于被除数
    while dividend >= divisor:
        count += 1
        divisor <<= 1
    result = 0
    while count > 0:
        count -= 1
        divisor >>= 1
        if divisor <= dividend:
            result += 1 << count #这里的移位运算是把二进制（第count+1位上的1）转换为十进制
            dividend -= divisor
    if sign: result = -result
    return result if -(1<<31) <= result <= (1<<31)-1 else (1<<31)-1 

```

## 大整数进制转换
附上自己做的思维导图截图

{% asset_img 1.png  %}

## 大整数乘法
附上自己做的思维导图截图

{% asset_img 2.png %}

## 总结
- 大整数运算，时间复杂度太高的，可以想想移位运算和二进制模拟。甚至是字符串。
- 联想到大整数乘除法。

------
