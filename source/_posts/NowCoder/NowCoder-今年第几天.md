---
title: NowCoder-今年第几天
tags:
  - NowCoder
  - Algorithm
  - Map
  - Easy
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-21 15:50:03
urlname: day-of-the-year
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/ae7e58fe24b14d1386e13e7d70eaf04d?tpId=40&tqId=21350&tPage=1&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
补之前的

输入年、月、日，计算该天是本年的第几天。
输入包括三个整数年(1<=Y<=3000)、月(1<=M<=12)、日(1<=D<=31)。
输出一个整数，代表Input中的年、月、日对应本年的第几天。

### Examples:
**Input:**
1990 9 20
2000 5 1
**Output:**
263
122

{% endnote %}
<!--more-->

## Solutions
- 用数组代替map表示每个月的天数，然后判断是否闰年，然后再累加天数


## C++ Codes

```C++
#include<iostream>
#include<cmath>
#include<string>
using namespace std;

bool judge(int year){
    //判断闰年
    if((year%4==0 && year%400!=0) || (year%4==0 && year%100==0)){
        return true;
    }
    return false;
}

int main(){
    int year, mon, day;
    int days[13]={0, 31,28, 31, 30, 31, 30, 31, 31,30, 31, 30, 31};
    while(cin>>year>>mon>>day){
        int res = 0;
        if(judge(year)) days[2]=29;
        else days[2]=28;
        
        for(int i=1;i<mon;i++){
            res+=days[i];
        }
        res+=day;
        cout<<res<<endl;
    }
    return 0;
}
```

------
