---
title: NowCoder-手机键盘
tags:
  - NowCoder
  - Algorithm
  - Map
  - Easy
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-12 20:59:22
urlname: keyboard-of-phone
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/20082c12f1ec43b29cd27c805cd476cd?tpId=40&tqId=21337&tPage=1&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
按照手机键盘输入字母的方式，计算所花费的时间 如：a,b,c都在“1”键上，输入a只需要按一次，输入c需要连续按三次。 如果连续两个字符不在同一个按键上，则可直接按，如：ad需要按两下，kz需要按6下 如果连续两字符在同一个按键上，则两个按键之间需要等一段时间，如ac，在按了a之后，需要等一会儿才能按c。 现在假设每按一次需要花费一个时间段，等待时间需要花费两个时间段。 现在给出一串字符，需要计算出它所需要花费的时间。

### Examples:
**Input:**    
一个长度不大于100的字符串，其中只有手机按键上有的小写字母
**Output:**   
输入可能包括多组数据，对于每组数据，输出按出Input所给字符串所需要的时间
**Input:**  
bob
www   
**Output:**   
7
7
{% endnote %}
<!--more-->

## Solutions
- 简单的用数组, 作为map使用, 下标就是key, 数组值就是value

## C++ Codes

```C++
#include<iostream>
#include<string>
using namespace std;

int main(){
    //十位代表按键数，各位代表按下次数
    int mp[26]={11,12,13,21,22,23,31,32,33,41,42,43,51,52,53,61,62,63,64,71,72,73,81,82,83,84};
    string tmp;
    while(cin>>tmp){
        int res = 0;
        for(int i=0;i<tmp.length();i++){
            if(i>0 && mp[tmp[i]-'a']/10==mp[tmp[i-1]-'a']/10){
                res+=2;
            }
            res += mp[tmp[i]-'a']%10;
        }
        cout<<res<<endl;
    }
    return 0;
}
```

## 总结
- 用好map很关键

------
