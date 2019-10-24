---
title: NowCoder-玛雅人的密码
tags:
  - NowCoder
  - Algorithm
  - BFS
  - Recursive
  - Medium
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-17 21:02:23
urlname: maya-code
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/761fc1e2f03742c2aa929c19ba96dbb0?tpId=40&tqId=21343&tPage=1&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
补之前的

玛雅人有一种密码，如果字符串中出现连续的2012四个数字就能解开密码。给一个长度为N的字符串，（2=\<N\<=13）该字符串中只含有0,1,2三种数字，问这个字符串要移位几次才能解开密码，每次只能移动相邻的两个数字。例如02120经过一次移位，可以得到20120,01220,02210,02102，其中20120符合要求，因此输出为1.如果无论移位多少次都解不开密码，输出-1。
### Examples:
**Input:**
输入包含多组测试数据，每组测试数据由两行组成。
第一行为一个整数N，代表字符串的长度（2<=N<=13）。
第二行为一个仅由0、1、2组成的，长度为N的字符串。
**Output:**
对于每组测试数据，若可以解出密码，输出最少的移位次数；否则输出-1。

{% endnote %}
<!--more-->

## Solutions
- 看到这种要求最少怎么怎么样，最低什么什么的，一般可以先考虑BFS
- 可以用队列完成BFS


## C++ Codes

```C++
/*
 * 玛雅人的密码，每次只能移动相邻的两个数字，问最低移动多少次才可以得到要的数字
 * 尝试1：BFS
 */
#include<iostream>
#include<string>
#include<vector>
#include<algorithm>
#include<queue>
#include<cstring>
using namespace std;

int main(){
  int N;
  string code;
  while(cin>>N>>code){
    auto pos2 = code.find_first_of("2");
    auto pos22 = code.find_last_of("2");
    //条件判断
    if(N<4 || pos2==-1 || pos22==-1 || pos2 == pos22 || code.find("0")==-1 || code.find("1")==-1) {
      cout<<-1<<endl;
      continue;
    }

    //使用队列完成BFS
    queue<string> mq;
    mq.push(code);
    int res = 0;
    bool flag = false;  //是否找到的标志
    while(!mq.empty()){
      int n = mq.size();
      //这个for循环代表一次，经过几次完整的for循环，结果就是几
      for(int i=0;i<n;i++){
        string first = mq.front();
        if(first.find("2012")!=-1) {
          //cout<<first<<endl;
          cout<<res<<endl;
          flag = true;
          break;
        }
        mq.pop();

        //只和后面元素交换，不和前面的交换，会重复
        //把交换一次后的所有串，都添加到队列中，供下个for循环判断
        for(int i=0;i<N-1;i++){
          string strf=first;
          strf[i]=first[i+1];
          strf[i+1]=first[i];
          mq.push(strf);
        }
      }
      
      if(flag) break;
      res++;
    }
  }
  return 0;
}
```


------
