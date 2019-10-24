---
title: NowCoder-代理服务器
tags:
  - NowCoder
  - Algorithm
  - Greedy
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-12 19:15:36
urlname:
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/1284469ee94a4762848816a42281a9e0?tpId=40&tqId=21335&tPage=1&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
使用代理服务器能够在一定程度上隐藏客户端信息，从而保护用户在互联网上的隐私。我们知道n个代理服务器的IP地址，现在要用它们去访问m个服务器。这 m 个服务器的 IP 地址和访问顺序也已经给出。系统在同一时刻只能使用一个代理服务器，并要求不能用代理服务器去访问和它 IP地址相同的服务器（不然客户端信息很有可能就会被泄露）。在这样的条件下，找到一种使用代理服务器的方案，使得代理服务器切换的次数尽可能得少。

每个测试数据包括 n + m + 2 行。
第 1 行只包含一个整数 n，表示代理服务器的个数。
第 2行至第n + 1行每行是一个字符串，表示代理服务器的 IP地址。这n个 IP地址两两不相同。
第 n + 2 行只包含一个整数 m，表示要访问的服务器的个数。
第 n + 3 行至第 n + m + 2 行每行是一个字符串，表示要访问的服务器的 IP 地址，按照访问的顺序给出。
每个字符串都是合法的IP地址，形式为“xxx.yyy.zzz.www”，其中任何一部分均是0–255之间的整数。输入数据的任何一行都不包含空格字符。
 其中，1<=n<=1000，1<=m<=5000。
### Examples:
**Input:**   
3
166.111.4.100
162.105.131.113
202.112.128.69
6
72.14.235.104
166.111.4.100
207.46.19.190
202.112.128.69
162.105.131.113
118.214.226.52

**Output:**    
1

{% endnote %}
<!--more-->

## Solutions
- 用贪心的思想, 每一步都是找能走的最远的那个ip, 然后一直到结束, 记录切换次数就好了
- 下面的for循环就是找一次的过程, n个ip都试一下, 找能连接最多的, 就是max个, 然后pos就设置为那个位置, 一直增加pos, 到所有的服务器被访问完


## C++ Codes

```C++
/*
 * 代理服务器
 * 贪心
 */
#include<iostream>
#include<string>
#include<algorithm>
#include<vector>
using namespace std;

int greedy(vector<string> &agency, int n, vector<string> &server, int m){
  //如果只有一个代理, 那找不全就返回-1了
  if(n==1){
    auto it = find(server.begin(), server.end(), agency[0]);
    if(it!=server.end()) return -1;
    else return 0;
  }
  
  int pos = 0;
  int res = 0;
  int max = 0;
  //while退出条件是找完
  while(pos<m){
    //每次for循环结束就是走一步
    for(int i=0;i<n;i++){
      //找下一个和代理服务器ip一样的服务器
      auto it = find(server.begin()+pos, server.end(), agency[i]);
      //如果没找到,就不用切换了, 直接返回结果
      if(it==server.end()) return res;
      else{
        if((it-server.begin())>max) max = it-server.begin();
      }
    }
    //设置下一次找的时候开始的地方
    pos = max;
    res++;
  }

  return res;
}

int main(){
  int n, m;
  while(cin>>n){
    //保存代理服务器ip
    vector<string> agency(n);
    for(int i=0;i<n;i++) cin>>agency[i];
    //保存目标服务器ip
    cin>>m;
    vector<string> server(m);
    for(int i=0;i<m;i++) cin>>server[i];
    //贪心
    int res = greedy(agency, n, server, m);
    cout<<res<<endl;
  }
  return 0;
}
```

## 总结
- 这道题刚开始以为贪心是没办法找全的, 后来看题解发现还是可以用贪心的, 每次最大匹配就完事了

------
