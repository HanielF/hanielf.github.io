---
title: NowCoder-谁是你的潜在朋友
tags:
  - NowCoder
  - Algorithm
  - Array
  - Easy
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-22 14:03:58
urlname: who-is-your-friend
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/0177394fb25b42b48657bc2b1c6f9fcc?tpId=40&tqId=21360&tPage=2&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
“臭味相投”——这是我们描述朋友时喜欢用的词汇。两个人是朋友通常意味着他们存在着许多共同的兴趣。然而作为一个宅男，你发现自己与他人相互了解的机会并不太多。幸运的是，你意外得到了一份北大图书馆的图书借阅记录，于是你挑灯熬夜地编程，想从中发现潜在的朋友。     

首先你对借阅记录进行了一番整理，把N个读者依次编号为1,2,…,N，把M本书依次编号为1,2,…,M。同时，按照“臭味相投”的原则，和你喜欢读同一本书的人，就是你的潜在朋友。你现在的任务是从这份借阅记录中计算出每个人有几个潜在朋友。

每个案例第一行两个整数N,M，2 <= N ，M<= 200。接下来有N行，第i(i = 1,2,…,N)行每一行有一个数，表示读者i-1最喜欢的图书的编号P(1<=P<=M)

每个案例包括N行，每行一个数，第i行的数表示读者i有几个潜在朋友。如果i和任何人都没有共同喜欢的书，则输出“BeiJu”（即悲剧，^ ^）
### Examples:
**Input:**
> 4  5
> 2
> 3
> 2
> 1 
 
**Output:**
> 1
> BeiJu
> 1
> BeiJu 

{% endnote %}
<!--more-->

## Solutions
- 用二维数组表示对应关系，遍历每本书的喜欢的人，就是潜在朋友 


## C++ Codes

```C++
#include<iostream>
#include<string>
#include<algorithm>
#include<vector>
using namespace std;

int main(){
  int N,M;
  while(cin>>N>>M){
    vector<vector<int> > mp(N, vector<int>(M+1,0));
    vector<int> friends(N,0);
    int p;
    for(int i=0;i<N;i++){
      cin>>p;
      mp[i][p]=1;
    }
    for(int i=1;i<=M;i++){
      vector<int> tmp;
      for(int j=0;j<N;j++)
        if(mp[j][i]==1) 
          tmp.push_back(j);
      for(int k=0;k<tmp.size();k++) friends[tmp[k]]=tmp.size()-1;
    }
    for(int i=0;i<N;i++){
      if(friends[i]==0) cout<<"BeiJu"<<endl;
      else cout<<friends[i]<<endl;
    }
  }
  return 0;
}
```

------
