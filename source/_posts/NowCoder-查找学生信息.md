---
title: NowCoder-查找学生信息
tags:
  - NowCoder
  - Algorithm
  - Search
  - Struct
  - Easy
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-22 13:16:09
urlname: find-information-of-stu
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/fe8bff0750c8448081759f3ee0d86bb4?tpId=40&tqId=21358&tPage=2&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
补之前的

输入N个学生的信息，然后进行查询。

输入描述:
输入的第一行为N，即学生的个数(N<=1000)
接下来的N行包括N个学生的信息，信息格式如下：
01 李江 男 21
02 刘唐 男 23
03 张军 男 19
04 王娜 女 19
然后输入一个M(M<=10000),接下来会有M行，代表M次查询，每行输入一个学号，格式如下：
02
03
01
04

输出M行，每行包括一个对应于查询的学生的信息。
如果没有对应的学生信息，则输出“No Answer!”

### Examples:
**Input:**
> 4
> 01 李江 男 21
> 02 刘唐 男 23
> 03 张军 男 19
> 04 王娜 女 19
> 5
> 02
> 03
> 01
> 04
> 03
 
**Output:**
> 02 刘唐 男 23
> 03 张军 男 19
> 01 李江 男 21
> 04 王娜 女 19
> 03 张军 男 19 

{% endnote %}
<!--more-->

## Solutions
- 使用结构体保存每个学生信息，然后找学号，找到就输出，找不到就No Answer 


## C++ Codes

```C++
#include<iostream>
#include<vector>
#include<string>
#include<algorithm>
#include<cstring>
using namespace std;

struct st{
    st(string sid="0", string sname="", string ssex="男", int sage=0):id(sid), name(sname),sex(ssex),age(sage) {};
    string id;
    string name;
    string sex;
    int age;
};

int main(){
    int N, M;
    while(cin>>N){
        vector<st> stu(N);
        for(int i=0;i<N;i++){
            cin>>stu[i].id>>stu[i].name>>stu[i].sex>>stu[i].age;
        }
        cin>>M;
        for(int i=0;i<M;i++){
            string target;
            cin>>target;
            int pos = -1;
            for(int j=0;j<N;j++){
                if(stu[j].id==target){
                    pos=j;
                    break;
                }
            }
            if(pos!=-1) cout<<stu[pos].id<<" "<<stu[pos].name<<" "<<stu[pos].sex<<" "<<stu[pos].age<<endl;
            else cout<<"No Answer!\n";
        }
    }
    return 0;
}
```

------
