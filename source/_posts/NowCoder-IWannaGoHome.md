---
title: NowCoder-IWannaGoHome
tags:
  - NowCoder
  - Algorithm
  - Graph
  - Dijkstra
  - Hard
categories:
  - NowCoder
comments: true
mathjax: false
date: 2019-09-22 13:19:10
urlname: I-Wanna-Go-Home
---

<meta name="referrer" content="no-referrer" />

{% note info %}
## [Problem](https://www.nowcoder.com/practice/0160bab3ce5d4ae0bb99dc605601e971?tpId=40&tqId=21359&tPage=2&rp=1&ru=%2Fta%2Fkaoyan&qru=%2Fta%2Fkaoyan%2Fquestion-ranking)   
补之前的

The country is facing a terrible civil war----cities in the country are divided into two parts supporting different leaders. As a merchant, Mr. M does not pay attention to politics but he actually knows the severe situation, and your task is to help him reach home as soon as possible.     "For the sake of safety,", said Mr.M, "your route should contain at most 1 road which connects two cities of different camp."     Would you please tell Mr. M at least how long will it take to reach his sweet home?

The input contains multiple test cases.

The first line of each case is an integer N (2<=N<=600), representing the number of cities in the country.
The second line contains one integer M (0<=M<=10000), which is the number of roads.

The following M lines are the information of the roads. Each line contains three integers A, B and T, which means the road between city A and city B will cost time T. T is in the range of [1,500].

Next part contains N integers, which are either 1 or 2. The i-th integer shows the supporting leader of city i.

To simplify the problem, we assume that Mr. M starts from city 1 and his target is city 2. City 1 always supports leader 1 while city 2 is at the same side of leader 2.

Note that all roads are bidirectional and there is at most 1 road between two cities.

Input is ended with a case of N=0.

For each test case, output one integer representing the minimum time to reach home.

If it is impossible to reach home according to Mr. M's demands, output -1 instead.

### Examples:
**Input:**
> 2
> 1
> 1 2 100
> 1 2
> 3
> 3
> 1 2 100
> 1 3 40
> 2 3 50
> 1 2 1
> 5
> 5
> 3 1 200
> 5 3 150
> 2 5 160
> 4 3 170
> 4 2 170
> 1 2 2 2 1
> 0
 
**Output:**
> 100
> 90
> 540

{% endnote %}
<!--more-->

## Solutions
- 单源最短路径问题，计算两次最短路，分别是计算阵营1里各点最短路径，和阵营2里各点最短路径，然后再遍历边界线上联通两个阵营的边，看哪个最短 


## C++ Codes

```C++
#include <stdio.h>
#include <limits.h>
#define N 601
 
int G[N][N];    //城市间的距离，用时间表示
int camp[N];    //各个城市的阵营
int n[3];       //n[0]: 城市数量; n[1]、n[2]: 两个阵营各自的城市数量
 
int Add(int a, int b)
{   
    //两个数相加
    if(a==INT_MAX||b==INT_MAX) return INT_MAX;
    else return a+b;
}
 
int Add3(int a, int b, int c)
{   
    //三个数相加
    if(a==INT_MAX||b==INT_MAX||c==INT_MAX)  return INT_MAX;
    else return a+b+c;
}
 
int d1[N], d2[N];       //分别存储1、2到同阵营各点最小距离
bool mark[N];           //标记点是否在集合里面
void Dijkstra(int d[N], int from)
{   
    //dijkstra算法求某点到同阵营各点的最小距离
    for(int i=0; i<=n[0]; i++)
    {
        //不是同阵营的先塞进集合里面
        if(camp[i]==from) mark[i]=false;
        else mark[i]=true;
    }

    //把起点塞进集合
    mark[from]=true;
    //初始化d[N]
    for(int i=0; i<=n[0]; i++)
    {
        if(camp[i]==from) d[i]=G[from][i];
    }

    //把同阵营的点全都塞进集合
    for(int i=0; i<n[camp[from]]-1; i++)
    {
        bool isFirst=true;
        int near;

        //挑一个d[?]最小的，下标为near
        for(int i=1; i<=n[0]; i++)
        {
            if(mark[i]==false)
            {
                if(isFirst)
                {
                    near=i;
                    isFirst=false;
                }
                else if(d[i]<d[near]) near=i;
            }
        }

        mark[near]=true;        
        //更新同阵营的点的d[i]信息
        for(int i=1; i<=n[0]; i++)
        {
            if(camp[i]==from&&mark[i]==false)
            {
                int sum=Add(d[near], G[near][i]);
                if(sum<d[i]) d[i]=sum;
            }
        }
    }
}
 
int main()
{
    int m;
    while(scanf("%d", &n[0])!=EOF)
    {
        if(n[0]==0) break;
        scanf("%d", &m);

        //初始化邻接矩阵
        for(int i=0; i<=n[0]; i++)
        {
            for(int j=0; j<=n[0]; j++)
            {
                if(i==j) G[i][j]=0;
                else G[i][j]=INT_MAX;
            }
        }
        //读取边
        while(m--)
        {
            int a, b, t;
            scanf("%d%d%d", &a, &b, &t);
            if(t<G[a][b]) G[a][b]=G[b][a]=t;
        }

        n[1]=n[2]=0;                //两个阵营暂时都没城市
        for(int i=1; i<=n[0]; i++)  //读取阵营信息
        {
            scanf("%d", &camp[i]);
            n[camp[i]]++;
        }

        Dijkstra(d1, 1);            //求出1到同阵营各点的最小距离
        Dijkstra(d2, 2);            //求出2到同阵营各点的最小距离

        int ans=INT_MAX;            //先假设最小距离是无穷大
        //开始找穿越边境的路
        for(int i=1; i<=n[0]; i++)
        {
            for(int j=i+1; j<=n[0]; j++)
            {
                //i是1阵营j是2阵营的话
                if(camp[i]==1&&camp[j]==2)
                {
                    //看看从这条路过境是否更划算
                    int sum=Add3(d1[i], d2[j], G[i][j]);
                    if(sum<ans) ans=sum;
                }
                else if(camp[i]==2&&camp[j]==1)
                {
                    int sum=Add3(d2[i], d1[j], G[i][j]);
                    if(sum<ans) ans=sum;
                }
            }
        }

        //输出结果
        //最短总耗时是无穷，说明无路可走
        if(ans!=INT_MAX) printf("%d\n", ans);
        else printf("-1\n");
    }
    return 0;
}
```

## 总结
- 不是模板题，需要注意下两个阵营不能一起使用dijkstra，而是应该分阵营计算好，然后加上边界的值，得到最短路劲
- dijkstra主要是两个for循环，最短路径，每次找距离源点最近的点加进去，维护的dis数组也是距离源点的数组

------
