---
title: 'ACM-2017WF-F-Posterize'
comments: true
mathjax: false
date: 2018-09-27 16:27:00
tags: [ACM-ICPC,Learning,DP,Notes]
categories: Notes
---

<meta name="referrer" content="no-referrer" />
  算法课居然直接安排了一个ACM-WF的题目...弄了半天弄懂了，记笔记记笔记!
<!--more-->
# ACM-ICPC World Finals 2017 F.Posterize DP

## 原题
### 题意
　　Pixels in a digital picture can be represented with three integers in the range 00 to 255255 that indicate the intensity of the red, green, and blue colors.

  To compress an image or to create an artistic effect, many photo-editing tools include a “posterize” operation which works as follows.
  
  Each color channel is examined separately; this problem focuses only on the red channel.
  
  Rather than allow all integers from 00 to 255255 for the red channel, a posterized image allows at most kk integers from this range.
  
  Each pixel’s original red intensity is replaced with the nearest of the allowed integers.
  
  The photo-editing tool selects a set of kk integers that minimizes the sum of the squared errors introduced across all pixels in the original image.
  
  If there are nn pixels that have original red values r1,…,rnr1,…,rn, and kk allowed integers v1,…,vkv1,…,vk, the sum of squared errors is defined as

　　∑i=1nmin1≤j≤k(ri−vj)2.∑i=1nmin1≤j≤k(ri−vj)2.

　　Your task is to compute the minimum achievable sum of squared errors, given parameter kk and a description of the red intensities of an image’s pixels.

{% note info %}
数字图像的像素可以用三个在0到255之间的整数表示,它们分别表示红色、绿色和蓝色的强度。

为了压缩图片或是为了生艺术效果,许多图像编辑工具收录了如下所述的”色调分离”操作。

每个颜色通道会分别考虑,本题只考虑红
色通道的情况。

不同于在红色通道使用0到255之间全部的整数,一张色调分离后的图片只会使用这些数字里至多 k
种整数。

每个像素原来的红色强度会被替换成最相近的可用强度。

图像编辑工具会选择k个整数来最小化替换过程
引起的平方误差之和。

假设原图有n个像素,它们的红色取值是r1,···,rn,而 k 种可用整数为v1,···,vk ,那么平方误差之和被定义为

　　∑i=1nmin1≤j≤k(ri−vj)2.∑i=1nmin1≤j≤k(ri−vj)2.

你的任务是计算可以实现的最小平方误差之和,参数k和图片的红色强度会给出。
{% endnote %}

### Input
The first line of the input contains two integers dd (1≤d≤2561≤d≤256), the number of distinct red values that occur in the original image, and kk (1≤k≤d1≤k≤d), the number of distinct red values allowed in the posterized image. The remaining dd lines indicate the number of pixels of the image having various red values. Each such line contains two integers rr (0≤r≤2550≤r≤255) and pp (1≤p≤2261≤p≤226), where rr is a red intensity value and pp is the number of pixels having red intensity rr. Those dd lines are given in increasing order of red value.

{% note info %}
第一行包含两个整数d(1≤d≤256)和k(1≤k≤d)  
分别表示原图中不同的红色强度有多少种,  
色调分离后可以使用的红色强度有多少种。  
接下来d行描述了每种红色强度在原图中占据的像素点数量。  
每行包含两个整数r(0≤r≤255)和p(1≤p≤226)  
这里r是一种红色强度的取值   
而p是这种取值对应的像素点数量。这d行信息按照红色强度取值升序给出。  
{% endnote %}

### Output 
Display the sum of the squared errors for an optimally chosen set of kk allowed integer values.

{% note info %}
输出最优的 k 种可选取值对应的平方误差之和。
{% endnote %}

 | Sample Input 1 | Sample Output 1 |
 | -----          |  -------        |
 | 2 1            |                 |  
 | 50 20000       |                 |      
 | 150 10000      |  66670000       | 

 | Sample Input 2 | Sample Output 2 |        
 | -----          |  -------        |
 | 2 2            |                 |  
 | 50 20000       |                 |    
 | 150 10000      | 0               |

 | Sample Input 3 | Sample Output 3 |        
 | -----          |  -------        |
 | 4 2            |                 |       
 | 0 30000        |                 |       
 | 25 30000       |                 |         
 | 50 30000       |                 |         
 | 255 30000      | 37500000        | 

### 思路
　　主要是dp,记忆化搜索＋剪枝,看后面的代码注释和题解

## 题目转化
### 题意
有256个位置，有d个位置上有人（每个位置上可能不止一个人），你可以在k个位置上插旗（每个位置上至多一面旗子），

每个人都会走到离自己最近的旗子，求所有人走的距离的平方和的最小值。

### 题解
要解决这个问题，我们可以建立一个二维数组*f[i][j]*，（j<=i），表示前i个位置，放了j个旗子，其中第i个位置一定放了旗子，前i个位置上的所有人走到旗子上的最小距离平方和。

如果我们假设第i个位置上的旗子是k个旗子中的最后一个（即第i个位置后面没有旗子），那么第i个位置后面的人只能全部走到第i个位置上，所以把_f[i][j]_加上i位置后面的人走到i位置的距离平方和就可以得到所有人走的距离平方和。

因此我们可以枚举最后一个旗子插的位置i，将_f[i][j_]加上i后面的人走到i位置的距离平方和，取最小值，即为答案。

*f[i][j]*数组的建立我们可以用下面的公式来计算：*f[i][j]=min{f[m][j-1]+w[m][i]},（j-1<=m< i)*，其中*w[m][i]*

表示m~i这一段只有m和i位置放了旗子，这一段上的人走的距离平方和。

由于第i个位置上固定了一面旗子，我们可以考虑另外j-1面旗子的位置。

若j-1面旗子在前m个位置（可以假设第m个位置上一定放旗子），那么*f[i][j]*可以分解成两段计算，

一段是1~m位置，放了j-1个旗子，其中第m个位置一定放了旗子，这就递归成了*f[m][j-1]*；

另一段是m+1~i位置，只有m和i位置放了旗子，这一段上的人只能走到m和i，即*w[m][i]*。

所以我们只要枚举m的值，取*f[m][j-1]+w[m][i]*的最小值，就可以得出*f[i][j]*的值。

下面给出一个f[i][j]的实例：考虑f[8][3]，前8个位置，放了3个旗子，其中第3个位置一定放了旗子。

剩余的2面旗子可能在前2/3/4/5/6/7个位置上，由此枚举出所有情况：

剩余的2面旗子在：
> 　　前2个位置：f[2][2]+w[2][8]
> 　　前3个位置：f[3][2]+w[3][8]
> 　　前4个位置：f[4][2]+w[4][8]
> 　　前5个位置：f[5][2]+w[5][8]
> 　　前6个位置：f[6][2]+w[6][8]
> 　　前7个位置：f[7][2]+w[7][8]

取出这些值中的最小值，即得到f[8][3]的值。

至于w[m][i]的建立较为简单，由于只有m和i位置放了旗子，m+1~i位置上的人只能就近走到m或i位置上，

我们可以求出m和i的中值，前一半的人走到m，后一半的人走到i，求出距离平方和即可。

### 代码参考
```C++
#include <cstdio>
#include <cstring>
#define N 260
#define ll long long

//读数据,返回一个整数
inline int read(){
    int x=0,f=1;
    char ch=getchar();

    //判断是否有符号
    while(ch<'0'||ch>'9'){if(ch=='-')f=-1;ch=getchar();}

    //按位读取组成整数
    while(ch>='0'&&ch<='9') x=x*10+ch-'0',ch=getchar();
    return x*f;
}

//一共n个位置，kk个旗子,a[N]表示每个位置的人数
int n,kk,a[N];

//f[i][j]表示前i个位置，放了j个旗子，其中第i个位置一定放了旗子的最小平方误差
ll w[N][N],f[N][N];

inline ll min(ll x,ll y){return x<y?x:y;}

int main(){
    //freopen("a.in","r",stdin);
    n=read();
    kk=read(); 

    //旗子数不可以比位置数多
    if(kk>=n){
      puts("0");
      return 0;
    }

    //读取每个位置的人数
    for(int i=1;i<=n;++i){
        int x=read();a[x+1]=read();
    }

    //建立w[i][j],i是第一个旗子位置,j是第二个旗子的位置
    for(int i=1;i<=256;++i)

        //i~j这一段只在i和j位置放旗子的平方误差
        for(int j=i+2;j<=256;++j){
          int mid=i+j>>1;

            //计算i到j之间所有的位置平方和
            for(int q=i+1;q<=j-1;++q){
                if(q>mid) 
                  w[i][j]+=(ll)(j-q)*(j-q)*a[q];
                else 
                  w[i][j]+=(ll)(q-i)*(q-i)*a[q];
            }
        }

    //只放一个旗子的情况,旗子在i处，遍历i之前所有的位置,累加
    for(int i=1;i<=256;++i)
        for(int j=1;j<i;++j)
            f[i][1]+=(ll)(i-j)*(i-j)*a[j];

    //建立f[i][j],j是旗子的数量，从２开始到kk
    for(int j=2;j<=kk;++j)
        //i是一共多少个位置，从j到256
        for(int i=j;i<=256;++i){
            f[i][j]=f[i-1][j-1];

            //前j-1个旗子放在[1,m]这里，并且m处有旗子,i处也有旗子,转化成求f[m][j-1]+w[m][i]的最小值
            for(int m=j-1;m<i-1;++m)
                f[i][j]=min(f[i][j],f[m][j-1]+w[m][i]);
        }

    ll ans=f[256][kk];
    //统计答案，如果最后一个旗子不在最后一个位置，就还需要加上最后一个旗子后面的位置的平方和，枚举最后一个旗子插的位置

    //i是最后一个旗子的位置,j是最后一个旗子到最后一个位置的每个位置,tmp是最后这部分的平方和
    for(int i=255;i>=kk;--i){
        ll tmp=0;
        for(int j=256;j>i;--j)
          tmp+=(ll)(j-i)*(j-i)*a[j];

        //比较不同位置的平方和，取最小值
        ans=min(ans,f[i][kk]+tmp);
    }

    printf("%lld\n",ans);
    return 0;
}
```

-------
