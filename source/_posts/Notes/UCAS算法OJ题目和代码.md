---
title: UCAS算法OJ题目和代码
comments: true
mathjax: true
date: 2020-12-22 11:42:24
tags:
  - UCAS
  - Algorithm
  - Notes
  - Math
categories: Notes
urlname: ucas-algorithm-oj-code
---

<meta name="referrer" content="no-referrer" />

{% note info %}

直接导出pdf太不方便看代码，直接放这里看了。

{% endnote %}
<!--more-->

## 测试

------

![lmYVa4](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/lmYVa4.png)

```c++
#include <stdio.h>
#include <iostream>
#include <algorithm>
using namespace std;

const int MAXN = 5000001;
int vec[MAXN];

int partition(int a[], int i, int j)
{
    int pivot = a[i];
    while (i < j)
    {
        while (i < j && a[j] <= pivot)
            j--;
        if (i < j)
            a[i++] = a[j];
        while (i < j && a[i] >= pivot)
            i++;
        if (i < j)
            a[j--] = a[i];
    }
    a[i] = pivot;
    return i;
}

void quick_sort(int a[], int l, int r)
{
    int pivot_pos;
    if (l < r)
    {
        pivot_pos = partition(a, l, r);
        quick_sort(a, l, pivot_pos - 1);
        quick_sort(a, pivot_pos + 1, r);
    }
}

int main(int argc, const char **argv)
{
    int n, k;
    scanf("%d%d", &n, &k);
    if (n == 0)
    {
        printf("%d", 0);
        return 0;
    }

    for (int i = 0; i < n; i++)
        scanf("%d", &vec[i]);

    // 方法1，快排，时间是1914ms，ac
    // quick_sort(vec, 0, n - 1);
    // printf("%d", vec[k - 1]);
    // for (int i = 0; i < n; i++)
    // {
    //     printf("%d ", vec[i]);
    // }

    // 方法2，快速选择，就是剪枝的快排，第k大在哪个区间就排哪个，另一个区间不排序，时间是1099ms，ac
    int l = 0, r = n - 1;
    int pivot_pos = 0;
    while (true)
    {
        pivot_pos = partition(vec, l, r);
        if (pivot_pos == k - 1)
            break;
        else if (pivot_pos > k - 1)
            r = pivot_pos - 1;
        else
            l = pivot_pos + 1;
    }
    printf("%d", vec[pivot_pos]);

    // 方法3，STL的sort，时间1801ms，ac
    // sort(vec, vec + n);
    // printf("%d", vec[n-k]);
    return 0;
}
```

## Divide and Conquer

------

![W4CxsA](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/W4CxsA.png)

```c++
#include <iostream>
#include <stdio.h>
#include <algorithm>

using namespace std;

// 普通公式法 TLE
long long normalPower(long long base, long long power)
{
    long long result = 1;
    base = abs(base) % 1000;
    for (int i = 1; i <= power; i++)
    {
        result = result * base;
        result = result % 1000;
    }
    return result % 1000;
}

// 快速幂 29ms
long long fastPower(long long base, long long power)
{
    long long result = 1;
    base = abs(base) % 1000;
    while (power > 0)
    {
        if (power & 1)
            //等价于if(power%2==1)
            result = result * base % 1000;

        power >>= 1; //等价于power=power/2
        base = (base * base) % 1000;
    }
    return result;
}

int main()
{
    long long base, power;
    scanf("%lld %lld", &base, &power);
    if (base == 0 && power == 0)
    {
        printf("%d", 0);
        return 0;
    }
    else if (power == 0)
    {
        printf("%d", 1);
        return 0;
    }

    printf("%lld", fastPower(base, power));
    return 0;
}
```

![PKEk0p](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/PKEk0p.png)

```c++
#include <iostream>
#include <stdio.h>
#include <algorithm>
#include <string.h>

using namespace std;

typedef unsigned long long ull;
typedef long long ll;
const int MAXN = 5000001;

struct p
{
    ll x;
    ll y;
    ull dis;
};

struct p points[MAXN];

int partition(p a[], int i, int j)
{
    p pivot = a[i];
    while (i < j)
    {
        while (i < j && a[j].dis >= pivot.dis)
            j--;
        if (i < j)
            a[i++] = a[j];
        while (i < j && a[i].dis <= pivot.dis)
            i++;
        if (i < j)
            a[j--] = a[i];
    }
    a[i] = pivot;
    return i;
}

bool cmp(p a, p b){
    return a.dis < b.dis;
}

int main()
{
    // 读数据
    int N, K;
    scanf("%d %d", &N, &K);
    memset(points, 0, sizeof(struct p) * MAXN);
    for (int i = 0; i < N; i++)
    {
        scanf("%lld %lld", &points[i].x, &points[i].y);
        points[i].dis = points[i].x * points[i].x + points[i].y * points[i].y;
    }

    // 快速选择，就是剪枝的快排，第k大在哪个区间就排哪个，另一个区间不排序
    // int l = 0, r = N - 1;
    // int pivot_pos = 0;
    // while (true)
    // {
    //     pivot_pos = partition(points, l, r);
    //     if (pivot_pos == K - 1)
    //         break;
    //     else if (pivot_pos > K - 1)
    //         r = pivot_pos - 1;
    //     else
    //         l = pivot_pos + 1;
    // }

    // printf("%lld %lld", points[pivot_pos].x, points[pivot_pos].y);
    sort(points, points + N, cmp);
    printf("%lld %lld", points[K - 1].x, points[K - 1].y);
    return 0;
}
```

LeetCode上有个一样的题，有三种解法，[题解链接](https://leetcode-cn.com/problems/k-closest-points-to-origin/solution/zui-jie-jin-yuan-dian-de-k-ge-dian-by-leetcode-sol/)
- 排序
- 优先队列
- 快速选择

```c++
//快速选择算法
class Solution {
    private:
        mt19937 gen{random_device{}()};

    public:
        int dis(vector<int>& point){
            return point[0] * point[0] + point[1] * point[1];
        }
        int partition(vector<vector<int>>& points, int left, int right) 
        {
            vector<int> pivot = points[left];
            int pivot_dis = dis(points[left]);
            while (left < right)
            {
                while (left < right && dis(points[right]) >= pivot_dis)
                    right--;
                if (left < right)
                    points[left++] = points[right];
                while (left < right && dis(points[left]) <= pivot_dis)
                    left++;
                if (left < right)
                    points[right--] = points[left];
            }
            points[left] = pivot;
            return left;
        }

        vector<vector<int>> kClosest(vector<vector<int>>& points, int K) {
            int n = points.size();
            int l = 0, r = n - 1;
            int pivot_pos;
            while (true)
            {
                pivot_pos = partition(points, l, r);
                if (pivot_pos == K - 1)
                    break;
                else if (pivot_pos > K - 1)
                    r = pivot_pos - 1;
                else
                    l = pivot_pos + 1;
            }
            return {points.begin(), points.begin() + K};
        }
};

```

## DP

------

![0g5KhH](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/0g5KhH.png)

```c++
#include <iostream>
#include <stdio.h>
#include <vector>
#include <string.h>
#include <set>
#include <cmath>
using namespace std;

const int N = 10001;

bool climb_stair(int *stones, int size)
{
    // 如果最后dp[n]非空就可以
    // 可以自己用opt表举例子
    int n = stones[size - 1];
    if ((1 + size) * size / 2 < n)
        return false;
    // dp vector保存每个石头的jump size，就是到这个位置的k值
    vector<set<int> > dp(n + 1);
    dp[1].insert(1);
    for (int i = 1; i < size; i++)
    {
        for (set<int>::iterator iter = dp[stones[i]].begin(); iter != dp[stones[i]].end(); ++iter)
        {
            int k = *iter;
            if (stones[i] + k - 1 <= n)
                dp[stones[i] + k - 1].insert(k - 1);
            if (stones[i] + k <= n)
                dp[stones[i] + k].insert(k);
            if (stones[i] + k + 1 <= n)
                dp[stones[i] + k + 1].insert(k + 1);
        }
    }
    return !dp[n].empty();
}
int main()
{
    int n;
    int w[N];
    memset(w, 0, sizeof(int) * N);
    if (scanf("%d", &n) == 1)
    {
        for (int i = 0; i < n; i++)
        {
            int a = scanf("%d", &w[i]);
        }
        bool res = climb_stair(w, n);
        if (res == true)
            printf("true");
        else
            printf("false");
    }
    return 0;
}
```

![C4z9Uw](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/C4z9Uw.png)

```c++
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <cmath>

using namespace std;

#define ll long long

int main()
{
    //这里会卡long long下界，设置成longlong最小值
    ll w=0, del_sum,no_del_sum;
    scanf("%lld", &w);
    ll max_sum = w, max_remove_one = w, remove_one_sum = w;
    ll max_data = -9223372036854775808;
    max_data = max(max_data, w);

    while (scanf("%lld", &w) != EOF)
    {
        // max_data保存数组最大值，如果数组全负，则返回最大值
        max_data = max(max_data, w);

        // 最多删除一个元素的dp
        no_del_sum = remove_one_sum + w;
        del_sum = max_sum;

        remove_one_sum = max(del_sum, no_del_sum);
        max_remove_one = max(remove_one_sum, max_remove_one);

        // 不删除元素的dp
        max_sum = max(max_sum + w, w);
    }
    if(max_data<0)
        printf("%lld", max_data);
    else
        printf("%lld", max(max_remove_one, max_sum));
    return 0;
}
```

## Greedy

------

![0BeIYo](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/0BeIYo.png)

```c++
#include <stdio.h>
#include <iostream>
#include <vector>
#include <string.h>
#include <cmath>

using namespace std;

const int N = 200001;
char s[N];

// ac
int main()
{
    scanf("%s", s);
    int cnta = 0, cntb = 0,i=0;

    while (s[i] != '\0')
    {
        if (s[i]=='A')
        {
            cnta++;
        }
        else
        {
            if(cnta>0)
                cnta--;
            else
                cntb++;
        }
        i++;
    }
    printf("%d", cnta+cntb%2);
    return 0;
}
```

![i6lDBv](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/i6lDBv.png)

```c++
#include <stdio.h>
#include <algorithm>

using namespace std;

const int N = 100000;
typedef long long ll;

ll s[N];

int main()
{
    int n;
    scanf("%d", &n);
    for (int i = 0; i < n; i++)
    {
        scanf("%lld", &s[i]);
    }
    sort(s, s + n);
    int flag = 0;
    for (int i = 2; i < n; i++)
    {
        if (s[i] < s[i - 1] + s[i - 2])
        {
            flag = 1;
            break;
        }
    }
    if (flag)
        printf("YES");
    else
        printf("NO");
    return 0;
}
```
