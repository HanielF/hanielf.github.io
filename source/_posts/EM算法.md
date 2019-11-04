---
title: EM算法
comments: true
mathjax: false
date: 2019-11-04 13:42:24
tags: 
  - EM
  - Notes
categories: Notes
urlname: EM-Notes
---

<meta name="referrer" content="no-referrer" />


## EM算法
下面主要介绍EM的整个推导过程。

<!--more-->

### 1. Jensen不等式

回顾优化理论中的一些概念。设f是定义域为实数的函数，如果对于所有的实数x，[![clip_image002](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161551524.png "clip_image002")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615511363.png)，那么f是凸函数。当x是向量时，如果其hessian矩阵H是半正定的（[![clip_image004](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615528987.png "clip_image004")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615523937.png)），那么f是凸函数。如果[![clip_image006](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615537068.png "clip_image006")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615525640.png)或者[![clip_image008](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161553755.png "clip_image008")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615534069.png)，那么称f是严格凸函数。

Jensen不等式表述如下：

如果f是凸函数，X是随机变量，那么[![clip_image010](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615543296.png "clip_image010")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615545772.png)

特别地，如果f是严格凸函数，那么[![clip_image012](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615541378.png "clip_image012")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615544725.png)当且仅当[![clip_image014](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615555871.png "clip_image014")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615549218.png)，也就是说X是常量。

这里我们将[![clip_image016](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615553952.png "clip_image016")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615557299.png)简写为[![clip_image018](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615554857.png "clip_image018")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615555380.png)。

如果用图表示会很清晰：

[![clip_image019](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615564400.png "clip_image019")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615564890.png)

图中，实线f是凸函数，X是随机变量，有0.5的概率是a，有0.5的概率是b。（就像掷硬币一样）。X的期望值就是a和b的中值了，图中可以看到[![clip_image010[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615576625.png "clip_image010[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615563561.png)成立。

当f是（严格）凹函数当且仅当-f是（严格）凸函数。

Jensen不等式应用于凹函数时，不等号方向反向，也就是[![clip_image021](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615577530.png "clip_image021")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615576102.png)。

### 2. EM算法

给定的训练样本是[![clip_image023](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615576168.png "clip_image023")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615579515.png)，样例间独立，我们想找到每个样例隐含的类别z，能使得p(x,z)最大。p(x,z)的最大似然估计如下：

[![clip_image024](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161558694.png "clip_image024")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615582613.png)

第一步是对极大似然取对数，第二步是对每个样例的每个可能类别z求联合分布概率和。但是直接求[![clip_image026](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615596333.png "clip_image026")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615589332.png)一般比较困难，因为有隐藏变量z存在，但是一般确定了z后，求解就容易了。

EM是一种解决存在隐含变量优化问题的有效方法。竟然不能直接最大化[![clip_image028](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616009812.png "clip_image028")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061615593891.png)，我们可以不断地建立[![clip_image030](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616017088.png "clip_image030")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616004862.png)的下界（E步），然后优化下界（M步）。这句话比较抽象，看下面的。

对于每一个样例i，让[![clip_image032](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616027502.png "clip_image032")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161601501.png)表示该样例隐含变量z的某种分布，[![clip_image032[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616033698.png "clip_image032[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616025060.png)满足的条件是[![clip_image034](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616031779.png "clip_image034")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616035126.png)。（如果z是连续性的，那么[![clip_image032[2]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616044419.png "clip_image032[2]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616041321.png)是概率密度函数，需要将求和符号换做积分符号）。比如要将班上学生聚类，假设隐藏变量z是身高，那么就是连续的高斯分布。如果按照隐藏变量是男女，那么就是伯努利分布了。

可以由前面阐述的内容得到下面的公式：

[![clip_image035](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616069601.png "clip_image035")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616052882.png)

（1）到（2）比较直接，就是分子分母同乘以一个相等的函数。

（2）到（3）利用了Jensen不等式，考虑到[![clip_image037](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616072491.png "clip_image037")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616061029.png)是凹函数（二阶导数小于0），而且

[![clip_image038](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616072557.png "clip_image038")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616072524.png)

就是[![clip_image039](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616085654.png "clip_image039")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616081718.png)的期望（回想期望公式中的Lazy Statistician规则）    


设Y是随机变量X的函数[![clip_image041](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616081784.png "clip_image041")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616082307.png)（g是连续函数），那么

（1） X是离散型随机变量，它的分布律为[![clip_image043](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616092373.png "clip_image043")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616083212.png)，k=1,2,&hellip;。若[![clip_image045](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616093835.png "clip_image045")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161609214.png)绝对收敛，则有

[![clip_image047](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616105296.png "clip_image047")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616091359.png)

（2） X是连续型随机变量，它的概率密度为[![clip_image049](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616115395.png "clip_image049")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161610346.png)，若[![clip_image051](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616117621.png "clip_image051")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616112920.png)绝对收敛，则有

[![clip_image053](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616123542.png "clip_image053")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616111557.png)


对应于上述问题，Y是[![clip_image039[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616133641.png "clip_image039[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616122703.png)，X是[![clip_image055](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616149837.png "clip_image055")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616138691.png)，[![clip_image057](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616144330.png "clip_image057")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616142901.png)是[![clip_image059](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616158017.png "clip_image059")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616142967.png)，g是[![clip_image055[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616167036.png "clip_image055[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616155575.png)到[![clip_image039[2]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616168498.png "clip_image039[2]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616164561.png)的映射。这样解释了式子（2）中的期望，再根据凹函数时的Jensen不等式：

[![clip_image060](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616179677.png "clip_image060")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616173547.png)

可以得到（3）。

这个过程可以看作是对[![clip_image028[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616189776.png "clip_image028[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616184726.png)求了下界。对于[![clip_image032[3]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616199875.png "clip_image032[3]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616196777.png)的选择，有多种可能，那种更好的？假设[![clip_image026[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616209974.png "clip_image026[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616198512.png)已经给定，那么[![clip_image028[2]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616212025.png "clip_image028[2]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616205023.png)的值就决定于[![clip_image057[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616216518.png "clip_image057[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616218677.png)和[![clip_image062](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616228187.png "clip_image062")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616213171.png)了。我们可以通过调整这两个概率使下界不断上升，以逼近[![clip_image028[3]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616239158.png "clip_image028[3]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616223237.png)的真实值，那么什么时候算是调整好了呢？当不等式变成等式时，说明我们调整后的概率能够等价于[![clip_image028[4]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616249257.png "clip_image028[4]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616235844.png)了。按照这个思路，我们要找到等式成立的条件。根据Jensen不等式，要想让等式成立，需要让随机变量变成常数值，这里得到：

[![clip_image063](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616242355.png "clip_image063")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616244830.png)

c为常数，不依赖于[![clip_image065](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616257994.png "clip_image065")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616249356.png)。对此式子做进一步推导，我们知道[![clip_image067](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616254123.png "clip_image067")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616259422.png)，那么也就有[![clip_image069](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161626568.png "clip_image069")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616259140.png)，（多个等式分子分母相加不变，这个认为每个样例的两个概率比值都是c），那么有下式：

[![clip_image070](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616261191.png "clip_image070")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616268649.png)

至此，我们推出了在固定其他参数[![clip_image026[2]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616275193.png "clip_image026[2]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616278192.png)后，[![clip_image072](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616283274.png "clip_image072")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616271846.png)的计算公式就是后验概率，解决了[![clip_image072[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616289404.png "clip_image072[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616282751.png)如何选择的问题。这一步就是E步，建立[![clip_image028[5]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161629375.png "clip_image028[5]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161628865.png)的下界。接下来的M步，就是在给定[![clip_image072[2]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616292044.png "clip_image072[2]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616295391.png)后，调整[![clip_image026[3]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616304095.png "clip_image026[3]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616305457.png)，去极大化[![clip_image028[6]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161631606.png "clip_image028[6]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616311096.png)的下界（在固定[![clip_image072[3]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616318687.png "clip_image072[3]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616317259.png)后，下界还可以调整的更大）。那么一般的EM算法的步骤如下：    

循环重复直到收敛 

（E步）对于每一个i，计算：

 &nbsp;&nbsp; [![clip_image074](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616324817.png "clip_image074")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616328164.png)

（M步）计算

&nbsp;&nbsp; [![clip_image075](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616321295.png "clip_image075")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616324850.png)


那么究竟怎么确保EM收敛？假定[![clip_image077](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616333345.png "clip_image077")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616338296.png)和[![clip_image079](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616347033.png "clip_image079")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616341983.png)是EM第t次和t+1次迭代后的结果。如果我们证明了[![clip_image081](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616359889.png "clip_image081")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616342049.png)，也就是说极大似然估计单调增加，那么最终我们会到达最大似然估计的最大值。下面来证明，选定[![clip_image077[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616367513.png "clip_image077[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616354100.png)后，我们得到E步

[![clip_image083](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616367231.png "clip_image083")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616364166.png)

这一步保证了在给定[![clip_image077[2]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161637710.png "clip_image077[2]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616375104.png)时，Jensen不等式中的等式成立，也就是

[![clip_image084](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616388824.png "clip_image084")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616382379.png)

然后进行M步，固定[![clip_image086](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616383317.png "clip_image086")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616381889.png)，并将[![clip_image088](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616397004.png "clip_image088")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616398367.png)视作变量，对上面的[![clip_image090](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616393449.png "clip_image090")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616392021.png)求导后，得到[![clip_image092](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616409645.png "clip_image092")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616401007.png)，这样经过一些推导会有以下式子成立：

[![clip_image093](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616434369.png "clip_image093")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616421729.png)

解释第（4）步，得到[![clip_image092[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616448928.png "clip_image092[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616447467.png)时，只是最大化[![clip_image090[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616447009.png "clip_image090[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616441993.png)，也就是[![clip_image095](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616451503.png "clip_image095")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616458438.png)的下界，而没有使等式成立，等式成立只有是在固定[![clip_image026[4]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616467141.png "clip_image026[4]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616458504.png)，并按E步得到[![clip_image097](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161647829.png "clip_image097")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616465779.png)时才能成立。

况且根据我们前面得到的下式，对于所有的[![clip_image097[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616471243.png "clip_image097[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616477830.png)和[![clip_image026[5]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616488518.png "clip_image026[5]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616486293.png)都成立

[![clip_image098](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616498585.png "clip_image098")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616498551.png)

第（5）步利用了M步的定义，M步就是将[![clip_image088[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616503144.png "clip_image088[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/2011040616164946.png)调整到[![clip_image100](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616519655.png "clip_image100")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616506241.png)，使得下界最大化。因此（5）成立，（6）是之前的等式结果。

这样就证明了[![clip_image102](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616522262.png "clip_image102")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161651801.png)会单调增加。一种收敛方法是[![clip_image102[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616535949.png "clip_image102[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161652900.png)不再变化，还有一种就是变化幅度很小。

再次解释一下（4）、（5）、（6）。首先（4）对所有的参数都满足，而其等式成立条件只是在固定[![clip_image026[6]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616544096.png "clip_image026[6]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616535459.png)，并调整好Q时成立，而第（4）步只是固定Q，调整[![clip_image026[7]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616552069.png "clip_image026[7]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616559179.png)，不能保证等式一定成立。（4）到（5）就是M步的定义，（5）到（6）是前面E步所保证等式成立条件。也就是说E步会将下界拉到与[![clip_image102[2]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616561852.png "clip_image102[2]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616566803.png)一个特定值（这里[![clip_image088[2]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161657315.png "clip_image088[2]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616575266.png)）一样的高度，而此时发现下界仍然可以上升，因此经过M步后，下界又被拉升，但达不到与[![clip_image102[3]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616582366.png "clip_image102[3]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616588953.png)另外一个特定值一样的高度，之后E步又将下界拉到与这个特定值一样的高度，重复下去，直到最大值。

如果我们定义

[![clip_image103](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616592432.png "clip_image103")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616582399.png)

从前面的推导中我们知道[![clip_image105](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616598594.png "clip_image105")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061616594658.png)，EM可以看作是J的坐标上升法，E步固定[![clip_image026[8]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617001758.png "clip_image026[8]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617009740.png)，优化[![clip_image107](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617017397.png "clip_image107")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617011268.png)，M步固定[![clip_image107[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617027812.png "clip_image107[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161702810.png)优化[![clip_image026[9]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617033451.png "clip_image026[9]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617031989.png)。

###  重新审视混合高斯模型

我们已经知道了EM的精髓和推导过程，再次审视一下混合高斯模型。之前提到的混合高斯模型的参数[![clip_image109](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617045502.png "clip_image109")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161704452.png)和[![clip_image111](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617055601.png "clip_image111")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617044139.png)计算公式都是根据很多假定得出的，有些没有说明来由。为了简单，这里在M步只给出[![clip_image113](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617069288.png "clip_image113")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617054238.png)和[![clip_image115](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617079387.png "clip_image115")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617064337.png)的推导方法。

E步很简单，按照一般EM公式得到：

[![clip_image116](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617077501.png "clip_image116")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617077468.png)

简单解释就是每个样例i的隐含类别[![clip_image055[2]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161708424.png "clip_image055[2]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617082551.png)为j的概率可以通过后验概率计算得到。

在M步中，我们需要在固定[![clip_image072[4]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161709141.png "clip_image072[4]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617097077.png)后最大化最大似然估计，也就是

 [![clip_image118](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617117831.jpg "clip_image118")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617102225.jpg)

这是将[![clip_image120](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617127930.png "clip_image120")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617111245.png)的k种情况展开后的样子，未知参数[![clip_image122](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617134998.png "clip_image122")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617128312.png)和[![clip_image124](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617145097.png "clip_image124")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617136459.png)。

固定[![clip_image126](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617143560.png "clip_image126")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161714146.png)和[![clip_image128](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617151707.png "clip_image128")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161715245.png)，对[![clip_image130](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617169821.png "clip_image130")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617168360.png)求导得

[![clip_image131](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617188591.png "clip_image131")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617174937.png)

等于0时，得到

[![clip_image132](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617186464.png "clip_image132")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617182527.png)

这就是我们之前模型中的[![clip_image115[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617199071.png "clip_image115[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617195658.png)的更新公式。

然后推导[![clip_image126[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617201363.png "clip_image126[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617202169.png)的更新公式。看之前得到的

[![clip_image133](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617212301.png "clip_image133")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617217808.png)

在[![clip_image113[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617222400.png "clip_image113[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617228987.png)和[![clip_image115[2]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617232499.png "clip_image115[2]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617227450.png)确定后，分子上面的一串都是常数了，实际上需要优化的公式是：

[![clip_image134](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617232009.png "clip_image134")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617233612.png)

需要知道的是，[![clip_image126[2]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617242108.png "clip_image126[2]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161724647.png)还需要满足一定的约束条件就是[![clip_image136](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617251825.png "clip_image136")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617255173.png)。

这个优化问题我们很熟悉了，直接构造拉格朗日乘子。

[![clip_image137](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617255272.png "clip_image137")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617258270.png)

还有一点就是[![clip_image139](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617265894.png "clip_image139")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617268336.png)，但这一点会在得到的公式里自动满足。

求导得，

[![clip_image141](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161727944.png "clip_image141")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617263419.png)

等于0，得到

[![clip_image142](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617279581.png "clip_image142")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617278468.png)

也就是说[![clip_image143](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/20110406161728486.png "clip_image143")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617277422.png)再次使用[![clip_image136[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617282471.png "clip_image136[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617289091.png)，得到

[![clip_image144](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617287521.png "clip_image144")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617287172.png)

这样就神奇地得到了[![clip_image146](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617297620.png "clip_image146")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617294206.png)。

那么就顺势得到M步中[![clip_image126[3]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617301307.png "clip_image126[3]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617306257.png)的更新公式：

[![clip_image147](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617312768.png "clip_image147")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617305244.png)

[![clip_image111[1]](https://images.cnblogs.com/cnblogs_com/jerrylead/201104/2011040616173244.png "clip_image111[1]")](http://images.cnblogs.com/cnblogs_com/jerrylead/201104/201104061617314994.png)的推导也类似，不过稍微复杂一些，毕竟是矩阵。结果在之前的混合高斯模型中已经给出。

### 总结

如果将样本看作观察值，潜在类别看作是隐藏变量，那么聚类问题也就是参数估计问题，只不过聚类问题中参数分为隐含类别变量和其他参数，这犹如在x-y坐标系中找一个曲线的极值，然而曲线函数不能直接求导，因此什么梯度下降方法就不适用了。但固定一个变量后，另外一个可以通过求导得到，因此可以使用坐标上升法，一次固定一个变量，对另外的求极值，最后逐步逼近极值。对应到EM上，E步估计隐含变量，M步估计其他参数，交替将极值推向最大。EM中还有&ldquo;硬&rdquo;指定和&ldquo;软&rdquo;指定的概念，&ldquo;软&rdquo;指定看似更为合理，但计算量要大，&ldquo;硬&rdquo;指定在某些场合如K-means中更为实用（要是保持一个样本点到其他所有中心的概率，就会很麻烦）。

另外，EM的收敛性证明方法确实很牛，能够利用log的凹函数性质，还能够想到利用创造下界，拉平函数下界，优化下界的方法来逐步逼近极大值。而且每一步迭代都能保证是单调的。最重要的是证明的数学公式非常精妙，硬是分子分母都乘以z的概率变成期望来套上Jensen不等式，前人都是怎么想到的。

在Mitchell的Machine Learning书中也举了一个EM应用的例子，明白地说就是将班上学生的身高都放在一起，要求聚成两个类。这些身高可以看作是男生身高的高斯分布和女生身高的高斯分布组成。因此变成了如何估计每个样例是男生还是女生，然后在确定男女生情况下，如何估计均值和方差，里面也给出了公式，有兴趣可以参考。
