---
title: UCAS模式识别与机器学习
tags:
  - UCAS
  - PRML
  - Review
  - Notes
categories:
  - Notes
comments: true
mathjax: true
date: 2020-12-14 13:34:18
urlname: ucas-prml-review
---

<meta name="referrer" content="no-referrer" />
{% note info %}

国科大模式识别与机器学习笔记

{% endnote %}

<!--more-->

## 期末答疑

- 选择题 8 题
- 大题 8-9 道
- 选择题用贝叶斯或者全贝叶斯都行，只要答案正确就行
- 作业题选做不算分
- svm 要知道最基本的形式，优化，软间隔之类的分析，C 怎么变化
- 考试不会比给的计算题难
- 大题目过程怎么算都行只要答案对就可以
- 势函数不考了
- SMO 不考
- KKT 只要了解下

## 习题总结

1. 朴素贝叶斯，算后验概率比一下
2. Fisher 线性判别
   1. 准则函数: $J = \frac{w^T S_b w}{w^T S_w w}$
   1. 类间离散度矩阵：$S_b = (m_1 - m_2)(m_1 - m_2)^T$
   2. 类内离散度矩阵：$S_w = S_{w_1} + S_{w_2} = \sum_{i=1}^{N_1} (m_i - m_1)(m_i -m_1)^T + \sum_{i=1}^{N_2}(m_i - m_2)(m_i - m_2)^T$
   3. 最佳变换向量：$w^* = S_w^{-1}(m_1 - m_2)$
   4. 如果两类有概率权重，计算 $m_1,\ m_2,\ S_w$的时候要加上权重。
   5. 计算 $S_w^{-1}$的时候，是把 $[S_w I]$进行行变换，变成 $I S_w^{-1}$，其中 $I$是单位阵。
   6. 最后投影计算：$w^* \cdot X$
3. K-L 变换
   1. 判断所有样本均值是否为 0，如果有类别概率，就加上这个概率权重
   2. 计算协方差矩阵 $R=E[E_i(XX^T)]$
   3. 求矩阵 $R$的特征值和特征向量。使用 $|R-\lambda I| = 0$计算出特征值，然后用 $(R-\lambda E) \Phi = 0$得到特征向量。
   4. 最后进行降维：$X = \Phi \cdot X$
4. 感知机
   1. 记得先把所有的负样本乘以（-1），同时变成增广矩阵的形式，就是在$x=(x_1, x_2, .., x_n)$最后添加一个类别特征，1 或者-1
   2. 初始化权重 $w$最好是全 0。
   3. 迭代的终止条件是所有的样本都被分对。
   4. 迭代的过程，对每个样本进行计算：$w^T(t)x_i$，其中 $t$是步骤数，如果结果大于 0，就进行梯度下降，$w(t+1) = w(t)+ C\cdot x_i$，其中 C 是步长，就是学习率。如果不大于 0，就不改变权重。
5. 广义线性判别
   1. $g(x) = c_0+c_1 x+c_2 x^2+...+c_n x^n$。
   2. 定义 $x^*$和 $w$，使得，$g(x^*)=wx^*= \sum w_i \cdot f_i(x)$，$x^*$的各个分量为 $f_i(x) = x_{p_1}^{s_1}x_{p_2}^{s_2}···x_{p_n}^{s_n}$，且$w = (c_0, c_1, c_2, ...c_n)$。
6. 拉普拉斯平滑
   1. 分子+1，分母+类别数
   2. $P(y= 0)=\frac{3+1}{7+2}$，$P(y=1) = \frac{4+1}{7+2}$
   3. $P(x_1=0|y=0) = \frac{2+1}{3+2}$
7. 带高斯噪声的模型计算对数似然
   1. 高斯函数：$P = \frac{1}{\sqrt{2\pi}\sigma}\exp^{-\frac{(x-\mu)^2}{2\sigma^2}}$
   2. 令$e_i = y_i - f(x)$，然后带入到高斯函数里，就可以得到 $P(y_i|x_i, \theta_i)$
   3. 然后累乘可以得到似然函数，再 log 得到对数似然。
8. 正则化项
   1. 正则化项越大，模型越简单，越小，模型越复杂
   2. $\lambda$越大，训练误差增大，测试误差先降后升
9. SVM
   3. 决策边界的问题
       1. hard margin 最小化的只是 $\frac{1}{2}||w||^2$，会让 margin 最大化
       2. 如果是 sorft margin，松弛因子 C 越大，误差 $\epsilon$占的比例越大。越不会存在分错的点。
       3. C 趋向于 0 的时候，损失函数近似 hard margin，决策边界 margin 会很宽，而且不会在意部分样本错分类，因为就算错分了，C=0 也不会有惩罚。
       4. margin $r =\frac{1}{2}\frac{w^T(x_1 - x_2)}{||w||} = \frac{1}{||w||}$
       5. margin 越大，泛化能力越强。
       6. 当 C 很大的时候，噪声点可能会影响决策边界。因为支持向量就是决策边界上的点，如果新噪声点出现在决策边界里面，就会改变决策边界。
   4. 注意对偶问题的损失函数是：$L_d({\alpha_i}) = - \frac{1}{2}\sum_i \sum_j \alpha_i \alpha_j y_i y_j x_i^T x_i + \sum_i^N \alpha_i$。有个负号的。$s.t.\ \alpha_i \ge 0\ \forall i,\ \sum_i^N \alpha_i  y^{(i)} = 0$
   5. 计算题，把所有样本点要带入$L$，然后算出所有的$\alpha_i$，观察两类中最近的两个点，就是支持向量，然后计算 $w\* = \sum_{i\in SV}\alpha_i y_i x_i,\ w_0=\frac{1}{|SV|}\sum_{i\in SV}(y_i - w^Tx_i),\ g(x) = w\* ^Tx+w_0^*$。注意 $w_0$算的时候要除以支持向量个数。
   6. 注意几个常见的核函数：
       1. 核函数是有效的，和核函数矩阵是半正定的，是充要条件。
       2. $f(x) = w*\Phi(x)+b= \sum_{x_i\in SV} \alpha_i y_i x_i \Phi(x) + b = \sum_{x_i\in SV}\alpha_i y_i K(x_i, x)+b$
       3. RBF 核/高斯核：$K(x,x') = exp(-\frac{||x-x'||^2}{2s^2})$
       4. 多项式核函数：$K(x,x') = (x^Tx+1)^q$
       5. sigmoidal 核：$K(x,x') = tanh(2x^Tx'+1)$
10. k-means 慢慢算
11. k-means 是按照欧氏距离来的，所以两个类别之间能看到明显的分界线，在两个簇头连线的中垂线那里。GMM 会形成圆圆的簇，同时会有一定的去噪声的作用，对噪声不敏感。
12. 半监督学习的假设：
    1. 平滑假设：如果高密度空间中两个点相似，则对应的输出也相似
    2. 聚类假设：如果点在同一个簇，则很有可能是同一类
    3. 流形假设：高维数据大体满足一个低维的流形分布；邻近样本拥有相似输出；邻近的程度常用相似程度刻画。
13. 贝叶斯球
    1. 球能从 A 跑到 B，就表示 A 和 B 不条件独立
    2. ![kKeI09](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/kKeI09.png)
    3. ![bwWOlT](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/bwWOlT.png)
14. HMM的推断，使用前向算法和维特比解码，应该只会考推断
    1. 初始化概率 $\pi$
    2. 计算转移概率矩阵A，$a_{ij} = P(y_{t+1}^j | y_t^i = 1)$
    3. 计算发射概率矩阵B，$b_{ij} = P(X_t = j | y_t^i = 1)$
    4. 两种考法，一种用前向算法计算算未知X序列出现的概率，一种用维特比解码计算已知X最可能的状态序列。
    5. 计算未知X序列出现概率：
       1. $\alpha_t(y_t)$表示产生部分输出序列 $x_1, x_2,...,x_t$并结束于$y_t$的概率
       2. $\alpha_{t+1}(y_{t+1}) = \sum_{y_t} \alpha_t (y_t)\cdot a_{y_t, y_{t+1}} \cdot P(x_{t+1}|y_{t+1})$
    6. 计算已知序列X最有可能的状态序列：
       1. $V_t^k$表示结尾状态$y_t=k$时，最可能状态序列的概率
       2. $V_t^k = max(a_{i,k} \cdot V_{t-1}^i )\cdot P(X|y_t=k)$
       3. 初始化：$V_0^0 = 1$，其他都为0
       4. 计算每个时刻：$V_t^1, V_t^2···V_t^k$，k是所有的状态的数目，也就是说要计算这个时刻，每个状态以它结尾的概率。
       5. 每个步骤记录的最有可能的状态就是max得到前一个状态值。哪个 $\alpha_t(y_t)\cdot a_{y_t, y_{t+1}}$最大就是哪个
15. 梯度提升
    1. 用上次的函数计算残差，当前的基学习器就来拟合这个残差数据集：${(x_i, r_{im})}, i=1, 2,..,n$
    2. 当前基学习器的权重是使得整体模型误差最小的权重：$r_m = argmin_r \sum{i=1}^N L(y_i, f_{m-1}(x_i)+rh_m(x_i))$
    3. ![sxTxHy](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/sxTxHy.png)
16. Adaboost
    1. 其实是一个不断更新样本权重的过程，用新的样本权重计算新的基分类器的权重
    2. ![GJeHBY](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/GJeHBY.png)
17. 模式的概念：
    1. ⼴义地说，存在于时间和空间中可观察的物体，如果我们可以区别它们是否相同或是否相似，都可以称之为模式。
    2. 模式所指的不是事物本身，⽽是从事物获得的信息，因此，模式往往表现为具有时间和空间分布的信息。
    3. 模式的直观特性:
       - 可观察性
       - 可区分性
       - 相似性
18. 卷积：提取局部特征的手段，部分特征与filter滤波器相乘的操作。
19. 池化：将局部特征进行压缩，逐步扩大卷积范围的有效手段，从而在计算量不显著上升的情况下获得更加全局的特征。
20. CNN和传统特征提取方法的异同：
    1. 不同的特征之间权值共享
    2. CNN的权重是学到的，Gabor的权重是预设的

## 贝叶斯系列

### 贝叶斯法则

![ozGncN](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ozGncN.png)

![pTW4Po](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pTW4Po.png)

### 朴素贝叶斯

朴素贝叶斯属于是生成式模型，要计算联合概率分布，再得到后验概率。

#### 使用后验概率判别

![K0Seg1](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/K0Seg1.png)

#### 使用似然比判别

似然比：$l_{12} = \frac{P(x|\omega_q)}{p(x|\omega_2)}$
判别阈值：$\theta_{21} = \frac{P(\omega_2)}{\omega_1}$
若使用似然比来判断，则为：

![mZfXWW](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/mZfXWW.png)

其实把似然比的分式去掉，就是原始的贝叶斯判别

### 朴素贝叶斯和逻辑回归

![drEOm2](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/drEOm2.png)

![5OxFaq](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/5OxFaq.png)

### 贝叶斯最小风险判别

![38bLDp](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/38bLDp.png)

M 类的条件平均风险

- $r_j(x) = \sum_{i=1}^M L_{ij} P(\omega_i | x)$，
- 其中，$L_{ij}$是本来应该属于$\omega_i$类的模式被判别成$\omega_j$类的是非代价。
  - $i=j$，则 $L_{ij}<=0$，这时候表示不失分。
  - $i \neq j$，则 $L_{ij} >0$，分类错误，这时候 $L_{ij}$相当于是失分
- $r_j(x)$越小，越有可能是那个类别

同原始贝叶斯判别相同，也可以分为使用条件平均风险判别，和似然比判别。

#### 两类下条件平均风险判别

![LOaEiu](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/LOaEiu.png)

#### 两类下似然比判别

![qFBLiD](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/qFBLiD.png)

当分错的时候的 $L_{ij}=1$时，形式就和原始的贝叶斯判别一样了。

![tnwASP](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/tnwASP.png)

### 多类情况贝叶斯最小风险判别

这里还是用原始的计算条件平均风险$r_j$的方法计算，然后风险越小，就是那类。

特殊情况，当 $L_{ij}=1$时，条件平均风险 $r_j(x) = p(x) - p(x| \omega_j)p(\omega_j)$

![TW5KfY](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/TW5KfY.png)

![Vd8IZV](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Vd8IZV.png)

### 正态分布模式的贝叶斯分类

#### M 类的多变量正态类密度函数

- 此时判别函数是一个超二次曲面
- 两个模式类别之间用一个二次判别界面分开，就可以求得最优的分类效果
- 这里说的多变量正态类密度，其实就是每个类别下，有多个特征，然后服从正态分布

![LJnD97](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/LJnD97.png)

知道了似然$P(x|\omega_i)$之后，就可计算后验概率，然后得到判别函数。这里因为是有指数，所以取对数。

![gTyiCZ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/gTyiCZ.png)

![gvh3e7](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/gvh3e7.png)

判别函数记不住的话，可以记正态密度函数，然后推一下。

#### 两类的特殊情况

分两种情况

- $C_1 \neq C_2$
  - 判别界面 $d_1(x) - d_2(x)=0$是二次型方程，两类模式可以用二次判别界面分开
  - x 是二维的时候，判别界面是二次曲线
- $C_1 = C_2 = C$
  - 判别界面是 x 的线性函数，为一超平面
  - x 是二维的时候，判别界面是一条直线

只是多类情况的特殊情况，一样算，最后的判别函数用两个后验概率相减得到。

![wsFPTT](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wsFPTT.png)

![ys9btE](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ys9btE.png)

### 参数估计

贝叶斯分类从公式里就能看出，需要先验和似然才能算出后验。

如果知道分布，那就只要知道分布的参数，比如正态分布就只要知道均值和协方差矩阵。

两种方式：

1. 将参数当做非随机变量处理，比如矩估计
2. 将参数当成随机变量，比如贝叶斯参数估计

#### 看成非随机变量

![JtamHh](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/JtamHh.png)

迭代运算形式（这里考到了就炸了）
![7khXZU](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7khXZU.png)

#### 看成随机变量

![rd1Fdi](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/rd1Fdi.png)

感觉看的有点头大。。

![fTJQlf](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/fTJQlf.png)

![5ThIgf](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/5ThIgf.png)

![tOloLp](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/tOloLp.png)

### 贝叶斯网络

## 线性判别系列

这一类是判别式模型，通过直接建模后验概率，得到判别函数，而不去管每一类是什么情况，不考虑联合概率分布。

### 线性判别

简单的两类问题下的线性判别，就是一条直线。

![SMIBar](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/SMIBar.png)

权向量，增广模式向量，增广权向量
![K4GsRo](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/K4GsRo.png)

### 多类情况的线性判别

二类情况就直接一个$d(x)$就可以，多类情况可以有三种划分的方法。

1. 分为属于 $w_i$类和不属于 $w_i$类。M 类分成 M 个两类问题。因此有 M 个判别函数。有时候会遇到分不开的情况，比如其中一类被一个环包围了，这就需要用其他划分方法。
   - ![3JiZdH](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3JiZdH.png)
2. 每两个类就弄一个判别函数，只把这两个类分开。因此 M 类就需要$\frac{M*(M-1)}{2}$个判别函数。除了参数多的缺点，还存在不确定区域，即这个区域，对所有的$d_{ij}$都找不到一个$d_{ij}>0$的情况。
   - ![wgnrTz](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wgnrTz.png)
3. 是第二种方法的特例，第二种方法是直接写出了两两之间判别的判别函数，而这里是分成每个类有一个自己的判别函数，但这个又不是情况一那样直接按照大于0还是小于0分。这里是比较各自判别函数的大小。如果某个类的判别函数值比所有其他类都大，那就分到这个类。
   - ![2vs266](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/2vs266.png)
4. 划分方法 1 和方法 2 的比较
   - 对于 M 类模式的分类，多类情况 1 需要 M 个判别函数，而多类情况 2 需要 M\*(M-1)/2 个判别函数，当 M 较大时，后者需要更多的判别式（这是多类情况 2 的一个缺点）。
   - 采用多类情况 1 时，每一个判别函数都要把一种类别的模式与其余 M-1 种类别的模式分开，而不是将一种类别的模式仅与另一种类别的模式分开。有时候情况 1 分不开
   - 由于一种模式的分布要比 M-1 种模式的分布更为聚集，因此多类情况 2 对模式是线性可分的可能性比 多类情况 1 更大一些（这是多类情况 2 的一个优点）。

### 广义线性判别

在原本的模式空间中线性不可分，但是映射到高维空间后，变得线性可分。只要对原来的模式 x 做非线性变换就可以，之后再进行线性判别就可以。

![7WxIdB](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7WxIdB.png)

这里的 $f(x)$有多种情况，可以分为一次函数、二次多项式函数、r 次多项式函数

1. $f(x)$为一次函数
   1. ![3upSLl](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3upSLl.png)
2. $f(x)$为二次多项式函数
   1. ![ZfamVv](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ZfamVv.png)
3. $f(x)$为高次函数
   1. 判别函数各项的一般化表达：$x_1^{r_1}x_2^{r_2}···x_n^{r_n}1^{r_{n+1}}$，其中 $r_i \geq 0, integer$，且 $\sum_{i=1}^{n+1} r_i = r$
   2. ![PzADJv](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/PzADJv.png)
   3. ![ndLf68](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ndLf68.png)
4. 例题
   1. ![Rw0QB9](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Rw0QB9.png)

![7Y6Kh1](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7Y6Kh1.png)

![dCVm3c](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/dCVm3c.png)

### 分段线性判别

在线性不可分的情况下，如果使用广义线性判别，会增加维数。维数的大量增加会使得在低维空间里解析和计算熵行得通的方法在高位空间遇到困难，增加计算复杂性。所以有时候要用分段线性判别。

分段线性判别函数设计上，采用最小距离分类的方法。

![Y8P9iG](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Y8P9iG.png)

![2IG6U8](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/2IG6U8.png)

对于各类交错分布的情况，如果再用每类一个均值代表点来产生最小距离分类器，那么就会产生很明显的错误率。在这种情况下，可以先用聚类方法把一些类分解成若干个子类，再用最小距离分类。

### 模式空间和权空间

#### 模式空间

模式空间是在模式 X 的空间中。

- 对线性方程 $w_1 x_1 + w_2x_2+w_3x_3=0$，系数 $w = (w_1,w_2,w_3^T$，把 w 作为该方程在三维空间中平面的法线向量，则该平面过原点且与 w 垂直。
- ![c291UP](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/c291UP.png)
- ![KcMJaO](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/KcMJaO.png)
- ![vI4li2](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/vI4li2.png)

#### 权空间

权空间是在权重 W 的空间中。

- ![xxrszv](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xxrszv.png)
- ![YSFVNf](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/YSFVNf.png)

### Fisher 线性判别

考虑到高维空间比较复杂的问题，Fisher 线性判别对其进行了降维，把原本的 d 维空间压缩到一条直线上，就是投影在一条线上。 然而，即使样本在 d 维空间里形成若干紧凑的互相分得开的集群，当把它们投影到一条直线上时，也可能会是几类样本混在一起而变得无法识别。但是，在一般情况下，总可以找到某个方向，使在这个方向的直线上，样本的投影能分得开。

所以，关键就是怎么找到一条最好的，最能把样本分开的投影线。

#### 结论

- Fisher 准则函数为：$J_F(w) = \frac{w^TS_bw}{w^TS_ww}$
- 其中，类间离散度矩阵$S_b=(m_1-m_2)(m_1-m_2)^T$
- 总样本类内离散度矩阵$S_w = S_1+S_2 = \sum_{x \in T_1}(x-m_1)(x-m_1)^T + \sum_{x \in T_2}(x-m_2)(x-m_2)^T$
- 各类的样本均值向量$m_i = \frac{1}{N_i}\sum_{x \in T_i}x$
- 最佳变换向量$w^* = S_w^{-1}(m_1-m_2)$
- $w^*$是 Fisher 准则函数取得最大值时的解，就是 d 维 X 空间到 1 维 Y 空间的最佳投影方向。是一种降维的映射。
- 降维之和，只要确定一个阈值 T 就可以将投影点$y_n$和 T 比较，进行分类判别。

#### 推导过程

![ZPxJIB](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ZPxJIB.png)

![ZxyF9d](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ZxyF9d.png)

![jYw0f9](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/jYw0f9.png)

### 感知机模型

#### 两类

对负例样本的模式乘以(-1)，统一得到：

如果$w^T(k)x_k \leq 0$，则 $w(k+1) = w(k)+C \times {x_k}$。否则 $w(k+1)=w(k)$。C 是一个校正增量，和对应分错的样本$x_k$相乘。

![bpy5cx](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/bpy5cx.png)

![h4kP9f](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/h4kP9f.png)

#### 多类

![jZNutr](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/jZNutr.png)

![oTcBHm](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/oTcBHm.png)

![pRp6m9](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pRp6m9.png)

要训练一个判别性能好的线性分类器，训练样本需要多少？如果 k 是模式的维数，令$C=2(k+1)$则训练样本需要 C 的 10-20 倍。

## 距离和散布矩阵

![m9JqsU](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/m9JqsU.png)

![8ZZs2y](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/8ZZs2y.png)

![EicRz5](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/EicRz5.png)

![eDSnyv](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/eDSnyv.png)

![Qua4Cb](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Qua4Cb.png)

![CZM098](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/CZM098.png)

## K-L 变换

原理不讲了，证明也不记了

### PCA 总结

其实就是 PCA。

![WrhT75](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/WrhT75.png)

要使得 KL 变换后的误差最小，那么就需要将整体模式进行 K-L 变换之前，应先将其均值作为新坐标轴的原点， 采用协方差矩阵 C 或自相关矩阵 R 来计算特征值。

误差的均方误差 $\bar{\epsilon^2} = \sum_{j=m+1}^n \lambda_j$，其中，m 是降维时选择的最大特征值的数量，也就是说，均方误差等于没有选择的特征值的和。

协方差矩阵 $C = E[(X - \bar{X})(Y-\bar{Y})] = \frac{\sum_{i=1}^n(X_i - \bar{X})(Y_i - \bar{Y})}{n-1}$

自相关矩阵 $R = E(XX^T)$，当样本是 0 均值的时候，和协方差矩阵相同。

![rmyukc](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/rmyukc.png)

- K-L 变换是在均方误差最小的意义下获得数据压缩（降维）的最佳变换，且不受模 式分布的限制。对于一种类别的模式特征提取，它不存在特征分类问题，只是实现用低维的 m 个特征来表示原来高维的 n 个特征，使其误差最小，亦即使其整个模式分布结构尽可能保持不变
- 通过 K-L 变换能获得互不相关的新特征。若采用较大特征值对应的特征向量组成变换矩阵，则能对应地保留原模式中方差最大的特征成分，所以 K-L 变换起到了减小相关性、突出差异性的效果。在此情况下，K-L 变换也称为主成分变换（PCA 变换）。
- 需要指出的是，采用 K-L 变换作为模式分类的特征提取时，要特别注意保留不同类别的模式分类鉴别信息，仅单纯考虑尽可能代表原来模式的主成分，有时并不一定有利于分类的鉴别。

### 举个栗子

![ONneOI](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ONneOI.png)

## 偏差和方差

![Mrym46](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Mrym46.png)

![vV90RK](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/vV90RK.png)

![WLtZNm](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/WLtZNm.png)

![OXinsJ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/OXinsJ.png)

![PFaB4c](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/PFaB4c.png)

## 正则化

![rulOCX](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/rulOCX.png)

![lz5bXx](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/lz5bXx.png)

![7yXUdK](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7yXUdK.png)

![nh0TQw](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/nh0TQw.png)

## 训练误差和泛化误差

### ERM 经验风险最小化

![50rztg](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/50rztg.png)

![WBo9Qh](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/WBo9Qh.png)

### bound

![p38RDC](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/p38RDC.png)

对样本数量是有要求的

![ZKIZYv](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ZKIZYv.png)

![5vFTNn](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/5vFTNn.png)

### VC 维

![bJXZrO](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/bJXZrO.png)

## 交叉验证和特征选择

![dqIpd9](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/dqIpd9.png)

![fXcvt2](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/fXcvt2.png)

## 梯度下降

最小二乘法的梯度下降

![TofNEb](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/TofNEb.png)

![lMAceJ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/lMAceJ.png)

## 最小二乘和最大似然估计

![xhOo76](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xhOo76.png)

![zY8oOT](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/zY8oOT.png)

## 生成模型和判别模型

![qCwKLl](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/qCwKLl.png)

## 一些分布模型

![IqsVZ3](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/IqsVZ3.png)

![kP2bgO](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/kP2bgO.png)

![SQOGOo](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/SQOGOo.png)

## GDA 高斯判别分析

![h1vy83](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/h1vy83.png)

![bz1Gfl](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/bz1Gfl.png)

![zRpjU9](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/zRpjU9.png)

![l1cD98](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/l1cD98.png)

## 参数估计

![upbfmM](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/upbfmM.png)

### 贝叶斯参数估计

贝叶斯参数估计是假设参数是随机变量

![2RUyBV](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/2RUyBV.png)

![3vG7BT](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3vG7BT.png)

![8wxDis](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/8wxDis.png)

![VzyTfv](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/VzyTfv.png)

### 最大似然估计

MLE 是假设参数是确定的值

![wB9nDC](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wB9nDC.png)

#### 伯努利分布例子

![38XQeA](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/38XQeA.png)

#### 多项式分布例子

![SKUEIL](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/SKUEIL.png)

![M4F9Kp](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/M4F9Kp.png)

#### 正态分布例子

![TpwxLE](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/TpwxLE.png)

#### 多元正态分布例子

![3EaikU](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3EaikU.png)

### 拉普拉斯平滑

分子加一，分母加类别数

### KKT 条件

![kMKfsI](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/kMKfsI.png)

## 硬间隔 SVM

### 原始问题

![pLSl4B](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pLSl4B.png)

### 对偶问题

![wyEt4u](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wyEt4u.png)

![i2V3MF](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/i2V3MF.png)

![o7oZ36](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/o7oZ36.png)

### 支持向量

![irVCWP](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/irVCWP.png)

### 求解参数$w$和$w_0$

![oqDXp1](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/oqDXp1.png)

### 判别函数

![eCI2by](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/eCI2by.png)

## 软间隔 SVM

![72z5Od](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/72z5Od.png)

松弛变量和对应的惩罚，松弛变量越大，惩罚越大。

- ![1TtsO1](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1TtsO1.png)
- ![Mc76p5](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Mc76p5.png)

### 原始问题

![93hloI](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/93hloI.png)

### 对偶问题

![OQpl3w](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/OQpl3w.png)

![VvdEPA](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/VvdEPA.png)

### 求解和判别函数

![AaMCUl](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/AaMCUl.png)

## 核 SVM

就是将原始的输入空间映射到高维的特征空间。核函数简单理解是一种内积。

![uKsl2f](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/uKsl2f.png)

### 一些常见的核函数

![y42kjm](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/y42kjm.png)

![LfzThU](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/LfzThU.png)

## 损失函数

## 聚类

## 半监督

## 概率图

### HMM 隐马尔科夫

### 贝叶斯球

![AIphZb](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/AIphZb.png)

## Boost 和 Bagging

### AdaBoost

参考[Wikipedia](https://en.wikipedia.org/wiki/AdaBoost)

流程如下：
1. 数据集为:${x_i, y_i}_{i=1, 2,...m}, y_i \in {1, -1}$
2. 初始化样本权重：$D_i = \frac{1}{m} (\forall i)$
3. $for\ t=1···T:$
   - 训练一个弱学习器 $h_t:\ X->{1,\ -1}$
   - $error\ =\ \sum_{i=1}^m D_t(i) I[h_t(x_i) \ne y_i]$
   - $h_t$权重：$\alpha_t = \frac{1}{2} ln(\frac{1-\sigma_t}{\sigma_t}$
   - 对所有样本更新权重：$D_{t+1}(i) = \frac{D_t(i)}{Z_t} \exp^{-\alpha_t y_i h_t(x_i)}$，其中$Z_t$是正则化因子
4. $H_{final}(x)=sign(\sum_t^T \alpha_t h_t(x))$
