---
title: UCAS数据挖掘
tags:
  - DataMining
  - Review
  - UCAS
  - Learning
categories:
  - Notes
comments: true
mathjax: true
date: 2020-11-19 15:04:21
urlname: ucas-datamining-review
---

<meta name="referrer" content="no-referrer" />

{% note info %}
UCAS 国科大数据挖掘（刘莹）期末复习笔记
{% endnote %}

<!--more-->

## Data WareHouse

-----

### 数据仓库

> A data warehouse is a subject-oriented, integrated, time-variant, and nonvolatile collection of data in support of management’s decision-making process.

#### 四个属性

1. 面向对象
   - 围绕主要对象组织数据，如消费者，产品；
   - 关注为决策者提供数据建模和分析；
   - 通过排除决策过程中的无用数据，提供关于特定主题问题的简单明了的视图
2. 集成的
3. 有时间维度
   - 时间维度上比操作型数据库长的多
   - 数据仓库中的key结构都会显式或者隐式地包含时间
4. 非易失性的
   - 数据被分开存储在物理设备上
   - 操作性的数据更新没有发生在数据仓库的环境中
   - 只需要两个数据访问操作：初始化加载数据，访问数据

#### OLTP对比OLAP

![OLTP and OLAP](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/2kuNPL.png)

![OLTP vs OLAP](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/w1MHjO.png)

#### Data WareHouse 对比 DBMS

![DataWareHouse vs DBMS](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/gbjHqh.png)

1. 两者都有很高的性能
2. DBMS是OLTP型，DataWareHouse是OLAP型
3. 方法和数据都不同
   - 决策需要历史数据，但是DBs没有
   - 决策需要整合异构数据
   - 决策需要高质量的数据

### Data Cube

数据仓库是以多维数据模型为基础的，就是数据立方体。

### 数据仓库模型

#### 三种模型

1. 星型模式
![star schema](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/0tBhOq.png)
![teBtyX](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/teBtyX.png)

2. 雪花模式
![7HCNr7](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7HCNr7.png)

3. 星系模式
![p5OQP0](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/p5OQP0.png)

#### 常见OLAP操作

1. Roll up(drill up): summarize data
2. Drill down(roll down): reverse of roll up
3. slice and dice: 投影和选择（就是切片）
4. pivot(rotate): 将多维数据集，可视化，3D重定向到一系列2D平面

**举个栗子**
![inIHsT](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/inIHsT.png)

**其他的一些操作**
![Sd6cgE](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Sd6cgE.png)

#### 三种数据仓库模型

1. Enterprise warehouse
   - collect all of the information about subjects spanning the entire organization
2. Data mart
   - a subset of corporate-wide data that is of value to a specific group of users.  Its scope is confined to specific, selected groups, such as marketing data mart
3. Virtual warehouse
   - A set of views over operational databases
   - Only some of the possible summary views may be materialized

### OLAP Server架构

1. Relational OLAP (ROLAP)
2. Multidimensional OLAP (MOLAP)
3. Hybrid OLAP (HOLAP)

### Data Cube的计算

Data Cube长这样
![xL7ylo](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xL7ylo.png)

使用bitmap来对OLAP数据进行index：

- 对某一列进行index
- 这一列的每个值都有一个单独的bit vector
- bit vector的长度 = base table中record个数
- bit vector对应的位置为1，表示原record是那个值
- 不适合基数比较高的域，因为每一个值都要开一个bit vector

![VZrbKK](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/VZrbKK.png)

对index还有`join`操作，即：$Join index: JI(R-id, S-id) where R(R-id) S(S-id,..)$

举个栗子：
![XdKACw](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/XdKACw.png)

#### Exercise

![tsYK26](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/tsYK26.png)
![CamOZb](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/CamOZb.png)

## Preprocessing

-----

### 主要任务

1. data cleaning 缺失值、噪声、异常值、不一致
2. data integration 集成多个数据库或文件
3. data transformation 标准化和聚合
4. data reduction 降维
5. data discretization 离散化

### 数据特征

#### 描述数据集中

1. mean:
   1. mean，普通均值
   2. weighted arithmetic mean，$\hat{x} = \frac{\sum_{i=1}^nw_ix_i}{\sum_{i=1}^nw_ix_i}$
   3. trimmed mean(去除了极值)
1. median:
   1. 中位数，奇数个就是中间的，偶数个就是中间两个平均
1. mode:
   1. 众数
   2. 单峰，双峰，多峰
   3. 经验公式：$mean-mode = 3 * (mean-median)$

#### 描述数据分散

1. 四分位数，0.25和0.75，计算过程参考[如何计算四分位数值&应用](https://zhuanlan.zhihu.com/p/235345817)
2. 四分位间距：$IQR = Q_3 - Q_1$
3. 五个值，$min, Q_1, Median, Q_3, max$
4. 按照上面五个数画boxplot图
5. 异常值通常为和$Q_1, Q_3$距离$1.5*IQR$之外的数
6. 方差分母是N或者N-1
7. 标准差

Exercise:
![Gs27oh](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Gs27oh.png)

- boxplot 举个栗子
![1MiL6u](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1MiL6u.png)
- 正态分布有$3\mu$法则
![8DBj44](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/8DBj44.png)
- 数据直方图
- 数据分位数图，横轴时0-1的分位数值f，$f=\frac{i-0.5}{n}$
![RpeUoi](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/RpeUoi.png)
- 分位数-分位数（Q-Q）图
![JIldvu](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/JIldvu.png)
> 分位数—分位数图与分位数图的不同之处是什么? 
> - 分位数图是一种用来展示数据值低于或等于在一个单变量分布中独立的变量的粗略百分比。这样,他可以展示所有数的分位数信息,而为独立变量测得的值(纵轴)相对于它们的分位数(横轴) 被描绘出来。
> - 但分位数—分位数图用纵轴表示一种单变量分布的分位数,用横轴表示另一单变量分 布的分位数。两个坐标轴显示它们的测量值相应分布的值域,且点按照两种分布分位数值展示。一 条线(y=x)可画到图中+以增加图像的信息。落在该线以上的点表示在y 轴上显示的值的分布比x 轴的相应的等同分位数对应的值的分布高。反之,对落在该线以下的点则低。
- scatter plot
- Loess (local regression) curve: add a smooth curve
to a scatter plot to provide better perception of the
pattern of dependence

### data cleaning

1. 缺失值
   1. 忽略那条数据
   2. 手动填充
   3. 自动填充
2. 噪声数据
   1. binning，首先排序数据
      1. equal-width：按照最大最小值的间距均分成N份
      2. equal-depth：按照个数均分成N份
   2. regression
   3. clustering
   4. 结合计算机和人工
![Py3S6t](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Py3S6t.png)
![TjZ6e9](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/TjZ6e9.png)

### data transfromation

1. 标准化：
   1. min-max标准化
   2. z-score标准化
   3. decimal scaling，让所有的数绝对值都小于1
![2Vch3x](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/2Vch3x.png)

### 数据集成

1. 实体命名问题
2. 检测数据冲突
3. 检测数据冗余
   - 可以通过相关性分析检测
4. 相关性分析
   1. 数值数据：pearson correlation coefficient 皮尔逊相关系数
   ![UaUZme](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/UaUZme.png)
   2. 分类数据：$x^2$测试
   ![ZNWEeZ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ZNWEeZ.png)
   ![ySlvfo](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ySlvfo.png)

### 数据降维

减少数据集的表示量，该表示量要小得多，但会产生相同（或几乎相同）的分析结果

1. 数据立方体聚合，从base cuboid向上聚合
1. 降维-例如，删除不重要的属性
   1. 特征选择，选择最少的特征集，以使概率给定这些要素的值，不同类的分布应尽可能接近原始分布
   2. 启发式方法，从前向后选择或者从后向前删除，或结合二者，或决策树
   ![lugMFF](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/lugMFF.png)
2. 数据压缩
   1. 小波变换
   2. PCA主成分分析
   3. Numerosity Reduction
      1. 使用另一种形式表示数据
      2. 参数化方法：拟合模型，只保存模型参数，如回归模型
      3. 非参数化方法：不假设模型，主要用直方图，聚类或采样
3. 降低浊度-例如，将数据拟合到模型中
4. 离散化和概念层次生成

### 离散化和概念层次

- 离散化：将连续数据按照间隔分成区间数据表示
- 概念层次表示：使用高层次的概念表示低层次的，如使用young, middle-aged,senior表示年龄

对于数值数据：

1. binning top-down
2. histogram top-down
3. clustering top-down
4. entropy-based
   ![imt3kO](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/imt3kO.png)
   ![UDqDSv](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/UDqDSv.png)
5. natural partitioning，使用`3-4-5`方法
   ![EsM474](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/EsM474.png)
   ![gdVCBa](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/gdVCBa.png)

对于分类数据：

- 指定不同属性的顺序，按照level排，可以根据人的先验知识，也可以自动生成，通过分析不同数值的数量，比如street有674339个不同值，而country只有15个不同值

## Classification

-----

分类用于预测分类数据标签，基于训练数据集和分类标签，用于预测新的reocrd，而预测是用于对连续数值建模，用于预测未知或者缺失值

分类常用于信贷审批,目标营销,医疗诊断,欺诈检测,入侵检测

监督学习：分类
非监督学习：聚类

### 评估模型方法

- 准确度
- 速度
- 鲁棒性
- 可扩展性
- 可解释性

### 决策树

有两个阶段：

- 建模：描述一个确定了分类的集合，关键词：类别标签，训练集，分类规则/决策树/数学公式
  - 分两个阶段
  - 开始是所有的训练样本在root，基于选择的属性，递归地划分样本
  - 然后进行树的剪枝，识别并移除那些反应噪声和异常值的枝干
- 模型使用：对未来数据或位置数据分类，关键词：估计准确率，

```C++
Generate_decision_tree (D, attribute_list)
(1) create a node N;

(2) if tuples in D are all of the same class, C then
(3)      return N as a leaf node labeled with the class C;

(4) if attribute_list is empty then
(5)      return N as a leaf node labeled with the majority class in D; // majority voting

// 一般基于启发式或者统计量方法选属性，如信息熵和基尼指数
(6) apply Attribute_selection_method(D, attribute_list) to find the highest information gain;
(7) label node N with test-attribute;

(8) for each value a_i of test-attribute // partition the tuples and grow subtrees for each
partition
(9)    Grow a branch from node N for test-attribute = a_i; // a partition
(10)   Let s_i be the set of samples in D for which test-attribute = a_i;
(11)   if s_i is empty then
(12)      attach a leaf labeled with the majority class in D to node N;
(13)   else attach the node returned by Generate_decision_tree(s_i , attribute_list) to node N;
(14) end for
```

#### 信息熵和信息增益

1. 信息增益Information Gain（用于ID3）
   - 全集S中有p个P种类和n个N种类，原本就按照**属性F1**这个分两类
   - 则经验熵：
   - $ I(p,n) = -\frac{p}{p+n}log_2 \frac{p}{p+n}-\frac{n}{p+n}log_2\frac{n}{p+n} $
   - 若这时训练集按照**属性F2**被分为${S_1, S_2, ..., S_v}$，每个$S_i$包含$p_i+n_i$个样本，则条件熵为：
   - $E(F2) = \sum_{i=1}^v \frac{p_i+n_i}{p+n}I(p_i,n_i)$
   - 信息增益为：
   - $Gain(F2) = I(p,n)-E(F2) = 经验熵-条件熵$
   - ![ifVlWa](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ifVlWa.png)
   - **选信息增益最大的**
2. 增益比（用于C4.5)
   - 将信息增益，除以，分裂的属性的条件熵
   - $GainRatio(F2) = Gain(F2)/I(F2) = \frac{I(F1)-E(F2)}{I(F2)}$
   - **选增益比最大的**
3. 基尼指数（用于CART，IBM Intelligent Miner）
   - 数据集T中有n个类，$p_j$是T中类别j的频率
   - 则先验基尼指数为：
   - $gini(T) = 1-\sum_{j=1}^n p_j^2$
   - 若将T分为两个子集$T_1$和$T_2$，
   - 则条件基尼指数为：
   - $gini_{split}(T) = \frac{N_1}{N}gini(T)+\frac{N_2}{N}gini(T_2)$
   - **遍历所有可能点，选基尼增益最大的，即选择最小的基尼指数来划分**

#### 从决策树中提取分类规则

- 用`IF-THEN`规则表示
- 从root到leaf是一条规则
- 叶子节点维护类别
- 规则容易理解
- ![j1vQUi](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/j1vQUi.png)

#### 过拟合和剪枝

防止过拟合：

1. 提前剪枝：尽早停止构建树-如果会导致优度度量值低于阈值，就不要拆分节点；但是很难确定阈值
2. 后剪枝：从“完全生长”的树上移除树枝-得到一系列逐渐修剪的树，就是把节点一个个去掉看评估结果；使用一个和训练集不同的集合来决定哪个是最好的剪枝

#### 验证准确度

1. Holdout: train和test分别为$\frac{2}{3}$和$\frac{1}{3}$
2. 交叉验证：k-fold，训练集k-1份，验证集1份，重复k次，取平均准确度

#### 决策树评价

1. 相对快
2. 可以转换为简单易懂的规则
3. 准确度相对较好
4. 可扩展性较好
   - 可扩展性：不同的分⽀计算互相独⽴互不影响，所以可以把任务分配给不同的机器上，保持时间的稳定

#### 决策树的优化

1. 允许连续值属性：动态定义新的离散值属性，该属性将连续属性值划分为离散的间隔集
2. 处理缺失值
3. 基于已有的属性创建新属性

### 贝叶斯分类

#### 基础

提前假设了每个属性是相互独立的

公式：

$$P(C_i|X)=\frac{P(X|C_i)P(H)}{P(C_i)} = \frac{似然*先验}{evidence}$$

在朴素贝叶斯中，由于$P(x)$是固定的，因此只要最大化$P(X|C_i)P(H)}{P(C_i)$

#### 推广

1. 属性之间条件独立
   - $$P(X|C_i) = \prod_{k=1}^n P(x_k|C_i) = P(x_1|C_i)\times P(x_2|C_i)\times ... \times P(x_n|C_i)$$
2. 如果$A_k$是类别数据，则$P(x_k|C_i)=\frac{s_{ik}}{s_i}$
3. 如果是连续型，则把$C_i$里面的数据看成是高斯分布：
   - $P(X_k|C_i) = g(x,\mu_{c_i},\sigma_{c_i})=\frac{1}{\sqrt(2\pi)\sigma }\exp^{-\frac{(x-\mu)^2}{2\sigma^2}}$
4. 栗子：
   ![OKkZ8N](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/OKkZ8N.png)

#### 评价

1. 优点
   1. 容易实现
   2. 结果一般较好
2. 缺点：
   1. 假设属性独立，因此损失精度
   2. 实际上，变量之间确实存在依赖关系，但是朴素贝叶斯无法对其建模
3. 解决这些依赖：
   - 贝叶斯信念网络

### 神经网络和反向传播

#### 前向传播

1. 对于输出层节点：
   - $$ O_j = I_j$$
2. 对于隐藏层节点：
   - 前一层所有节点权重和，加上当前节点的偏差
   - $$ I_j = \sum_i w_{ij}O_i+\theta_j $$

   - $$ O_j = g(I_j) = \frac{1}{1+\exp^{-I_j}}$$

#### 反向传播误差

1. 对于输出层节点
   - $$Err_j = O_j(1-O_j)(T_j-O_j)$$
2. 对于隐藏层节点，当前j个节点，后一层为k个节点
   - $$Err_j = O_j(1-O_j)\sum_k Err_k w_{jk}$$ // 输出，乘以（1-输出），乘以，后一层节点的误差乘以权重，的和
3. 对于网络中的每个权重$w_{ij}$
   - $$\delta w_{ij} = (lr) Err_j O_i$$ // 后一层的节点误差乘以当前层节点的输出
   - $$ w_{ij} = w_{ij}+\delta w_{ij}$$
4. 对于网络中的偏置$\theta_j$
   - 前一层所有节点权重和，加上当前节点的偏差
   - $$\delta \theta_j = (lr) Err_j$$

   - $$\theta_j = \theta_j + \delta \theta_j$$

#### 栗子

![IDTOqC](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/IDTOqC.png)
![YNnM97](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/YNnM97.png)

#### 评价

1. 优点
   - 对噪声数据的高容忍度
   - 非常适合于连续值输入和输出
   - 在大量实际数据中获得成功
   - 最近针对该技术开发了一些技术从训练有素的神经网络中提取规则
2. 缺点
   - 较长的训练时间
   - 要求通常根据经验确定最佳的许多参数，例如，网络拓扑或“结构”
   - 可解释性差：难以解释所学权重和网络中“隐藏单位”背后的符号含义

### KNN-K近邻

#### Basic

1. 所有样本对应于D维空间的点
2. 使用欧氏距离计算$dist(X_1, X_2)$
3. 目标函数可以是离散值或者实数值
4. 对每个离散值，KNN返回在k个样本中值最多的值（投票）

#### Exercise

![aY5Evl](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/aY5Evl.png)

#### 评价

1. k-NN用于给定未知元组的实值预测
   - 返回最近的k个邻居的平均值
1. 通过对k个最近的邻居进行平均来对噪声数据进行鲁棒处理
1. 邻居之间的距离可以由不相关的属性决定
   - 要克服它，请消除不相关的属性
1. 惰性学习器
   - 不构建分类器
   - 存储所有训练样本
   - 每个新元组的计算成本很高

### 集成学习

分为bagging和boosting

#### bagging

- 基于多个基学习器的投票
- 训练时，每次采样一部分样本训练，迭代N次，每次产生一个分类器
- 预测时使用这些分类器投票决定

![TVvg5Q](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/TVvg5Q.png)
![porewY](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/porewY.png)

#### boosting

- 基于前一次的预测结果分配权重
- 后一次的分类器会更关注前一次分类错误的样本
- 迭代产生K个分类器
- 最后根据加权组合得到预测结果，权重是准确度

![cjivYO](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/cjivYO.png)
![Lu1kaq](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Lu1kaq.png)

## Regression

-----

DM好像不是很关注回归，具体的看PRML的笔记吧

### 线性回归

![ymHngI](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ymHngI.png)

### 非线性回归

![L79wwo](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/L79wwo.png)

## 评价指标

-----

具体的可以看[精确率、召回率、F1 值、ROC、AUC 各自的优缺点是什么？ - 东哥起飞的回答 - 知乎](https://www.zhihu.com/question/30643044/answer/510317055)

![oGcKun](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/oGcKun.png)

TP和FP之类的是相对真实样本说的：TP就是真的正样本;FP就是假的正样本，其实是负样本；TN是真的正样本；FN是假的负样本，其实是正样本

1. $准确率 Acc = \frac{TP+TN}{TP+FN+FP+TN}$
2. $误差率 Error rate = 1- Acc$
3. $召回率/查准率/灵敏度 Recall/Sensitivity = \frac{TP}{TP+FN}$
4. $精确率/查准率 Precision = \frac{TP}{TP+FP}$
5. $特异度 Specificity = \frac{TN}{FP+TN}$
6. $真阳率 TPR = 灵敏度 = \frac{TP}{TP+FN}$
7. $假阳率 FPR = 1- 特异度 = \frac{FP}{FP+TN}$
8. $ROC和AUC$
9. $F1=\frac{2\times 查准率 \times 查全率}{查准率 + 查全率}$
10. $PR曲线，横坐标是查全率，纵坐标查准率$

## Clustering

-----

### 评价聚类

- high intra-class similarity
- low inter-class similarity
- Dissimilarity/Similarity metric: Similarity is expressed in terms of a distance function, typically metric: d(i, j)
   - ![hTQfqz](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hTQfqz.png)

### 数据类型

1. 区间标度类型 Interval-scaled（数值类型）
   1. 标准化：z-score $z_{if} = \frac{x_{if}-m_f}{s_f}$
      - ![N6EuS5](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/N6EuS5.png)
   3. 距离计算：
      1. ![wBFBLW](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wBFBLW.png)
      2. $q=2$时为欧氏距离
      3. ![M8VhhY](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/M8VhhY.png)
2. 二值数据类型Binary
   1. 对于对称二值变量和非对称的，计算距离时分母不同
   2. ![1lFOfs](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1lFOfs.png)
   3. ![wkVgPE](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wkVgPE.png)
3. 标称数据类型Nominal
   1. 标称属性的值是⼀些符号或实物的名称，每个值代表某种类别、 编码或状态，所以标称属性⼜被看做是分类型的属性(categorical）。这些值不必具有有意义的序，并且不是定量的。
   2. 计算距离
      - ![Z9UOIx](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Z9UOIx.png)
4. 有序变量Ordinal
   1. 相比标称数据类型，它的顺序是比较重要的，按照顺序排序后，再用rank表示值，并归一到0和1内。$z_{if} = \frac{r_{if}-1}{M_f-1}$
   2. 计算距离：使用区间标度类型的方式计算
5. 比率标度属性 Ratio-scaled
   1. ⽐率标度属性的度量是⽐率的，可以⽤⽐率来描述两个值，即⼀个值是另⼀个值的倍数，也可以计算值之间的差
   2. 一般不直接把他们当成区间标度数据，使用log处理后，
   3. 使用归一化处理
   4. 当成区间标度数据处理-
6. 混合数据类型
   1. 可能存在多种数据类型
   2. 使用加权距离表示：$d(i,j) = \frac{\sum_{f=1}^p \phi_{ij}^{(f)}d_{ij}^{(f)}}{\sum_{f=1}^p \phi_{ij}^{(f)}}$
   3. 如果f是非对称属性，且有缺失值或者$x_{if}=x_{jf}=0$， 则$\phi_{ij}$为0，否则权重为1
   4. 如果f是二值属性或标称类型，则：如果$x_{if}=x_{jf}$则$d_{ij}^{(f)}=0$，否则权重为1
   5. 如果是categorical的，看下面的例子，不相等就是1
   6. f是区间标度的，直接使用标准化距离
   7. f是有序的，计算rank代替值，并使用$z_{if} = \frac{r_{if}-1}{M_f-1}$处理
   8. f是比率类型的，使用log处理，然后min-max归一化，然后当成区间标度计算距离
   9. 注意最后要进行累加，然后min-max归一化
7. 其实最后，除了二值类型，和标称数据类型Nominal，其他都是转化后当成区间标度类型Interval-scaled处理。
8. Exercise
   1. ![hiZh08](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hiZh08.png)
   2. ![bwSzEr](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/bwSzEr.png)
   3. ![9giJhV](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/9giJhV.png)

### 类别属性

质心，半径和直径

![cV2x6l](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/cV2x6l.png)

### Partitioning Methods划分方法

主要有k-means, k-medoids, k-modes, k-prototype等等。

#### k-means

##### 基础

- 目标是最小化每个类内距离之和$\sum_{m=1}^k\sum_{t_{mi}}\in Km (C_m -t_{mi})^2$
- 常终止在局部最优，要用确定性退火和遗传算法来找全局最优
- 缺点
  - 要定义k值
  - 不好处理类别数据
  - 无法处理异常值和噪声
  - 不适合用于非凸的形状

##### 算法

1. k个随机初始化的质心
2. 划分每个点到最近的各个簇
3. 对每个簇计算新的质心
4. 重复2
5. 例子
   - ![c2N5lS](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/c2N5lS.png)

### Hierarchical Methods层次聚类

#### AGNES和DIANA

1. 需要终止条件
2. 分两种，一种是凝聚的层次聚类，一种是分类的层次聚类，分别是自底向上和自顶向下
   1. **AGNES (Agglomerative Nesting)** 自底向上：使用单连接，使用相异度矩阵，合并最小距离的节点，直到所有点都被划分，缺点： 简单，但遇到合并点选择困难的情况。 ⼀旦⼀组对象被合并，不能撤销
   2. ![SR5uRg](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/SR5uRg.png)
   3. **DIANA (Divisive Analysis)** 自顶向下：⾸先将所有的对象初始化到⼀个簇中，然后根据⼀些原则（⽐如最邻近的最⼤欧式距离），将该簇分类。 直到到达⽤户指定的簇数⽬或者两个簇之间的距离 超过了某个阈值。 缺点是已做的分裂操作不能撤销，类之间不能交换对象。如果在 某步没有选择好分裂点，可能会导致低质量的聚类结果。⼤数据集 不太适合
   4. ![Cs5gtf](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Cs5gtf.png)

#### BIRCH

1. ![5Lw3i4](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/5Lw3i4.png)
2. ![TO0BsU](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/TO0BsU.png)
3. ![TlwRD8](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/TlwRD8.png)
4. 大概过程：合并->超过直径->分裂->树太大->从叶子节点开始重构树

#### CURE

1. 是一种结合了采样的方法，每次采样时选离上次选的点，距离最远的
2. ![mLdS93](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/mLdS93.png)
3. ![7RyBfK](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7RyBfK.png)
4. 大概流程：采样->聚类->消除异常点->将未采样的点分配到包含距离最近的点的簇里
5. ![9TKb6B](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/9TKb6B.png)

### Density-Based密度聚类

#### DBSCAN 基础

1. 领域：在一个范围内的圆
2. MinPts：领域内最少点数
3. core object：点的领域内有超过MinPts的点为核心点
4. directly density-reachable 直接密度可达：q在p（核心点）的领域内
5. density-reachable 密度可达：相比密度可达，中间多了其他core object形成链，每个点都是前一个点的直接密度可达
6. density-connected 密度连接：有个中间点o，p和q相对o来说都是密度可达，那么就说p和q是密度连接的

#### 伪代码

![TOtXRj](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/TOtXRj.png)

#### Exercise

![nSPC0v](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/nSPC0v.png)
![IHxwYA](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/IHxwYA.png)

### Grid-based网格聚类

使用多分辨率网格数据结构，主要有 **STING, WaveCluster, CLIQUE**这些方法

#### STING方法

参考：<https://segmentfault.com/p/1210000009787953/read>

1. 空间区域被划分矩形单元
2. 不同level的单元都有多个
3. 每个高层单元被划分成多个低层单元
4. 有两个超参
   1. ⽹格的步⻓——确定空间⽹格划分
   2. 密度阈值——⽹格中对象数量⼤于等于该阈值，表示该⽹格为稠密⽹格
5. 每个单元的统计信息都被提前计算好，用于查询，比如count, mean, s, min, max，以及分布类型——normal/uniform/NONE等（使用假设检验验证分布，什么都不是就为NONE）
6. 可以通过低层单元的统计信息，计算出高层单元的参数：
   - $$n = \sum_i n_i$$
   - $$m = \frac{\sum_i m_i n_i}{n}$$
   - $$s = \sqrt{\frac{\sum_i(s_i^2+m_i^2)n_i}{n}-m^2}$$
   - $$min = min_i(min_i)$$
   - $$max = max_i(max_i)$$
7. Query的过程大概是
   1. 使用自顶向下的方法
   2. 从预选图层开始，通常只有少量的单元
   3. 对每个单元算置信区间
   4. 删掉不相关的单元
   5. 检查完这一层就进入下一层
   6. 重复这一过程直到到底层
8. Exercise
   - ![XMypZk](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/XMypZk.png)

### 异常值检测

通常有三种类型的方法：基于统计的，基于距离的，基于偏差的

#### 基于统计

1. 统计的⽅法先假定在训练集中有⼀个分布模式存在，并且⽤⼀个“ discordancy test ”的⽅法来定义和发现 outliers 。
2. 这个⽅法的应⽤⾸先要设定⼀些参数——假定的分布模式，分布的参数，期望得到的 outlier 的的个数等等
3. ⼯作过程:⾸先检测 2 个假说，⼀个 working hypothesis 和 alternative hypothesis 。
4. working hypothesis ， H, 是说所有的记录的数据都应该遵从⼀个分布模式， F ，然后⽤⼀个 “ discordancy test ”来检测记录 Oi 是否与 F 的关系很⼤。这⼉的“ discordancy test ” 是根据对于数据的了解以及 F 的选择得出的。
5. 通过 T 计算出 Oi 对应的值 vi, 如果 SP(vi)=prob(T>vi) 的值很⼩的话，我们就可以考虑 Oi 是个 outlier
6. ![lbyItQ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/lbyItQ.png)

缺点很明显
- 大多数测试只能针对单一属性
- 很多情况下分布未知
- 要设置输入参数

#### 基于距离

主要是为了克服统计方法中的限制，能不能不直到数据分布，没有统计测试

假设，异常点是那些至少在概率 p 的情况下会距离点 O 至少 d 的点。因此在对象O的d范围内，最多 $(1-p) \times T$数量的对象，圈外最少有$p \times T$的对象

如果⼀个记录的距离⼤于 d 的邻居的个数⼤于⼀个设定值 p 的话，就可以认为这个记录是个 outlier 。换句话说，就是这个记录没有⾜够的邻居数⽬，这⼉的邻居是根据距离来确定的

#### Index-based

给定⼀个数据集，这个算法检查每⼀个记录 o 的 d 半径的邻居个数，定义 M 为⼀个 outlier 的最⼤ d- 邻居个数，那么⼀旦⼀个记录 o 有 M+1 个 d- 邻居，显然这个记录不是个 outlier

多维的index 结构可以使用kd树存，时间复杂度是$O(n^2)$

#### Cell-based

主要就是将其分成一个个cell单元格，计算两层里的节点数，通过邻居数够不够来判断是否是异常值。

这两层的厚度分别是 1 和 $2\sqrt{k}-1$个cell。每个cell边长是$\frac{d}{2\sqrt{k}}$.

```cpp
if(第一层节点数 > M)
   cell里的节点都不是异常值
else if(第一层节点数 <= M && 第二次节点数 <= M)
   cell里的节点都是异常值
else if(第一层节点数 <= M && 第二层节点数 >M)
   一个个检查cell里的节点对象
```

![eRUYW3](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/eRUYW3.png)
![gaTpiF](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/gaTpiF.png)

#### Deviation-based

这个不重要，看下ppt吧，不写了
![jO0zn3](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/jO0zn3.png)

## Association Rules Mining

-----

关联规则挖掘主要是要会计算support， confidence，算法掌握Apriori，FP树找频繁项等。

### Basic

1. 项目集，k项集
2. 规则形式：$\alpha => \beta, \alpha \subset X, \beta \subset X, \alpha \cap \beta = \varnothing$，如果有交集，那肯定可以忽略，比如买牛奶到买牛奶和面包，后面的牛奶就可以忽略了
3. support支持度：$support(\alpha=>\beta) = P(\alpha \cap \beta)$
4. 频繁项集：超过次数大于最小支持度的就是频繁项
5. 频繁项集挖掘：就是找到所有满足最小支持度的$\alpha => \beta$
6. confidence置信度：$confidence(\alpha => \beta) = P(\beta|\alpha) = \frac{P(\alpha \cap \beta)}{P(\alpha)} = \frac{count(\alpha \cap \beta)}{count(\alpha)}$
7. 强规则：满足最小支持度和最小置信度的规则
8. 先找出频繁项，再看是否是强规则
9. lift相关性：$lift = \frac{P(A \cap B)}{P(A)P(B)}$度量相关性，等于1时相互独立，大于1的时候正相关，小于1负相关

> 分别说明利用支持度、置信度和提升度评价关联规则的优缺点。
>
> - 支持度
>   - 优点:支持度高说明这条规则可能适用于数据集中的大部分事务。
>   - 缺点:若支持度阈值过高，则许多潜在的有意义的模式由于包含支持度小的项而被删去;若支持度阈值过低，则计算代价很高而且产生大量的关联模式。
> - 置信度
>   - 优点:置信度高说明如果满足了关联规则的前件，同时满足后件的可能性也非常大。
>   - 缺点:找到负相关的关联规则。
> - 提升度:
>   - 优点:提升度可以评估项集 A 的出现是否能够促进项集 B 的出现
>   - 缺点:会产生出现伪相互独立的规则。


### 一维布尔关联规则

#### Apriori 法则

##### Basic

主要思想就是如果A不是频繁的，那么后面也没必要算包含A的了，包含A的肯定也是不频繁的。

先找一项集，找到频繁的，然后找二项集，再找三项集，每个频繁项集的子集必须是频繁的。

每找到k项集就需要扫描一次
![y2hgVf](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/y2hgVf.png)

##### 伪代码

![jmZ6fb](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/jmZ6fb.png)

##### 如何产生候选集C

看上面例子里的L2到C3就可发现，产生候选集有两个步骤：
1. 对L的k项集进行self-join自交，产生一堆k+1项集
   1. 自交也是有方法的，比如L里的k项集，每个项集按照字母顺序排序后，如果两个k项集的前**k-1**项是相同的，才可连接。比如{a,b,c}和{a,b,d}可以，和{a,c,d}不可以
2. 判断这些k+1项集的所有k项集子集是否是在L中，如果不在就去掉

##### 计算候选集的支持度

这里简单的就自己扫一眼数数，复杂的可以使用hash树计算，参考：[Apriori算法中使用Hash树进行支持度计数](https://blog.csdn.net/george_red/article/details/105411784)

#### 对Apriori的优化

****
1. 只扫描两次数据库
   1. 将数据划分成N个部分
   2. 对每个部分找到本地频繁项集
   3. 集合所有的本地频繁项集，再扫描一次数据库，得到所有的全局频繁项集
   4. 任何潜在的频繁项集至少会在本地频繁项集中出现1次
2. 减少候选集个数，DHP
   1. 生成频繁k项集$L_k$的时候，同时产生所有k+1项集
   2. 对所有的k+1项集进行hash，hash到桶buckets里，记录bucket的cnt
   3. 如果一个k+1项集的bucket计数小于最小支持度，就从$C_{k+1}$里面移除这些
   4. 主要是应用了：任何k项集的hash bucket小于最小支持度就不会是频繁的
   5. 例子:![2bWtWQ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/2bWtWQ.png)
3. 减少transaction
   1. 扫描transaction生成k项集的时候，对不包含k项候选集的transaction进行标记
   2. 移除所有被标记的transaction
4. 减少扫描次数：DIC
   1. 将数据划分成块，并标记为起始点
   2. 一单某个候选集的所有子集都被确定为频繁项集的时候，它就可以被加入任意的起始点
   3. Exercise
5. 避免产生候选集：FP树

#### FP树

重点，除了下面的PPT截图，还可以参考：[FP Tree算法原理总结-刘建平](https://www.cnblogs.com/pinard/p/6307064.html)和[Frequent Pattern 挖掘之二(FP Growth算法)](http://blog.sina.com.cn/s/blog_68ffc7a40100uebg.html)

##### Basic

1. 扫描一次数据，得到频繁一项集
2. 将它们按照频率降序排序
3. 新建一个根节点为null
4. 把每个transaction按照之前的顺序排序，然后再扫描一次数据，构建fp树
5. 每个transaction都会产生一个branch，生成的时候，每次路过同样的路径，就对路上的节点计数加一
6. 创建header table，并把每个项连起来

##### 创建FP树

![tWJQaf](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/tWJQaf.png)

##### 创建条件模式基

从header table开始向上回溯到根节点，每个路径的计数是最下面对应的item的计数，条件模式基就是所有的前缀路径，不包括节点本身的。
![XXwzEl](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/XXwzEl.png)

##### 从条件模式基到条件FP树

对于每个item的条件模式基，把前缀的每个item的值累加起来，然后去掉小于最小支持度计数的item，然后就可以构建一个单一路径的条件FP树。从FP树叶子节点向上地柜，就可以得到所有的和当前item相关的频繁项。

其实，条件FP树就是所有条件模式基合并起来的一条路径，去掉了不满足min_support的item，条件FP树上的计数是累加的。
有个特殊情况是，如果两条路径，里面都有节点m，单独看这两个路径里的m是不满足最小支持度计数的，但是合并起来就满足，那也是算的，不能删了。

例子：
![OejJI0](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/OejJI0.png)

条件FP树是递归创建的：
![ZMR8s2](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ZMR8s2.png)

##### Exercise

![GROq8p](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/GROq8p.png)
![pVAcgK](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/pVAcgK.png)

##### 对比Apriori

- FP-Growth的效率比Apriori高，尤其是最小支持度比较低的时候。
- 相比Apriori方法，FP-Growth通过记录树结构，根据到目前为止获得的频繁模式分解挖掘任务和数据库，同时，没有像Apriori那样产生候选子集，也不用去测试候选集是否满足最小支持度。
- 关键的是，在扫描数据库方面，FP-Growth只需要扫描数据库两次，降低了运行时间。
- 最后，FP-Growth是计算本地频繁项集并以此构建子FP树，无模式搜索和匹配。

### 多层级关联规则

- 自顶向下的
- 每层使用Apriori
- 如果祖先是非频繁的，那就不用搜索子孙了

1. uniform support
   - 每层使用uniform min support
   - 缺点是如果设比较高的阈值，会丢失很多关联规则，如果阈值较低就会产生很多不感兴趣的规则
2. reduced support
   - 每层有自己的min support
   - level越低，support越低
   - 如果祖先是非频繁的，容易丢失很多低层级的频繁项
3. 优化reduced support
   - 控制通过每层的最小支持度
   - next level min sup < level passage threshold < min sup
   - 允许检查那些祖先不满足min_sup的节点，只要祖先节点满足level passage threshold
4. 过滤冗余规则
   - 会产生很多规则冗余，祖先a->b和子孙c->b，如果子孙规则的期望支持度和祖先差不多，就认为是冗余的

### 多维关联规则

挖掘关联规则分两步：
1. 生成频繁项集
2. 对每个频繁项集(XY)进行二元划分，生成confidence最高的一系列规则(X->Y)

和一维关联规则的区别：
![IlCj4s](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/IlCj4s.png)

对于定量属性的处理（连续值）：
1. 基于层次概念的静态离散化
   1. 连续数值被范围代替
   2. 找到所有k项集需要扫描k次或k+1次
   3. 使用数据立方体比较合适
2. 基于数据分布的动态离散化
3. 基于距离的规则聚类（先对一维聚类然后关联）
   1. 关联规则聚类系统，ARCS
   2. binning，2维的网格
   3. 扫描数据，计算每个cell的支持度，找到频繁谓词集
   4. 对规则聚类，相邻的cell表示为一个规则
   5. ARCS的例子：![7aYsZV](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/7aYsZV.png)

## Recommend

-----

考点不多

### Basic

1. 对比搜索系统：
   - Search engine 要用户主动query
   - recommendation 不需要
   - Search engine的结果不可以个性化
2. 结果通常提供相关度排名

### Content-based Methods

基本思路是：物品的描述和兴趣，越相似越推荐；找到用户间兴趣和已有物品描述的相似性

#### 步骤

步骤很简单：
1. 描述推荐的物品
2. 创建描述用户喜欢类型的物品的用户画像
3. 使用向量表示物品和用户画像，例如TF-IDF方法
4. 比较用户画像和物品，例如使用余弦相似度
   - $$ sim(U_i, I_j) = cos(U_i, I_j) = \frac{\sum_{i=1}^k u_{i,l}i_{j,l}}{\sqrt{\sum_{l=1}^k u_{i,l}^2}\sqrt{\sum_{l=1}^k i_{j,l}^2}}$$

#### 协同过滤

- user-based：评价rating相似的用户可能对新东西的rate相似
- item-based：评价相似的物品，以后的评价也相似

- 计算用户 or 物品的相似度
  - 余弦相似度，对应项乘积的和除以各自的L2范数
  - $$ sim(U_i, U_j) = cos(U_i, U_j) = \frac{\sum_{i=1}^k u_{i,l}u_{j,l}}{\sqrt{\sum_{l=1}^k u_{u,l}^2}\sqrt{\sum_{l=1}^k u_{j,l}^2}}$$
  - 皮尔逊相关系数，各自减去均值后，再计算余弦相似度
  - $$sim(U_i, U_j) = \frac{\sum_k (r_{i,k}-\bar{r_i})(r_{j,k}-\bar{r_j})}{\sqrt{\sum_k(r_{i,k}-\bar{r_i})^2}\sqrt{\sum_k(r_{j,k}-\bar{r_j})^2}}$$
- 更新评价
  - $r_{u,i}$: 预测的用户u对物品i的评价
  - $\bar{r_u}$: 用户u的平均评价
  - $r_{v,i}$: 观测到的用户v对物品i的评价
  - $\bar{r_v}$: 用户v的平均评价
  - $$r_{u,i} = \bar{r_u} + \frac{\sum_{v \in N(u)}sim(u,v)(r_{v,i}-\bar{r_v})}{\sum_{v\in N(u)}sim(u,v)} $$

Exercise：
![3lffjQ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/3lffjQ.png)
![Cf9tdJ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Cf9tdJ.png)

#### 评价指标

1. $MAE = \frac{\sum_{ij}|\hat{r_{ij}}-r_{ij}|}{n}$
2. $NMAE = \frac{MAE}{(r_{max}-r_{min})}$
3. $RMSE = \sqrt{\frac{\sum_{i,j}(\hat{r_{ij}}-r_{ij})^2}{n}}$
4. $Precision = P = \frac{TP}{TP+FP}$
5. $Recall = R = \frac{TP}{TP+FN}$
6. Spearman's Rank Correlation
7. Kendall’s 𝜏
   - ![cxdMgj](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/cxdMgj.png)
   - ![ZWfHFu](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ZWfHFu.png)

## 注意点

-----

1. FP树挖掘找频繁项集的时候，条件模式基和条件FP树注意不要合并错
2. k-means划分节点到簇的时候，用的不是欧式距离，没有开根号，知识平方和
3. binning方法进行smooth的时候，要先进行排序，再划分
4. 频繁项集挖掘关联规则的时候，对元素个数大于1的项集进行的是二元划分
5. 决策树计算信息熵和信息增益的时候，是算$log_2X$，计算器上默认是以10为底，要注意
6. 分布式计算那里一般的思路都是map-reduce，照搬大概写就行

## 期末考题

-----

1. 设计数据仓库，star schema
2. binning smooth的题，还有算min,max,median,Q1,Q3,IQR，找异常值
3. 简答题（1）什么是过拟合（2）决策树避免过拟合（3）简述一个决策树分裂属性的方法（4）神经网络怎么避免过拟合
4. FP树的题，找频繁项和强关联规则
5. 推荐系统的题，算相似度和评分
6. 根据similarity matrix使用AGNES凝聚层次聚类，图和上面的差不多
7. 大规模流式数据，要求只能扫描一次，基于k-means处理。要求能够体现簇的演变和处理新数据。
