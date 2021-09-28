---
title:      LTR精排序笔记整理[转]
subtitle:   Learning to Rank
date:       2021-09-17 20:11:01
tags:
    - LTR
    - Ranking
    - Notes
categories: MachineLearning
mathjax: true
urlname: ltr
---

# 前言

看到的一篇博客，转自[LTR精排序](https://ningshixian.github.io/2020/07/21/LTR%E7%B2%BE%E6%8E%92%E5%BA%8F/)

Elasticsearch 内部有几种不同的算法来评估相关度(relevance)：TF-IDF、BM25、BM25F等。但是这种模型 $f(q,d)$ 其实是简单的概率模型，评估 query 在 doc 中出现的位置、次数、频率等因素，但是不能有效的根据用户反馈（点击／停留时间）来持续优化搜索结果。比如说用户搜索了某些关键词，点击了某些结果，而这些结果并不是排在最前面的，但确实是用户最想要的。那有没有什么方法可以使它们排在前面呢？

 <!-- more -->

- 什么是排序学习？

- Ranking模型评价指标（NDCG）

- Ranking模型的三种训练方式

- LTR三大挑战

- Ranking检索排序实践
  - 检索召回
  - 精准排序
  
- Ranking开源库

- Ranking在大厂的应用

- 排序模型的应用

- 拓展：LTR常用算法详细介绍

  拓展：LTR模型详细介绍

  - Pointwise Ranker（LR / GBDT / DNN）
  - Pairwise Ranker（RankSVM / RankNet / LambdaRank / LambdaMART）
  - Listwise Ranker（LambdaRank / LambdaMART）

# Learning to Rank

信息展示个性化的典型应用主要包括搜索列表、推荐列表、广告展示等等。很多人不知道的是，看似简单的个性化信息展示背后，涉及大量的数据、算法以及工程架构技术，这些足以让大部分互联网公司望而却步。究其根本原因，个性化信息展示背后的技术是排序学习问题（Learning to Rank）。-----[深入浅出排序学习：写给程序员的算法系统开发实践](https://tech.meituan.com/2018/12/20/head-in-l2r.html)

## 什么是排序学习？

传统的排序方法通过构造相关度函数，按照相关度对于每一个文档进行打分，得分较高的文档，排的位置就靠前。但是，随着相关度函数中特征的增多，使调参变得极其的困难。所以我们自然而然的想到使用机器学习来优化搜索排序，**这就导致了排序学习 LTR （Learning to rank）的诞生**。

![](https://ningshixian.github.io/resources/images/排序学习.png)

可以将LTR归纳为这样一个问题：基于q-d特征向量，通过机器学习算法训练来学习一个最佳的拟合公式，从而对文档打分并排序。在这个过程中，可以选定学习器（如神经网络、树模型、线性模型等），去自动学习对应的权重应该是多少？学习权重的时候需要确定损失函数，以最小化损失函数为目标进行优化即可得到学习器的相关参数。损失函数的作用相当于告诉学习器应该沿着那个方向去优化，才能保证获得最优的权重。

**LTR的目标**

排序学习是机器学习在信息检索系统里的应用，其目标是构建一个排序模型，根据用户的行为反馈来对多路召回的 item 进行打分，从而实现对多个条目的排序。LTR 相比传统的排序方法，优势有很多:

- 整合大量复杂特征并自动进行参数调整，自动学习最优参数；
- 融合多方面的排序影响因素，降低了单一考虑排序因素的风险；
- 能够通过众多有效手段规避过拟合问题。
- 实现个性化需求（推荐）
- 多种召回策略的融合排序推荐（推荐）



## 排序评价指标

### Binary Relevance

**Precision and Recall(P-R)**

本评价体系通过准确率（Precision）和召回率（Recall）两个指标来共同衡量列表的排序质量。对于一个请求关键词，所有的文档被标记为相关和不相关两种。

Precision的定义如下：

<img src="https://awps-assets.meituan.net/mit-x/blog-images-bundle-2018b/5847cc5c.png" alt="img" style="zoom:50%;" />

Recall的定义如下：

<img src="https://awps-assets.meituan.net/mit-x/blog-images-bundle-2018b/19942480.png" alt="img" style="zoom:50%;" />

**Mean Average Precision (MAP):**

Average Precision (AP):
$$
\mathrm{AP}(\pi, l)=\frac{\sum_{k=1}^{m} P @ k \cdot I_\left\{l_{\pi-1}(k)=1\right\}}{m_{1}}
$$
如果想要衡量一个模型的好坏，就需要对多次排序结果的AP值求平均，这样就得到MAP. MAP它反映了查询结果中前 k 名的相关性。

**MRR(Mean Reciprocal Rank)**

对于给定查询q，如果第一个相关的文档的位置是R(q)，则$MRR(q)=1/R(q)=rank_i$。
$$
M R R=\frac{1}{|Q|} \sum_{i=1}^{|Q|} \frac{1}{\operatorname{rank}_{i}}
$$



### Graded Relevance

**Discounted Cumulative Gain(DCG)**

P-R的有两个明显缺点：

- 所有文章只被分为相关和不相关两档，分类显然太粗糙。
- 没有考虑位置因素。

DCG解决了这两个问题。对于一个关键词，所有的文档可以分为多个相关性级别，这里以rel1，rel2…来表示。文章相关性对整个列表评价指标的贡献随着位置的增加而对数衰减，位置越靠后，衰减越严重。

系统通常为某用户返回一个item列表，假设列表长度为K，这时可以用NDCG@K评价该排序列表与用户真实交互列表的差距。

**解释：**

- **Gain：** 表示列表中每一个item的相关性分数（标签增益）
  $$
  Gain(i)={2^{r(i)}-1}
  $$
  其中，${r(i)}$表示第 *i* 个文档的真实相关度;
  
- **Cumulative Gain：**表示对K个item的Gain进行累加
  $$
  \mathrm{CG}@{K}=\sum_{i=1}^{K} Gain(i)
  $$

- **Discounted Cumulative Gain：**考虑排序顺序的因素，使得排名靠前的item增益更高，对排名靠后的item进行折损。(把每个位置对应的标签增益乘以位置折扣，最后累加到一起)
  $$
  \sum_{i=1}^{k} Gain\left(i\right) \eta(i)
  $$
  
  位置折扣 $\eta(i)$ 的常用设置：

$$
n(j)=\frac{1}{\log (j+1)}
$$

​		从中可以看出DCG的一个特性：排名靠前的doc增益更高，对排名靠后的doc进行折损

- **Normalized Discounted Cumulative Gain：**DCG能够对一个用户的推荐列表进行评价，如果用该指标评价某个推荐算法，需要对所有用户的推荐列表进行评价，而不同用户之间的DCG相比没有意义（不同用户兴趣不同，对同样的item排名前后不同）。所以要对不同用户的指标进行归一化，自然的想法就是对每个用户计算一个理想情况（或者说是实际情况）下的DCG分数，用IDCG表示，然后用每个用户的DCG与IDCG之比作为每个用户归一化后的分值，最后对每个用户取平均得到最终的分值，即NDCG。
  $$
  \mathrm{NDCG}@{K}=\frac{DCG@{K}}{IDCG}
  $$
  **IDCG的定义如下：**
  $$
  \mathrm{IDCG}_{\mathrm{K}}=\sum_{i=1}^{|R E L|} \frac{2^{r(i)}-1}{\log _{2}(i+1)}
  $$
  DCG中的相关性得分$r(i)$是根据预测概率，来对相关度label进行排序；

  IDCG中的相关性得分$r(i)$是根据实际label大小，来对相关度label进行排序；
  
  **图示：**
  
  ![img](https://pic3.zhimg.com/80/v2-8a73036dda12c326fecc9de67481ba0a_1440w.jpg)

*PS: NDCG(Normalized Discounted Cumulative Gain) 是一种综合考虑模型排序结果和真实序列之间的关系的一种指标，也是最常用的衡量排序结果的指标，越相关的结果排到越前，分数越高*；

*NDCG的形象理解 https://blog.csdn.net/anshuai_aw1/article/details/83117012*

NDCG的Python实现可参考[链接](https://www.kaggle.com/wendykan/ndcg-example)



**Expected Reciprocal Rank(ERR)**

与DCG相比，除了考虑位置衰减和允许多种相关级别（以R1，R2，R3…来表示）以外，ERR更进了一步，还考虑了排在文档之前所有文档的相关性。举个例子来说，文档A非常相关，排在第5位。如果排在前面的4个文档相关度都不高，那么文档A对列表的贡献就很大。反过来，如果前面4个文档相关度很大，已经完全解决了用户的搜索需求，用户根本就不会点击第5个位置的文档，那么文档A对列表的贡献就不大。ERR的定义如下：

$$
E R R=\sum_{\mathrm{r}=1}^{\mathrm{n}} \frac{1}{\mathrm{r}} \prod_{\mathrm{i}=1}^{\mathrm{r}-1}\left(1-R_{\mathrm{i}}\right) R_{r}
$$



## LTR训练算法

> 参考《[推荐系统中排序学习的三种设计思路](https://mp.weixin.qq.com/s?__biz=MzA4NTUxNTE4Ng==&mid=2247483834&idx=1&sn=3be93ae88b28d5e07bcd880825c55851&chksm=9fd78f67a8a0067123a08bc470b8587a96e2d7287ccda44d84a978fda5bf13a86a5920909d71&scene=21#wechat_redirect)》

在训练阶段输入是n个query对应的doc集合，通常数据来源有两种，一种是人工标注，即通过对实际系统中用户query返回的doc集合进行相关性标注，标签打分可以是三分制（相关，不相关，弱相关），也可以是更细的打分标准。另外一种是**点击日志中获取**，通过对一段时间内的点击信息进行处理获得优质的点击数据。**这些输入的doc的表示形式是多个维度的特征向量**，特征的设计也尤其重要，对网页系统检索而言，常用的有查询与文档匹配特征，其中细化了很多角度的匹配，比如紧密度匹配，语义匹配，精准匹配等等，还有通过将文档分为不同域后的各个域的匹配特征，关键词匹配特征，bm系列特征, 以及通过dnn学习得到的端到端的匹配特征。

通过排序模型的不断迭代，当一个用户输入一个query之后，排序系统会根据现有模型计算各个doc在当前特征下的得分，并根据得分进行排序返回给用户。

Learning to rank需要以排序的评价指标作为优化目标，来让整体排序效果更佳。但是评判排序好坏的这些指标的特点是**不平滑、不连续（?）**，无法求梯度，因此无法直接用梯度下降法求解。所以诞生了3种近似求解的训练方法（优化策略）：

- **Pointwise**
- **Pairwise**
- **Listwise**

其中pointwise将其转化为多类分类或者回归问题，pairwise将其转化为pair分类问题，Listwise为对查询下的一整个候选集作为学习目标，**三者的区别在于loss不同、以及相应的标签标注方式和优化方法的不同。**

![](https://ningshixian.github.io/resources/images/point-pair-list.png)



### Pointwise

该方法将排序问题转化为分类问题或者回归问题，只考虑query和单个文档的绝对相关度作为标签，而不考虑其他文档和query的相关度。比如标签有三档，问题就变为多类分类问题,模型有McRank,svm，最大熵等。另一种思想是将query与doc之间的相关程度作为分数利用回归模型拟合，经典回归模型有线性回归，dnn，mart等。

输入数据应该是如下的形式：

![img](https://pic1.zhimg.com/80/v2-99a39cb859348c624cec188b76582cc4_1440w.jpg)

- 输入空间：doc 的特征向量 $(x_i,y)$

- 输出空间：每个doc的相关性分数

- 损失函数：回归loss，分类loss
  $$
  L(f(x),y)=\sum_{i=1}^n{l(f(x_i),y_i)}=\sum_{i=1}^n{f(x_i)log(y_i)}
  $$
  

**Pointwise 方法存在的问题：**

Pointwise 方法通过优化损失函数求解最优的参数，可以看到 Pointwise 方法非常简单，工程上也易实现，但是 Pointwise 也存在很多问题：

- Pointwise 只考虑单个文档同 query 的相关性，没有考虑文档间的关系，然而排序追求的是排序结果，并不要求精确打分，只要有相对打分即可；
- 通过分类只是把不同的文档做了一个简单的区分，同一个类别里的文档则无法深入区别，虽然我们可以根据预测的概率来区别，但实际上，这个概率只是准确度概率，并不是真正的排序靠前的预测概率；
- Pointwise 方法并没有考虑同一个 query 对应的文档间的内部依赖性。一方面，导致输入空间内的样本不是 IID 的，违反了 ML 的基本假设，另一方面，没有充分利用这种样本间的结构性。其次，当不同 query 对应不同数量的文档时，整体 loss 将容易被对应文档数量大的 query 组所支配，应该每组 query 都是等价的才合理。
- 很多时候，排序结果的 Top N 条的顺序重要性远比剩下全部顺序重要性要高，因为损失函数没有相对排序位置信息，这样会使损失函数可能无意的过多强调那些不重要的 docs，即那些排序在后面对用户体验影响小的 doc，所以对于位置靠前但是排序错误的文档应该加大惩罚。



### Pairwise

**Pairwise**是目前比较流行的方法，相对**pointwise**他将重点转向**文档顺序关系**。比如对于query 1返回的三个文档 doc1，doc2，doc3, 有三种组对方式，<doc1,doc2>,<doc2,doc3>,<doc1,doc3>。

![img](https://imgconvert.csdnimg.cn/aHR0cHM6Ly9tbWJpei5xcGljLmNuL21tYml6X3BuZy9iUVg0aWIyUnV0aEJ2Tk9kUG0zYW1Yb1pFS3dnSkVxVTNwbnhhUjN4T2JqeDBNMzBpYUFWaWJDOUlnY0pBeE96UVRnUW9ZTFAyRFI5NmZpYk9LUzdnTkVnTncvNjQw?x-oss-process=image/format,png)

- 输入空间：同一查询的一对文档 $(x_i,x_j,y)$

- 输出空间：两个文档的相对关系 $r_{ij} \in \{-1, +1\}$​

- 损失函数：铰链损失( hinge loss)、交叉熵损失( cross entropy loss)
  - ir 领域一个被广泛使用的 pairwise loss 就是 Hingle Loss，其基本形式为：
  - $$ \mathcal{L}(f ; \mathcal{Q}, \mathcal{D}, \mathcal{Y})=\sum_{i} \sum_{ \{j,k\}, y_{i,j} \succ y_{i,k} } \max \left(0,1-f \left( q_{i}, d_{i,j} \right)+f \left(q_{i}, d_{i, k} \right) \right) $$
  - 还有一个 Pairwise 的损失函数就是上文提到的 RankNet 中使用的基于交叉熵的损失函数及其改进型 LambdaRank，但是应用似乎并不广泛，可能原因是在 IR 的 Matching 任务中，query 对应的 doc 之间没有很明显的等级关系，而通常是 0-1 的相关与否，并且多数只有一个正样本，因此 RankNet 中的 Pairwise Loss 并不能有效的起到作用。
  - $$ C=-\frac{1}{n} \sum_{x}[y \ln a+(1-y) \ln (1-a)] $$

  式中，x 表示样本; n 表示样本总数; y 为期望输 出; a 为神经元实际输出，并且交叉熵具有非负性，当 a 和 y 接近时，损失函数接近于 0。
- 常用算法：Ranking SVM (ICANN 1999)、RankBoost (JMLR 2003)、RankNet (ICML 2005)、lambdaRank (NIPS 2006)、LambdaMART (inf.retr 2010)

**这里讨论下，关于人工标注标签怎么转换到pairwise类方法的输出空间：**

Pairwise 方法的输出空间应该是包括所有文档的两两文档对的偏序关系（pairwise preference），其取值为 {+1,−1}，+1 表示文档对中前者更相关，-1 则表示文档对中后者更相关，我们可以把不同类型的人工标注的标签转换到 Pairwise 的真实标签。

1. 如果标注直接是相关度 $s_j$ (即单点标注)，则文档对 $(x_u,x_v)$ 的真实标签定义为 $y_{u,v}=2*I_{s_u>s_v}-1$，相关度打分 $l_j$，
2. 如果标注是 pairwise preference $s_{u,v}$，则 doc pair $(x_u,x_v)$ 的真实标签定义为 $y_{u,v}=s_{u,v}$
3. 如果标注是整体排序 π，则 doc pair $(x_u,x_v)$ 的真实标签定义为 $y_{u,v}=2*I_{π_u,π_v}-1$

**Pairwise 方法存在的问题：**

Pairwise 方法通过考虑两两文档之间的相关对顺序来进行排序，相比 Pointwise 方法有明显改善。但 Pairwise 方法仍有如下问题：

- 使用的是两文档之间相关度的损失函数，而它和真正衡量排序效果的指标之间存在很大不同，甚至可能是负相关的，如可能出现 Pairwise Loss 越来越低，但 NDCG 分数也越来越低的现象。
- 只考虑了两个文档的先后顺序，且没有考虑文档在搜索列表中出现的位置，导致最终排序效果并不理想。
- 不同的查询，其相关文档数量差异很大，转换为文档对之后，有的查询可能有几百对文档，有的可能只有几十个，这样不加均一化地在一起学习，模型会优先考虑文档对数量多的查询，减少这些查询的 loss，最终对机器学习的效果评价造成困难。
- Pairwise 方法的训练样例是偏序文档对，它将对文档的排序转化为对不同文档与查询相关性大小关系的预测；因此，如果因某个文档相关性被预测错误，或文档对的两个文档相关性均被预测错误，则会影响与之关联的其它文档，进而引起连锁反应并影响最终排序结果。



### Listwise

Pointwise将训练集里每一个文档当做一个训练实例，pairwise方法将同一个査询的搜索结果里任意两个文档对作为一个训练实例，Listwise方法是将每一个查询对应的所有搜索结果列表整体作为一个训练实例，根据这些训练样例训练得到最优评分函数 $F()$ 

- 输入空间：Set of documents $x=\{x_j\}_{j=1}^m$
- 输出空间：Ranked list $\pi_y$
- 假设空间：Sort function $F(x)$
- 损失函数：常见的模型比如ListNet使用正确排序与预测排序的排列概率分布之间的**KL距离**（交叉熵）作为损失函数，LambdaMART的框架其实就是MART，主要的创新在于中间计算的梯度使用的是Lambda，**将IR评价机制如NDCG这种无法求导的IR评价指标转换成可以求导的函数**，并且富有了梯度的实际物理意义

PS: 我们的输入是整个List，直接优化NDCG。但NDCG非凸且不可导，该怎么办呢? **既然无法定义一个符合要求的loss，那就直接定义一个符合我们要求的梯度**。[*策略算法工程师之路-损失函数设计](https://zhuanlan.zhihu.com/p/94596648?utm_source=wechat_session&utm_medium=social&utm_oi=30098367447040&utm_campaign=shareopn)

<img src="https://pic1.zhimg.com/80/v2-8f2b57edebd2edb62a9e9fbeeb0fe604_1440w.jpg" alt="img" style="zoom: 67%;" />

上述推导过程就不必细看了，直接看最后的结果。

<img src="https://pic1.zhimg.com/80/v2-a3a3fa238349056b1564671aa11d78c8_1440w.jpg" alt="img" style="zoom:50%;" />

而我们可以根据业务需求自定义:

![img](https://pic1.zhimg.com/80/v2-860fd587bf879df1338a94884d3e6c6c_1440w.png)

既然如此回到我们的优化目标NDCG，就可以定义:

![img](https://pic1.zhimg.com/80/v2-aa874218075fae15f45ca900ffb083b4_1440w.png)

为对特定List，将Ui,Uj的位置调换，NDCG值的变化。所以我们就做到在NDCG非凸且不可导的情况下直接优化NDCG了。

- 常用算法：AdaRank (SIGIR 2007)、SVM-MAP (SIGIR 2007)、SoftRank (LR4IR 2007)、ListNet (ICML 2007)、LambdaMART (inf.retr 2010)（也可以做Listwise）

**推荐中使用较多的 Listwise 方法是 LambdaMART**（LambdaMART是LambdaRank的提升树版本，而LambdaRank又是基于pairwise的RankNet。因此LambdaMART本质上也是属于pairwise排序算法，只不过引入Lambda梯度后，还显示的考察了列表级的排序指标，如NDCG等，因此，也可以算作是listwise的排序算法！！！）



**列表法和配对法的中间解法**

第三种思路某种程度上说是第一种思路的一个分支，因为很特别，这里单独列出来。其特点是在纯列表法和配对法之间寻求一种中间解法。具体来说，这类思路的核心思想，是从 NDCG 等指标中受到启发，设计出一种替代的目标函数。这一步和刚才介绍的第一种思路中的第一个方向有异曲同工之妙，都是希望能够找到替代品。

找到替代品以后，接下来就是把直接优化列表的想法退化成优化某种配对。这第二步就更进一步简化了问题。这个方向的代表方法就是微软发明的 LambdaRank 以及后来的 LambdaMART。微软发明的这个系列算法成了微软的搜索引擎 Bing 的核心算法之一，而且 LambdaMART 也是推荐领域中可能用到一类排序算法。



**[三类 L2R 排序学习算法的优缺点](http://ir.nsfc.gov.cn/paperDownload/ZD3487949.pdf)**

| L2R       | 优点                                   | 缺点                                                         | 输入                  | 时间复杂度 |
| --------- | -------------------------------------- | ------------------------------------------------------------ | --------------------- | ---------- |
| PointWise | 算法简单，只以每个文档为单独训练数据   | 只考虑单个文档同 query 的相关性，没有考虑文档间的关系        | $(x,y)$               | $O(n)$     |
| PairWise  | 对文档对进行分类，得到文档集的偏序关系 | 只考虑每对文档之间的偏序关系 <br />预测错误的文档对可能会引起连锁反应<br />无法衡量真正的排序指标 | $(x_1,x_2,y)$         | $O(n^2)$   |
| ListWise  | 直接对文档结果进行优化，效果最好       | 很难找到合适的优化算法进行求解                               | $(x_1,x_2,...,x_n,y)$ | $O(n!)$    |



<img src="https://nuoku.vip/uploads/pic_bucket/picurl/26/different_letor_approach.jpg" alt="img" style="zoom: 50%;" />

 

<img src="https://nuoku.vip/uploads/pic_bucket/picurl/25/different_models.jpg" alt="img" style="zoom: 67%;" />



<img src="https://img-blog.csdnimg.cn/20190109105730558.PNG?x-oss-process=image/watermark,type_ZmFuZ3poZW5naGVpdGk,shadow_10,text_aHR0cHM6Ly9ibG9nLmNzZG4ubmV0L2Fuc2h1YWlfYXcx,size_16,color_FFFFFF,t_70" alt="img" style="zoom:67%;" />

在工业界用的比较多的应该还是`Pairwise`，因为构建训练样本相对方便，并且复杂度也还可以。



# 在线排序架构三大挑战

在线排序架构主要面临三方面的挑战：特征、模型和召回。

- 特征挑战包括特征添加、特征算子、特征归一化、特征离散化、特征获取、特征服务治理等。
- 模型挑战包括基础模型完备性、级联模型、复合目标、A/B实验支持、模型热加载等。
- 召回挑战包括关键词召回、LBS召回、推荐召回、粗排召回等。

三大挑战内部包含了非常多更细粒度的挑战，孤立地解决每个挑战显然不是好思路。在线排序作为一个被广泛使用的架构值得采用领域模型进行统一解决。Domain-driven design(DDD)的三个原则分别是：领域聚焦、边界清晰、持续集成。

基于以上分析，我们构建了三个在线排序领域模型：召回治理、特征服务治理和在线排序分层模型。



**召回治理**

召回就是找到用户可能喜欢的几百条资讯，召回总体而言分成四大类：

- 关键词召回，我们采用Elasticsearch解决方案。
- 距离召回，我们采用K-D tree的解决方案。
- 粗排召回。
- 推荐类召回。



**特征服务治理**

传统的视角认为特征服务应该分为用户特征（User）、查询特征（Query）和文档特征（Doc），如下图：

![img](https://awps-assets.meituan.net/mit-x/blog-images-bundle-2018b/1bec83fb.png)

这是比较纯粹的业务视角，并不满足DDD的领域架构设计思路。由于特征数量巨大，我们没有办法为每个特征设计一套方案，但是我们可以把特征进行归类，为几类特征分别设计解决方案：

1. Doc特征：类别category_id、总体点击率、最近一周点击率、发布时间、更新时间-新鲜度、标题向量、文章长度、来源source、点赞数、点击反馈、高亮字数
2. Query-Doc相关性特征：向量相似度（cosine、TF-IDF、vsm、BOW）、query 和 doc 之间的词共现特性、点击次数和点击率、bm25
3. context特征：页数、曝光顺序位置SearchOrder、请求时间段等；
4. 后延统计特征：在一段时间内的点击数和点击率CTR、交叉特征
5. 稀疏特征：docID、itemID、answerID、pageNum、pageSize、baseCode、botCode、projectCode

其中ID类特征和交叉特征是离散型特征，相关特征和统计特征是连续型特征. ID 特征也是有重要意义的，在特征工程中应该予以考虑。**用户特征和Query特征暂不考虑。**



**在线排序分层模型**

如下图所示，典型的排序流程包含六个步骤：场景分发（Scene Dispatch）、流量分配（Traffic Distribution）、召回（Recall）、特征抽取（Feature Retrieval）、预测（Prediction）、排序（Ranking）等等。

![img](https://awps-assets.meituan.net/mit-x/blog-images-bundle-2018b/5b01b65a.png)



**智能重排序的实现方案与算法原理**

1. 事先计算型智能重排序

   重排序引擎通过处理用户行为日志，并基于用户兴趣将用户聚类后，再构建排序模型，为每一个聚类生成唯一的排序结果并存放到数据库中，当用户在产品（即APP）上访问该列表时，智能重排序web服务模块将该列表的排序结果取出来并在前端展现给用户。

2. 实时装配型智能重排序

   近实时地计算出用户特征向量与每个标的物特征向量的内积（该内积即用户对该标的物的偏好得分），并对列表中所有标的物的偏好得分降序排列

   - FAISS向量近似搜索方案
   - **TensorFlow Serving方案**，需要事先训练出一个排序模型，将用户特征和item特征按照某种方式灌入该模型

3. 先对所有标的物排序再筛选

   对标的物也只需要计算一次排序，并且排序结果也可以事先存储起来。在提供服务时只需要进行过滤就够了



# 排序任务的公开数据集

- 公开的数据集 (LETOR datasets)
  - 微软 [LETOR3.0](http://research.microsoft.com/en-us/um/beijing/projects/letor/letor3dataset.aspx) and [LETOR 4.0](http://research.microsoft.com/en-us/um/beijing/projects/letor/letor4dataset.aspx)
  - [MSLR-WEB10k and MSLR-WEB30k](http://research.microsoft.com/en-us/projects/mslr/download.aspx)
  - [Yahoo! LETOR dataset](http://webscope.sandbox.yahoo.com/catalog.php?datatype=c)
  - [Yandex imat’2009](http://imat2009.yandex.ru/en/datasets)



# LTR开源库

https://zhuanlan.zhihu.com/p/138436960

1. [LEROT](https://bitbucket.org/ilps/lerot)：用python *在线学习*编写*排名*框架。另外，数据集的列表也不那么详细，但列表更长：[https://bitbucket.org/ilps/lerot#rst-header-data](https://bitbucket.org/ilps/lerot#rst-header-data) 

2. [IPython演示](https://github.com/ogrisel/notebooks/blob/master/Learning to Rank.ipynb)学习排名

3. [LambdaRank的实现](https://github.com/arifqodari/ExpediaLearningToRank) （在Python中专门用于kaggle排名竞赛）

4. **[TF-Ranking](https://mp.weixin.qq.com/s?__biz=MzU1OTMyNDcxMQ==&mid=2247485361&idx=1&sn=fe8dad5e5dfe8baabe4d60af68b32415&chksm=fc184cf9cb6fc5ef562e20ade54be893cf5f8ffbe89c07ed9e5d6b25cac9bb17fa7a73b52865&mpshare=1&scene=1&srcid=0118obA2nmTOwM9mkfe8HZmY&pass_ticket=yoIK672aXk4WPiJRK3zkCxK5C5wwnua1%2B%2F115s%2FKJyXjdHQlvctIkGZpDsP%2FPVPZ#rd) ⭐2K**

   - Github：[TensorFlow Ranking](https://link.zhihu.com/?target=https%3A//github.com/tensorflow/ranking)
   - Steps to get started：[Tutorial: TF-Ranking for sparse features](https://link.zhihu.com/?target=https%3A//github.com/tensorflow/ranking/blob/master/tensorflow_ranking/examples/handling_sparse_features.ipynb/%23)
   - 论文：[TF-Ranking: Scalable TensorFlow Library for Learning-to-Rank](https://arxiv.org/pdf/1812.00073.pdf)

   TF-Ranking 是 google 在 2018 年底开源的基于 tf 设计的 ltr 算法框架，目的是形成可扩展、适用于系数特征的 dnn 的 ltr 算法。TF-Ranking提供了一个统一的框架，包含了一系列**state-of-arts排序学习算法**，并提供了灵活的API，支持了多种特性，如：**自定义pairwise 或 listwise 损失函数**，评分函数，指标等；**非平滑排序度量指标的直接优化LamdaLoss**；**无偏的排序学习方法**。同时，TF-Ranking还能够利用Tensorflow提供的深度学习特性，如稀疏特征的embedding，大规模数据扩展性、泛化性等。

5. XGBoost

   - 官方主页：[Scalable and Flexible Gradient Boosting](https://link.zhihu.com/?target=https%3A//xgboost.ai/)
   - 官方文档：[XGBoost Documentation](https://link.zhihu.com/?target=https%3A//xgboost.readthedocs.io/en/latest/)
   - 开源代码： [eXtreme Gradient Boosting](https://link.zhihu.com/?target=https%3A//github.com/dmlc/xgboost)

   早期版本的 XGBoost 通过目标函数(objective) 中 rank:pairwise 来实现 ltr 任务，官方的教程也是这么写的。这里是一种使用交叉熵的 pairwise 实现，可以看做是不考虑 [公式] 的 LambdaMART。大概是 18 年(记不清了)的时候增加了 rank:ndcg 的实现，也就是完整版的 LambdaMART 实现。配合上 xgboost 在 gbm 基础上的各种优化，理论上 xgboost 中的 LambdaMART 可以有着比 Ranklib 更好的性能。

6. **lightGBM ⭐12.2K**

   - 官方文档：[LightGBM’s documentation](https://link.zhihu.com/?target=https%3A//lightgbm.readthedocs.io/en/latest/)
   - 开源代码：[LightGBM, Light Gradient Boosting Machine](https://link.zhihu.com/?target=https%3A//github.com/microsoft/LightGBM)

   另一个非常著名的 gbm 实现就是有微软开源的 lightGBM，lightGBM 当年在已有 xgboost 在前的情况下通过一系列对模型具体实现的优化从而在速度上占据了优势。

7. [Pyltr](https://github.com/jma127/pyltr/tree/master/pyltr) ⭐346
   Python learning to rank (LTR) toolkit

8. [Ranknet](https://github.com/airalcorn2/RankNet) ⭐187
   My (slightly modified) Keras implementation of RankNet and PyTorch implementation of LambdaRank.

9. [Awesome Search ⭐**68**](https://awesomeopensource.com/project/frutik/awesome-search)

   [it's all about search and its awesomeness](https://awesomeopensource.com/project/frutik/awesome-search)

10. **[DeepCTR](https://github.com/shenweichen/DeepCTR) ⭐4.6K**

    DeepCTR是基于深度学习的CTR模型的**易于使用**，**模块化**和**可扩展的**软件包，以及许多可用于轻松构建自定义模型的核心组件层。包含Wide & Deep、DeepFM、Deep & Cross Network、Deep Interest Network、DIEN、FiBiNET等Models。



# LTR模型在大厂的应用

★ [策略算法工程师之路-排序模型(LTR)及应用](https://zhuanlan.zhihu.com/p/113302654)

- **[WSDM Cup 2020](https://link.zhihu.com/?target=http%3A//www.wsdm-conference.org/2020/wsdm-cup-2020.php) **DiggSci 2020赛题冠军队伍成员
- 给了一堆LTR相关的资料链接和大厂的实践指南！
- 信息量非常大！

[美团DR-BERT模型应用于MACRO文档排序任务](https://tech.meituan.com/2020/08/20/mt-bert-in-document.html)

[EMBEDDING 在大厂推荐场景中的工程化实践](https://mp.weixin.qq.com/s?__biz=MjM5MzY4NzE3MA==&mid=2247486079&idx=1&sn=7d2289dbbba53fce268587d1fb480b27&chksm=a692799291e5f084a19bce5b6a44941474f0c0b4fae011ab12e242b45479caf82fa0d11e2c29&mpshare=1&scene=1&srcid=0816y7i4VKbYDYtlbzk2qLNy&sharer_sharetime=1597557030366&sharer_shareid=52006a0d19edf83d2b8be98f4d8fe935&key=c3402f98b9ff364623c486d950fdff57148711d1475a71313f9a1df4b277753256c1649706970b79f12138e010d5dee579acc86d23f7607b43510110d17b45ca33c685b259b7f303d07877cdef1612cb016db00342300829d03ef458cb76ffcef814261850c9f1c7a9b0949ee4c2247808151a5adf9afb7968fde105d1f335a4&ascene=1&uin=MjM2MDA1NjcyMQ%3D%3D&devicetype=Windows+10+x64&version=62090529&lang=zh_CN&exportkey=A%2BIM%2FZ8Z3e6T5EIVHlNEggo%3D&pass_ticket=z2Figv09I3ObJ0R7A0vTQDfbvJypeSv2uOlWkIAfXeTdj9qeEfRii8WABXVIsaYM)

[机器学习驱动的Airbnb体验搜索排名](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789)

[小菜鸡. 搜索排序算法]([http://octopuscoder.github.io/2020/01/19/%E6%90%9C%E7%B4%A2%E6%8E%92%E5%BA%8F%E7%AE%97%E6%B3%95/](http://octopuscoder.github.io/2020/01/19/搜索排序算法/))

[58同镇下沉市场中的推荐技术实践](https://mp.weixin.qq.com/s?__biz=MzA3MTU2ODMzNA==&mid=2247486773&idx=1&sn=05275aa510399038564f40a33eb169a1&chksm=9f2ad394a85d5a82d57fd50545206f18c00d6a067fdb6a51bace422964c386260db8812850ef&mpshare=1&scene=1&srcid=1108qhY6l2v35EWwgeF7mQOH&sharer_sharetime=1604844718141&sharer_shareid=7470b5f543d9cda449788eaef8277a5c&key=5399cec99f1fa29b8011ae798092c4b7c56f407ae4febd5880169bab7b55ecaf5943e30bc520fa7e1de5799123a5a43654e9156c79d0f22794188ec5fc2810f65cd7bf3b14113c0c45ed238c94a46ce222f8366bb0436e47fecb4fe3e512f2820700337cc53b00463c5cb4ed9937635fb55831d0b7cb5522be1ad7aea4b2a158&ascene=1&uin=MjM2MDA1NjcyMQ%3D%3D&devicetype=Windows+10+x64&version=6300002f&lang=zh_CN&exportkey=A0q%2FW4drhgX%2Fhdf0MISfuBs%3D&pass_ticket=GGe1BYM%2BwbKBFgwM0vEsAvwHEbKPQudDK6tQQFLnByTgbJQQy1MrWdfZ0ecMT4N7&wx_header=0)

[深度学习在58商业排序CTR预估建模中的一些实践经验](https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247506940&idx=2&sn=df6d88efa9f45dc7e2e4592018979692&chksm=fbd76990cca0e08637035afb74fb94cfaec3250c81c8744f41ac6be316d2bee96db16c9fc56c&mpshare=1&scene=1&srcid=0903jLTHvXrlguMR91jT2bdL&sharer_sharetime=1599142216899&sharer_shareid=7470b5f543d9cda449788eaef8277a5c&key=5399cec99f1fa29bb5456f768e6d28663acac453358b8844f624e72553f4252eb39e3fa12777379a4303e8005a254e4f310c359ede81504878564af2182e30e7c632110b7e3b2eb719aa67c628eaa304dff9c67aa3beff61c4d16278f10428c93ef1fb296e15abb0e4cfa7d730bed20b40ee8c16223fb0e5704304c678a8a4e6&ascene=1&uin=MjM2MDA1NjcyMQ%3D%3D&devicetype=Windows+10+x64&version=6300002f&lang=zh_CN&exportkey=AwAwgY0c6PPxtKAtg0YufkI%3D&pass_ticket=GGe1BYM%2BwbKBFgwM0vEsAvwHEbKPQudDK6tQQFLnByTgbJQQy1MrWdfZ0ecMT4N7&wx_header=0)

[深度学习在58租房搜索排序的应用](https://mp.weixin.qq.com/s/cMyXnqKbemt4kBpLMgl8GA)

[大众点评搜索基于知识图谱的深度学习排序实践---LambdaDNN](https://tech.meituan.com/2019/01/17/dianping-search-deeplearning.html)

[2020年精排模型调研](https://hub.baai.ac.cn/view/6299?from=timeline)

[知乎搜索排序模型的演进](https://mp.weixin.qq.com/s/DZZ_BCiNw0EZg7V0KvhXVw)



# 检索排序探索与实践❤❤❤

> [WSDM Cup 2020检索排序评测任务第一名经验总结](https://www.zhuanzhi.ai/document/24ad9f7f873da90dc072b0c777e1acfe)

检索排序包括“检索召回”和“精准排序”两个阶段。其中，检索召回阶段负责从候选集中高效快速地召回候选Documents，从而缩减问题规模，降低排序阶段的复杂度，此阶段注重召回算法的效率和召回率；精准排序阶段负责对召回数据进行重排序，采用Learning to Rank相关策略进行排序最优解求解。

典型的搜索排序算法框架如下图所示，分为线下训练和线上排序两个部分。模型包括相关性模型、时效性模型、个性化模型和**点击模型**等。特征包括Query特征、Doc特征、User特征和Query-Doc匹配特征等。日志包括展现日志、点击日志和Query日志。

<img src="https://ningshixian.github.io/resources/images/排序核心处理流程.png" style="zoom: 50%;" />



**点击模型介绍**

点击模型又称为**点击调权**，搜索引擎根据用户对搜索结果的点击，可以挖掘出哪些结果更符合查询的需求。点击模型基于如下基本假设：

1. 用户的浏览顺序是从上至下的。
2. 需求满足好的结果，整体点击率一定高。
3. 同一个query下，用户点击的最后一个结果之后的结果，可以假设用户已经不会去查看了(一定程度上减弱了位置偏见)。
4. 用户进行了翻页操作，或者有效的query变换，则可以认为前页的结果用户都浏览过，并且不太满意。
5. 用户点击的结果，如果引发后继的转化行为（比如电商搜索中的加购物车），则更有可能是用户满意的结果。



## 检索召回（Recall）

- **目标任务**：使用高效的匹配算法对候选集进行粗筛，为后续精排阶段缩减候选排序的数据规模。
- **性能要求**：召回阶段的方案需要权衡召回覆盖率和算法效率两个指标，一方面召回覆盖率决定了后续精排算法的效果上限，另一方面单纯追求覆盖率而忽视算法效率则不能满足评测时效性的要求。
- **检索召回方案** ：比赛过程中对比实验了两种召回方案，基于“文本语义向量表征“和“基于空间向量模型 + Bag-of-Ngram”。由于本任务文本普遍较长且专有名词较多等数据特点，实验表明“基于空间向量模型 + Bag-of-Ngram”的召回方案效果更好，下表中列出了使用的相关模型及其实验结果（recall@200）。可以看到相比于传统的BM25和TFIDF等算法，F1EXP、F2EXP等公理检索模型（Axiomatic Retrieval Models）可以取得更高的召回覆盖率，该类模型增加了一些公理约束条件，例如基本术语频率约束， 术语区分约束和文档长度归一化约束等等。

为了提升召回算法的效果，我们使用倒排索引技术对数据进行建模，然后在此基础上实现了F1EXP、DFR、F2EXP、BM25、TFIDF等多种检索算法，极大了提升了召回部分的运行效率。为了平衡召回率和计算成本，最后使用F1EXP、BM25、TFIDF 3种算法各召回50条结果融合作为后续精排候选数据，在验证集上测试，召回覆盖率可以到70%。

![img](https://cdn.zhuanzhi.ai/images/wx/801cb0bf1e9bbdac516a95272fbebaf6)



## 精准排序（Ranking）

> 参考 [jiangnanboy / learning_to_rank](https://github.com/jiangnanboy/learning_to_rank)
>
> [利用lightgbm做(learning to rank)排序学习](https://my.oschina.net/u/4585416/blog/4398683/print)

从整体框架的角度看，当用户每次请求时，系统就会将当前请求的数据写入到日志当中，利用各种数据处理工具对原始日志进行清洗，格式化，并存储到文件中。在训练时，我们利用特征工程，从处理过后的数据集中选出训练、测试样本集，并借此进行线下模型的训练和预估。我们采用多种机器学习算法，并通过线下AUC、NDCG、Precision等指标来评估他们的表现。线下模型经过训练和评估后，如果在测试集有比较明显的提高，会将其上线进行线上AB测试。同时，我们也有多种维度的报表对模型进行数据上的支持。

Ranking的主要职责如下：

- 获取所有列表实体进行预测所需特征。
- 将特征交给预测模块进行预测。
- 对所有列表实体按照预测值进行排序。



### 数据采集

- 人工标注 ground truth

从搜索日志中随机选取一部分Query，让受过专业训练的数据评估员对”Query-Url对”给出相关性判断。常见的是5档的评分:差、一般、好、优秀、完美。以此作为训练数据。人工标注是标注者的主观判断，会受标注者背景知识等因素的影响。

- 日志挖掘：

当搜索引擎搭建起来之后用户的点击数据就变得非常好使，用户点击记录是可以当做机器学习方法训练数据的一个替代品，比如用户发出一个查询，搜索引擎返回搜索结果，用户会点击其中某些网页，可以假设用户点击的网页是和用户查询更加相关的页面。

点击数据隐式反映了同Query下搜索结果之间相关性的相对好坏。在搜索结果中，高位置的结果被点击的概率会大于低位置的结果，这叫做”点击偏见”（Click Bias）。但采取以上的方式，就绕过了这个问题。因为我们只记录发生了”点击倒置”的高低位结果，使用这样的”偏好对”作为训练数据。

但是日志抽取也存在问题：

1. 用户总是习惯于从上到下浏览搜索结果
2. 用户点击有比较大的噪声
3. 一般头查询（head query）才存在用户点击

| 2级标注         | 5级标注      |
| --------------- | ------------ |
| 1: relevant     | 4: perfect   |
| 0: not relevant | 3: excellent |
|                 | 2: good      |
|                 | 1: fair      |
|                 | 0: bad       |



我们使用搜索积累的大量用户行为数据（如点击、点赞等）， 这些行为数据可以作为弱监督训练数据。使用到的用户行为日志表如下：

- 输入日志（input表）详细记录了用户每次的输入，用于和点击／转化数据做联合分析；
- 推荐日志（output表）详细打印了每次请求的参数信息和返回信息，如userid、输入的问题、请求时间、第几页、返回的搜索结果及对应的顺序位置；
- 点击日志（click表）详细记录了每次点击行为（点击item列表及其对应的参数信息）；

我们把query与搜索结果的**相关性标签分为三档，即弱相关，相关与不相关**。在query的选择上，通过对近六个月的query进行统计挖掘



### [数据去噪](https://blog.csdn.net/v_july_v/article/details/81319999)

- 数据清洗

  - 考虑哪些信息对于模型结果是有效的，去掉脏数据
  - 补齐缺省值（中位数补齐、平均数补齐）

- **样本去噪**

  - 无意义单字Query过滤。由于单字Query表达的语义通常不完整，用户点击行为也比较随机，这部分数据如果用于训练会影响最终效果。我们去除了包含无意义单字Query的全部样本。
  - 正样本（点击label=1，点赞label=2），若quey=doc情况下的doc未被点击，强制令其label=1；
  - 负样本中存在大量噪声数据，需要针对语料补充对应的过滤规则来过滤数据；

- 数据采样

  正负样本是不均衡的情况

  - 上采样
  - 下采样
  - 收集更多数据
  - 数据增强EDA
  - 修改损失函数
    - 使损失函数不要有太强的倾向性



### [特征处理★](https://blog.csdn.net/v_july_v/article/details/81319999)

在排序学习模型中，文档都是转化成特征向量来表征的，这便涉及一系列文本特征提取的工作。下面介绍特征类型以及对应的处理方法：

- 数值型
  - 幅度调整：提高梯度下降法的收敛速度
    - 归一化：`sklearn.preprocessing.MinMaxScaler` 把数据幅度调整到[0,1]范围内
    - 标准化：`sklearn.preprocessing.scale` 处理后的数据符合标准正态分布，即均值为0，标准差为1
    - **树形模型不需要特征归一化也不必担心缺失值，因为它们不关心变量的值，而是关心变量的分布和变量之间的条件概率，如决策树、随机森林！**
  - log数据域变化
  - 统计值 max, min, mean, std
- 类别型
  - one-hot编码
- 时间型
  - 连续值
    - 持续时间（单页浏览时长）
    - 间隔时间（上次点击距离现在的时间）
  - 离散值
    - 一天中哪个时间段(hour_0-23)
    - 一周中星期几(week_monday...)
- 文本型
  - 词袋模型（映射成稀疏向量）
  - Jaccard、Levenshtein、Simhash、TF-idf、Bm25
- 统计型
  - 比例类
  - 次序性
- 组合特征
  - .....

有这么一句话在业界广泛流传：数据和特征决定了机器学习的上限，而模型和算法只是逼近这个上限而已。我们收集了如下几类特征：

![](https://ningshixian.github.io/resources/images/LTR-features.png)

在具体实践中，特征分别从查询、主问题、关键词等多个维度进行抽取，最终构建成特征集合。我们的特征体系主要包括如下：

- 文档组成

  - query
  - primary question
  - similar question
  - answer

- 特征分类

  - 交叉特征

    主要是搜索的相关性特征，即计算用户Query和候选Doc两者之间相关度！[参考]([https://ningshixian.github.io/2020/10/26/%E6%96%87%E6%9C%AC%E5%8C%B9%E9%85%8D%E4%BB%A5%E5%8F%8AMatchZoo/](https://ningshixian.github.io/2020/10/26/文本匹配以及MatchZoo/))

    - BM25相似度
    - TF-IDF相似度
    - ES返回的Score
    - String Cosine相似度
    - Embedding相似度（基于Glove、Doc2vec、BERT得到Query和Doc的表示向量，然后计算余弦相似度.）
    - 编辑距离、词共现
    - query关键词出现次数及比例
    - jaccard相似度
    - Query-Doc点击次数和点击率

  - 单边特征

    - [query词权重](https://mp.weixin.qq.com/s?__biz=MzU1NTMyOTI4Mw==&mid=2247508936&idx=2&sn=78be7e9852a78ddf0da07198b6c7978b&chksm=fbd711a4cca098b245bcdba6932d42927a95ebf01fb949f3e4dbf8d74efc63e6c619357fc702&mpshare=1&scene=1&srcid=1013bKeTJiZPMKWHeGyO8il8&sharer_sharetime=1602575301868&sharer_shareid=1539cddeb9761bf9ab11f818eaf32ccd&key=0e194573360c8aa2a4aac7e900023ab49de9e41da0eed6dfefe4c448b54365646757fa4353f3c564de211b8858ac1b7a4aeebdad0b447f8cec6eacb8bb0bb911a6b67814994175cb4fda6c6602f8c6cd7a81c45a1632431fba29863fe20ac7c9af9fd488973d9c3feed2ba51cf484002e291a847faaf3be2e21a76dc31b8e830&ascene=1&uin=MjM2MDA1NjcyMQ%3D%3D&devicetype=Windows+10+x64&version=6300002f&lang=zh_CN&exportkey=A8ykpIlyUpzN8fKs63fFFqM%3D&pass_ticket=Ci2XyVC7HEwQ1U4ddZ%2BTa0dDdVzykUszBehfKZob3hleSuYqaB%2Bj56ke%2FGHlx9Pa&wx_header=0)
    - query长度
    - Doc长度
    - Doc时间戳（问题的新鲜度）
    - Doc更新时间
    - Doc的具体信息，包括xxx
    - ID类特征（docID、itemID、answerID、baseCode、botCode、projectCode）
    - ~~点击率~~
    - ~~来源source、点赞数、点击反馈、高亮字数~~

  - 全局特征

    - 召回阶段doc的全局位置特征SearchOrder
    - 召回阶段doc所在的当前页面
    - 请求时间段



PS：根据计算方法相关性主要分为字面相关性和语义相关性。

**字面相关性**

早期的相关性匹配主要是根据Term的字面匹配度来计算相关性，如字面命中、覆盖程度、TFIDF、BM25等。字面匹配的相关性特征在搜索排序模型中起着重要作用，但字面匹配有它的局限，主要表现在：

- **词义局限**：字面匹配无法处理同义词和多义词问题，如“宾馆”和“旅店”虽然字面上不匹配，但都是搜索“住宿服务”的同义词；而“COCO”是多义词，在不同业务场景下表示的语义不同，可能是奶茶店，也可能是理发店。
- **结构局限**：“蛋糕奶油”和“奶油蛋糕”虽词汇完全重合，但表达的语义完全不同。 当用户搜“蛋糕奶油”时，其意图往往是找“奶油”，而搜“奶油蛋糕”的需求基本上都是“蛋糕”。

**语义相关性**

**BERT语义相关性**

基于BERT的语义匹配有两种应用方式：

- **Feature-based**：属于基于表示的语义匹配方法。类似于DSSM双塔结构，通过BERT将Query和Doc编码为向量，Doc向量离线计算完成进入索引，Query向量线上实时计算，通过近似最近邻（ANN）等方法实现相关Doc召回。
- **Finetune-based**：属于基于交互的语义匹配方法，将Query和Doc对输入BERT进行句间关系Fine-tuning，最后通过MLP网络得到相关性分数。



### [特征选择](https://blog.csdn.net/v_july_v/article/details/81319999)

各种特征处理之后会产生很多特征，他们之间会存在如下问题：

- 冗余：部分特征的相关度太高了，消耗计算性能
- 噪声：部分特征是对预测结果有**负影响**

针对这两个问题，咱们便得做下特征选择，包括降维

- 特征选择是指踢掉原本特征里和结果预测关系不大的
- SVD或者PCA确实也能解决一定的高维度问题

从众多特征中挑选出少许有用特征。与学习目标不相关的特征和冗余特征需要被剔除，如果计算资源不足或者对模型的复杂性有限制的话，还需要选择丢弃一些不重要的特征。特征选择方法常用的有以下几种：

![dw topic](https://awps-assets.meituan.net/mit-x/blog-images-bundle-2017/94ec9737.png)

上图来自：https://tech.meituan.com/2017/07/28/dl.html

**1 过滤型**

评估单个特征和结果值之间的相关程度，排序留下Top相关的特征部分。而计算相关程度可以用Pearson相关系数、互信息、距离相关度来计算。

这种方法的缺点是：没有考虑到特征之间的关联作用，可能把有用的关联特征误踢掉。

```python
from sklearn.datasets import load_iris
from sklearn.feature_selection import SelectKBest
from sklearn.feature_selection import chi2
iris = load_iris()
X,y = iris.data, iris.target
X. shape
# (150, 4)
X_new = SelectKBest(chi2, k=2).fit_transform(X,y)
X_new.shape
# (150, 2)
```

**2 包裹型**

包裹型是指把特征选择看做一个特征子集搜索问题，筛选各种特征子集，用模型评估效果。典型的包裹型算法为 “递归特征删除算法”(recursive feature elimination algorithm)。

比如用逻辑回归，怎么做这个事情呢？

1. 用全量特征跑一个模型
2. 根据线性模型的系数(体现相关性)，删掉5-10%的弱特征，观察准确率/auc的变化
3. 逐步进行，直至准确率/auc出现大的下滑停止

**3 嵌入型**

嵌入型特征选择是指根据模型来分析特征的重要性，比如决策树的feature_importances_属性，返回的重要性是按照决策树种被用来分割后带来的增益(gain)总和进行返回。

```
gbm.feature_importance()
```



### 训练样本构建

首先，将数据格式处理成 libsvm格式：

- 每一行的行首是该样本的因变量y，是我们要预测的目标值，比如{1, -1}；

- y后面跟着的是各个自变量，即各维度的特征。但是注意这些自变量不是直接排在因变量后面，而是以 “索引: 取值” 的方式依次排在后面——索引就是第几个自变量；依次类推。如果样本中某个自变量的取值为0，那么该自变量的可以不出现，例如，如果第20个自变量取值为0，那么可以采用“19:XXXX 21:XXXX”形式，跳过“20:0”，这样做可以减少内存的使用，并提高做矩阵内积时的运算速度。



### 模型训练与评估

> 1. LambdaRank，基于xgboost实现，参考：https://github.com/dmlc/xgboost/tree/master/demo/rank
> 2. LambdaMART，基于LightGBM实现，参考：https://github.com/jiangnanboy/learning_to_rank
> 3. Wide&Deep基于TensorFlow实现，参考：https://github.com/tensorflow/ranking

> **Lightgbm代码示例**
>
> [Learning to Rank Explained (with Code)](https://mlexplained.com/2019/05/27/learning-to-rank-explained-with-code/)
>
> **Xgboost代码示例**
>
> [Xgboost排序学习的接口](https://github.com/dmlc/xgboost/tree/master/demo/rank)
>
> [XGBoost在携程搜索排序中的应用](https://zhuanlan.zhihu.com/p/97126793)
>
> [xgboost实现learning to rank算法以及调参](https://www.jianshu.com/p/392b6ef4656b)
>
> **tensorflow_ranking代码示例**
>
> [Implement Listwise LTR using `tensorflow`](https://everdark.github.io/k9/notebooks/ml/learning_to_rank/learning_to_rank.html)



![](https://ningshixian.github.io/resources/images/SOTA-LTR.png)



我们选择 IR 领域是最先进的**LambdaMART**模型，基于 lightGBM 的 **GBDT** 树模型进行训练，方便dump和load模型，采用 **lambdarank** 损失来进行训练，以 **NDCG** 指标作为优化方向。

- GBDT(Gradient Boosting Decision Tree) 又叫 MART（Multiple Additive Regression Tree)，一种迭代的决策树算法
- LambdaRank pairwise loss 引入RankNet的Lambda梯度，直接用 $|ΔNDCG|$ 乘以 $λ_{ij}$ 得到 LambdaRank 的 Lambda
- $C_{i j}=\log \left(1+e^{-\sigma\left(s_{i}-s_{j}\right)}\right) \cdot\left|\Delta Z_{i j}\right|$



**GBDT和LR 的融合模型**

[github地址](https://github.com/princewen/tensorflow_practice/tree/master/recommendation/GBDT%2BLR-Demo)

针对GBDT和LR模型的优缺点，做了进一步的模型组合的尝试：

- 第一种方式，用 LR 模型把高维稠密特征进行学习，学习出高维特征，把该特征和原始特征做拼接，学习 gbdt 模型。
  - 该办法效果不好，提升很弱。
  - **剖析缘由：** 把高维特征刚在一个特征去表达，丢掉了原始的特征。
- 第二种方式，用 gbdt 去学，学习后把样本落入叶子节点信息来进来与高维稠密特征拼接，在此根底上用 LR 学习。
  - 该模型效果变差。
  - **剖析缘由：** 点击类和穿插类特征是对排序影响最大的特征，这类特征和大量的稠密类特征做拼接时，重要性被稀释了，导致模型的学习能力变弱。

![img](https://upload-images.jianshu.io/upload_images/4155986-8a4cb50aefba2877.png?imageMogr2/auto-orient/strip|imageView2/2/w/508/format/webp)

**深度神经网络模型**

如Google提出的Wide&Deep模型搭建的网络结构[2]。其中Wide部分输入的是LR、GBDT阶段常用的一些细粒度统计特征。通过较长周期统计的高频行为特征，能够提供很好的记忆能力。Deep部分通过深层的神经网络学习Low-Order、高纬度稀疏的Categorical型特征，拟合样本中的长尾部分，发现新的特征组合，提高模型的泛化能力。同时对于文本、头图等传统机器学习模型难以刻画的特征，我们可以通过End-to-End的方式，利用相应的子网络模型进行预处理表示，然后进行融合学习。

![图3 Deep&Wide模型结构图](https://p0.meituan.net/travelcube/5b5f704e77a7bb903c5b5f1c8050127d53152.png)



**如何构造pair对？**
xgboost/src/objective/rank_obj.cc,75行开始构造pair对。如上理论所说，每条文档移动的方向和趋势取决于其他所有与之 label 不同的文档。因此我们只需要构造不同label的“正向文档对”。其方法主要为:遍历所有的样本，从与本样本label不同的其他label桶中，任意取一个样本，构造成正样本；



### 模型预测

```
preds_test = gbm.predict(X, num_iteration=best_iteration, num_threads=1)
# 根据group切分预测概率并排序
rank = ut.Sort.resort(preds_test)
```



### 模型分发

模型分发的目标是把在线流量分配给不同的实验模型，具体而言要实现三个功能：

- 为模型迭代提供在线流量，负责线上效果收集、验证等。
- A/B测试，确保不同模型之间流量的稳定、独立和互斥、确保效果归属唯一。
- 确保与其他层的实验流量的正交性。

采用如下步骤将流量分配到具体模型上面去：

- 把所有流量分成N个桶。
- 每个具体的流量Hash到某个桶里面去。
- 给每个模型一定的配额，也就是每个策略模型占据对应比例的流量桶。
- 所有策略模型流量配额总和为100%。
- 当流量和模型落到同一个桶的时候，该模型拥有该流量。

![](https://ningshixian.github.io/resources/images/哈希在线分流.png)



### [补充：完整的特征获取流程](https://tech.meituan.com/2018/12/20/head-in-l2r.html)

- Ranking模块从FeatureModel里面读取所有的原始特征名。
- Ranking将所有的原始特征名交给Feature Proxy。
- Feature Proxy根据特征名的标识去调用对应的Feature Service，并将原始特征值返回给Ranking模块。
- Ranking模块通过Expression将原始特征转化成复合特征。
- Ranking模块将所有特征交给级联模型做进一步的转换。

**特征模型（Feature Model）**

我们把所有与特征获取和特征算子的相关信息都放在一个类里面，这个类就是FeatureModel，定义如下：

```java
    //包含特征获取和特征算子计算所需的meta信息 
    public class FeatureModel {   
        //这是真正用在Prediction里面的特征名 
        private String featureName;  
        //通过表达式将多种原子特征组合成复合特征。
        private IExpression expression; 
        //这些特征名是真正交给特征服务代理(Feature Proxy)去从服务端获取特征值的特征名集合。
        private Set<String> originalFeatureNames;
        //用于指示特征是否需要被级联模型转换  
        private boolean isTransformedFeature; 
        //是否为one-hot特征    
        private boolean isOneHotIdFeature;
        //不同one-hot特征之间往往共享相同的原始特征，这个变量>用于标识原始特征名。 
        private String oneHotIdKey; 
        //表明本特征是否需要归一化
        private boolean isNormalized; 
    }
```

**特征服务代理（Feature Proxy）**

特征服务代理负责远程特征获取实施

**表达式（Expression）**

表达式的目的是为了将多种原始特征转换成一个新特征，或者对单个原始特征进行运算符转换。我们采用前缀法表达式（Polish Notation）来表示特征算子运算。例如表达式(5-6)*7的前缀表达式为* - 5 6 7。



## 下一步迭代计划

> [知乎搜索排序模型的演进](https://mp.weixin.qq.com/s/DZZ_BCiNw0EZg7V0KvhXVw)

1. 由GBDT升级到DNN模型，这么做主要是考虑到两点：

- 数据量变大之后，DNN能够实现更复杂的模型，而GBDT的模型容量比较有限。

- 新的研究成果都是基于深度学习的研究，采用深度神经网络之后可以更好的应用新的研究成果，比如后面介绍的多目标排序。

  DNN排序模型整体是在谷歌开源的TF-Ranking[1] 框架的基础上开发，包括特征输入、特征转化、模型打分和损失计算等几个独立的模块。

  ```
  深度模型之Wide&Deep、DeepFM模型、DCN模型、DIN模型、DIEN模型
  *Pointwise*类型的语义相似模型，如文本匹配库matchzoo，构建MV-LSTM、DRMM和DUET这三类使用较多的文本匹配模型
  ```

2. Unbias LTR

   在使用用户点击日志作为训练数据时，点击日志中的一些噪声会对训练带来一定的负面影响。其中最常见的噪声就是Position bias，由于用户的浏览行为是从上到下的，所以同样的内容出现的位置越靠前越容易获得点击。解决方案：**对Position bias建模**。用户实际点击 = 真实点击率+ Position Bias

3. Context Aware LTR

   借鉴了CV 中的 SE Block[4] 结构

4. 尝试融合模型 Ensemble( BERT(pairwise) + LightGBM(pairwise) )

   ```
   WSDM Cup 2020检索排序评测任务第一名经验总结
   https://www.zhuanzhi.ai/document/24ad9f7f873da90dc072b0c777e1acfe
   
   开源的代码，也包括了集成的部分
   https://github.com/myeclipse/wsdm_cup_2020_solution
   ```




# 拓展：LTR模型详细介绍

下文主要对应用较广的一些排序模型ranker进行调研。[小菜鸡 - 排序学习调研](http://xtf615.com/2018/12/25/learning-to-rank/)

## 模型分类

![img](http://octopuscoder.github.io/images/search_kinds.png)

## 迭代过程

[![img](http://octopuscoder.github.io/images/search_process.png)](http://octopuscoder.github.io/images/search_process.png)

## 主流模型及评测指标

![img](https://octopuscoder.github.io/images/search_evalute.png)

## 主流模型对比

![img](https://octopuscoder.github.io/images/serach_model_compare.png)

已发表并广泛接受的重排或者 LTR 工作分为这么几类（[参考](https://mp.weixin.qq.com/s?__biz=MjM5MzY4NzE3MA==&mid=2247489906&idx=2&sn=ef3edb0c5ea102b92beaf3731f928af3&chksm=a6926a9f91e5e389230aa102870b24102d73e7073c75c5bfd997f8d8b8d6b9b14808326ff011&mpshare=1&scene=1&srcid=1218cGdyK7ksuwEousDuBPh9&sharer_sharetime=1608254048384&sharer_shareid=52006a0d19edf83d2b8be98f4d8fe935&key=de338918f7ab44e5b935e87b468af80f804e40fbbecb01e4bf7116fb04b8ce5fa962dcb12a6aec954a9a0e803f98a1930a92f5885832f1462a0333ef27523952eb52426a2d3c060b9816aabb22857c7a5b663bb789dcb0b49ae257cdd2e57211a93765f3f1c5da5cd1ae3995563e9dd706915eb4f3de5b4b422d607d5f9d3f79&ascene=1&uin=MjM2MDA1NjcyMQ%3D%3D&devicetype=Windows+10+x64&version=6300002f&lang=zh_CN&exportkey=AwWjy1O7bSe33R346Is4SBA%3D&pass_ticket=yE6RfYIA0zBp6B9kD2Bxn6MvchszZXW4mBLV47AnsHQVr0BUyLuexzU3nmPi1neS&wx_header=0)）：

**Point-wise 模型：**和经典的 CTR 模型基本结构类似，如 DNN [8]， WDL [9] 和 DeepFM [10]。和排序相比优势主要在于实时更新的模型、特征和调控权重。LTR 对于调控权重的能力还是基于单点的，没有考虑到商品之间互相的影响，因此并不能保证整个序列收益最优。

**Pair-wise 模型：**通过 pair-wise 损失函数来比较商品对之间的相对关系。具体来说，RankSVM [12], GBRank [13] 和 RankNet [2] 分别使用了 SVM、GBT 和 DNN。但是，pair-wise 模型忽略了列表的全局信息，而且极大地增加了模型训练和预估的复杂度。

**List-wise 模型：**建模商品序列的整体信息和对比信息，并通过 list-wise 损失函数来比较序列商品之间的关系。LambdaMart [14]、MIDNN [3]、DLCM[6]、PRM [5] 和 SetRank [4] 分别通过 GBT、DNN、RNN、Self-attention 和 Induced self-attention 来提取这些信息。但是由于最终还是通过贪婪策略进行排序，还是不能真正做到考虑到排列特异性的上下文感知。随着工程能力的升级，输入序列的信息和对比关系也可以在排序阶段中提取出来。



## Pointwise Ranker

首先介绍Pointwise排序模型，这种模型直接对标注好的pointwise数据，学习$P(y|\boldsymbol{x})$或直接回归预测$y$。从早期的线性模型LR，到引入自动二阶交叉特征的FM和FFM，到非线性树模型GBDT和GBDT+LR，到最近全面迁移至大规模深度学习排序模型。

### LR

**LR(logistics regression)**，是**CTR**预估模型的最基本的模型，也是工业界最喜爱使用的方案，**排序模型化**的首选。实际优化过程中，会给样本添加不同权重，例如根据反馈的类型点击、浏览、收藏、下单、支付等，依次提高权重，优化如下带权重的损失：
$$
J(\theta)=-\frac{1}{m}\sum_{i=1}^m w_i \cdot \Big( y_i log(\sigma_\theta(\boldsymbol{x}_i))+(1-y_i)log(1-\sigma_\theta(\boldsymbol{x}_i))\Big)
$$
其中，$w_i$是样本的权重，$y_i$是样本的标签。$\sigma_{\theta}(\boldsymbol{x}_i)=\frac{1}{1+e^{-(\boldsymbol{\theta}\cdot \boldsymbol{x}+ \boldsymbol{b})}}$。

**模型优点是可解释性强。通常而言，良好的解释性是工业界应用实践比较注重的一个指标，它意味着更好的可控性，同时也能指导工程师去分析问题优化模型。**但是LR假设各特征间是相互独立的，忽略了特征间的交互关系，因此需要做大量的特征工程。同时LR对非线性的拟合能力较差，限制了模型的性能上限。通常LR可以作为**Baseline**版本。

### GBDT

GBDT (梯度提升决策树) 又叫 MART（Multiple Additive Regression Tree)，是一种表达能力比较强的非线性模型。GBDT是基于Boosting 思想的ensemble模型，由多棵决策树组成，具有以下优点：

- 处理稠密连续值特征（特征的量纲、取值范围无关，故可以不用做归一化）
- 学习模型可以输出特征的相对重要程度，可以作为一种特征选择的方法
- 对数据字段缺失不敏感
- 根据熵增益自动进行特征转换、**特征组合**、特征选择和离散化，挖掘高维的组合特征，省去了人工转换的过程，并且支持了多个特征的Interaction

GBDT的主要缺陷是依赖连续型的统计特征，对于高维度稀疏特征、时间序列特征不能很好的处理。如果我们需要使用GBDT的话，则需要将很多特征统计成连续值特征(或者embedding)，这里可能需要耗费比较多的时间。鉴于此Facebook提出了一种**GBDT+LR**的方案。

**GBDT+LR**

GBDT+LR方法是Facebook2014年提出在论文[Practical Lessons from Predicting Clicks on Ads at Facebook](http://quinonero.net/Publications/predicting-clicks-facebook.pdf)提出的点击率预估方案。其动机主要是因为LR模型中的特征组合很关键， 但又无法直接通过特征笛卡尔积解决，只能依靠人工经验，耗时耗力同时并不一定会带来效果提升。如何自动发现有效的特征、特征组合，弥补人工经验不足，缩短LR特征实验周期，是亟需解决的问题。

GBDT+LR的核心思想是利用GBDT训练好的多棵树，某个样本从根节点开始遍历，可以落入到不同树的不同叶子节点中，将叶子节点的编号作为GBDT提取到的高阶特征，该特征将作为LR模型的输入。原论文中的模型结构如下：

[![lr_gbdt](http://xtf615.com/picture/machine-learning/lr_gbdt.png)](http://xtf615.com/picture/machine-learning/lr_gbdt.png)

图中$\boldsymbol{x}$是一个样本，$\boldsymbol{x}$两个子节点对应两棵子树Tree1、Tree2为通过GBDT模型学出来的两颗树。遍历两棵树后，$\boldsymbol{x}$样本分别落到两棵树的不同叶子节点上，每个叶子节点对应提取到的一个特征，那么通过遍历树，就得到了该样本对应的所有特征。由于树的每条路径，是通过最小化均方差等方法最终分割出来的有区分性路径，根据该路径得到的特征、特征组合都相对有区分性，效果理论上不会亚于人工经验的处理方式。 最后，将提取到的叶子节点特征作为LR的输入。

实验结果表明：`GBDT-LR` 比单独的 `GBDT` 模型，或者单独的 `LR` 模型都要好。另外，GBDT树模型可以使用目前主流的XGBoost或者LightGBM替换，这两个模型性能更好，速度更快。

**GBDT+FM**

GBDT+LR排序模型中输入特征维度为几百维，都是稠密的通用特征。这种特征的泛化能力良好，但是记忆能力比较差，所以需要增加高维的（百万维以上）内容特征来增强推荐的记忆能力，包括物品ID，标签，主题等特征。GBDT不支持高维稀疏特征，如果将高维特征加到LR中，一方面需要人工组合高维特征，另一方面模型维度和计算复杂度会是$O(n^2)$级别的增长。所以设计了GBDT+FM的模型如图所示，采用Factorization Machines模型替换LR。

[![fm_gbdt](http://xtf615.com/picture/machine-learning/fm_gbdt.png)](http://xtf615.com/picture/machine-learning/fm_gbdt.png)

该模型既支持稠密特征，又支持稀疏特征。其中，稠密特征通过GBDT转换为高阶特征（叶子节点编号），并和原始低阶稀疏特征作为FM模型的输入进行训练和预测，并支持特征组合。且根据上述，FM模型的复杂度降低到$O(n)$(忽略隐向量维度常数)。

### DNN（下一步计划）

GBDT+FM模型，对embedding等具有结构信息的深度特征利用不充分，而深度学习（Deep Neural Network）能够同时对原始稀疏特征(通过嵌入式embedding方式)和稠密特征进行学习，抽取出深层信息，提高模型的准确性，并已经成功应用到众多机器学习领域。因此可以将DNN引入到排序模型中，提高排序整体质量。

YouTube 2016发表了一篇基于神经网络的推荐算法[Deep Neural Networks for YouTube Recommendations](https://static.googleusercontent.com/media/research.google.com/zh-CN//pubs/archive/45530.pdf)。文中使用神经网络分别构建召回模型和排序模型。简要介绍下召回模型。召回模型使用基于DNN的超大规模多分类思想，即在时刻$t$，为用户$U$（上下文信息$C$）在视频库$V$中精准的预测出用户想要观看的视频为$i$的概率，用数学公式表达如下：
$$
p(w_t=i|U,C)=\frac{e^{v_i^T u}}{\sum_{j \in V} e^{v_j^Tu}}
$$
向量$u$是信息的高维embedding，而向量$v$则是视频的embedding向量，通过u与v的点积大小来判断用户与对应视频的匹配程度。所以DNN的目标就是在用户信息和上下文信息为输入条件下学习视频的embedding向量$v$以及用户的embdedding向量$u$。召回模型架构如下：

[![youtube](http://xtf615.com/picture/machine-learning/youtube_recall.png)](http://xtf615.com/picture/machine-learning/youtube_recall.png)

分为离线计算和在线服务两个部分。离线计算的时候，通过embedding提取用户的不同特征(历史观影序列、搜索词)，同类型特征先average，然后再将不同类型特征concat起来，作为MLP的输入，最后一层隐藏层的输出作为用户的embedding向量，再和视频的embedding向量点乘，并用于预测该视频概率。（输出层神经元由所有视频构成，最后一层隐藏层和该视频对应的输出层某个神经元相连接的权重向量就是该视频的embedding）。在线服务的时候，利用学习到的用户embedding和视频embedding，并借助近似近邻搜索，求出TopN候选结果(这里的问题是最后使用的video vectors如何构建的问题，是直接使用最后全连接层的权重还是使用最底下embedded video watches中的各视频的embedding。这个问题已有大神讨论过，见[关于’Deep Neural Networks for YouTube Recommendations’的一些思考和实现](https://www.jianshu.com/p/f9d2abc486c9))

排序模型和上述几乎一样，只不过最后一层输出不是softmax多分类问题，而是logistic regression二分类问题，并且输入层使用了更多的稀疏特征和稠密特征。之所以在候选阶段使用softmax，是因为候选阶段要保证多样性等，因此面向的是全量片库，使用softmax更合理。而在排序阶段，需要利用实际展示数据(impression data)、用户反馈行为数据来对候选阶段的结果进行校准，由于召回阶段已经过滤了大量的无关物品，因此排序阶段可以进一步提取描述视频的特征、用户和物品关系的特征等。

如下图：

[![youtube_rank](http://xtf615.com/picture/machine-learning/youtube_rank.png)](http://xtf615.com/picture/machine-learning/youtube_rank.png)

同样分为训练部分和服务部分。训练时，提取了更多关于视频、用户、上下文的特征，其中，稀疏特征通过embedding方式传入，稠密特征通过归一化、幂率变换等传入，最后使用加权的logistic regression来训练，权重使用观看时长表示，最大化$P(y|\boldsymbol{x})$的概率。

训练好后，使用$e^{Wx+b}$来计算得分并排序即可(不需要进行sigmoid操作，排序时二者等价)。

#### Wide & Deep Learning

该模型是Google 2016论文 [Wide & Deep Learning for Recommender Systems](https://arxiv.org/pdf/1606.07792.pdf) 提出的，用于Google Play应用中app的推荐。该模型将线性模型组件和深度神经网络进行融合，即：广义线性模型表达能力不强，容易欠拟合；深度神经网络模型表达能力太强，容易过拟合。二者结合就能取得平衡。

- 宽度学习组件 `linear model:LM` 的输入主要是稀疏特征以及稀疏特征间转换后的交叉特征，能够实现记忆性，记忆性体现在从历史数据中直接提取特征的能力；
- 深度学习组件 `neural network:NN` 的输入包括了稠密特征以及稀疏特征，其中稀疏特征通过embedding方式转换成稠密特征，能够实现泛化性，泛化性体现在产生更一般、抽象的特征表示的能力。

我们希望在宽深度模型中的宽线性部分可以利用交叉特征去有效地记忆稀疏特征之间的相互作用，而在深层神经网络部分通过挖掘特征之间的相互作用，提升模型之间的泛化能力。下图是宽深度学习模型框架：

![Wide-Deep-Learning](https://static.sitestack.cn/projects/huaxiaozhuan-ai/80b0a720a77448ab9457ddd77edff59e.png)

wide learning 形式化为：$y=W_{\text{wide}}^T \{\boldsymbol{x}, \phi(\boldsymbol{x})\} + b$，其中$\boldsymbol{x}$是原始稀疏特征，$\phi(\boldsymbol{x})$是手动精心设计的、变换后的稀疏特征， $\{\boldsymbol{x}, \phi(\boldsymbol{x})\}$表示特征拼接。

deep learning形式化为：$a^{l+1} = W^T_{deep} a^{l} + b^{l}$，deep组件的输入包括连续值特征以及稀疏特征。其中稀疏特征通过embedding方式转换成稠密特征。sigmoid激活后：
$$
P(\hat{r}_{ui}=1|x) = \sigma(W^T_{\text{wide}}\{\boldsymbol{x},\phi(\boldsymbol{x})\} + W^T_{\text{deep}}a^{(l_f)}+ bias)
$$
其中，$\hat{r}_{ui}=1$表示正反馈行为，$P(\hat{r}_{ui}=1|x)$是正反馈行为的概率。



在Google Play App推荐应用上，针对该场景设计的神经网络架构如下：

[![wide_deep_learning](http://xtf615.com/picture/machine-learning/wide_deep_learning.png)](http://xtf615.com/picture/machine-learning/wide_deep_learning.png)

从上图可以看出，宽度组件主要从稀疏特征(类别特征)中进行特征组合操作；深度组件从原始稠密特征(连续值特征)，以及将稀疏特征通过embedding方式，输入到MLP提取高阶特征。宽度和深度组件输出结果相加，并通过sigmoid函数激活，最后一起联合优化Logistic Loss进行学习。

#### Deep FM

[DeepFM](https://arxiv.org/pdf/1703.04247.pdf)（Deep Factorization Machine）是哈工大Guo博士在华为诺亚实验室实习期间，提出的一种深度学习方法，它基于Google的经典论文**Wide&Deep**基础上，通过**将原论文的wide部分(LR部分)替换成FM**，从而改进了原模型依然需要人工特征工程的缺点，得到一个end-to-end 的深度学习模型。

也就是将**W&D**中的**线性部分**![[公式]](https://www.zhihu.com/equation?tex=linear%28X%29)替换成2.4中的 ![[公式]](https://www.zhihu.com/equation?tex=FM%28X%29) 就是所谓的DeepFM模型。模型的架构如下:

![img](https://pic1.zhimg.com/80/v2-313c8a340c4df2aef8604b466c17259c_1440w.jpg)

数学公式如下:

![[公式]](https://www.zhihu.com/equation?tex=f%28x%29+%3D+logistics%28linear%28X%29%2B%5Csum_%7Bi%3D1%7D%5E%7Bn%7D%7B%7D%5Csum_%7Bj%3Di%2B1%7D%5E%7Bn%7D%7B%3Cv_%7Bi%7D%2Cv_%7Bj%7D%3Ex_%7Bi%7Dx_%7Bj%7D%7D%2BDNN%28X%29%29%3Dlogistics%28FM%28X%29%2BDNN%28X%29%29)

将**W&D线性部分**![[公式]](https://www.zhihu.com/equation?tex=linear%28X%29)替换成 ![[公式]](https://www.zhihu.com/equation?tex=FM%28X%29) 后增强了低阶特征的表达能力(**二阶交叉**)，克服了**W&D**的**线性部分**依然需要对**低维特征做特征工程**的缺点。

#### DCN：Deep&Cross

上一小节讲到为了加强**低阶特征的表达能力**，将**W&D**中的**LR线性部分**替换成**FM**。FM的本质是**增加了低纬特征间的二阶交叉能力，**为了挖掘**更高阶的特征交叉的价值**提出了[DCN](https://arxiv.org/pdf/1708.05123.pdf) (Deep &Cross Network)模型。

**DCN的网络架构如下:**

[![dcn](http://xtf615.com/picture/machine-learning/dcn.png)](http://xtf615.com/picture/machine-learning/dcn.png)

比较**DCN**与**DeepFM**的网络架构，可以发现本质区别是将**DeepFM**的**FM结构**替换为Cross Network结构。在输入侧，稀疏特征通过embedding层转化为稠密特征后，和连续特征concat起来，一起作为模型的输入。因此，输入层面只有embedding column+continuous column, 特征组合在网络结构中自动实现。上图右侧是常见的MLP深度网络，左侧是本文提出的交叉网络。下面看下什么是Cross Network? 以及Cross Network是如何实现**高阶特征组合**的?

**Cross Network结构:**

[![cross-network](http://xtf615.com/picture/machine-learning/cross_network.png)](http://xtf615.com/picture/machine-learning/cross_network.png)

**Cross Network整体看**是一个递推结构:
$$
x_{l+1} = x_0 x_l^Tw_l + b_l + x_l = f(x_l,w_l,b_l) + x_l
$$
首先, $x_0 \in \mathbb{R}^d$ 为输入的特征(第一层输入)， $x_l \in \mathbb{R}^d$ 为第 L 层的输入， $w_l$($w_l \in \mathbb{R}^{d}$)和$b_l$($b_l\in \mathbb{R}^{d}$)为第$l$层的参数矩阵。结合图可以看出Cross Network通过矩阵乘法实现特征的组合。**每一层的特征都由其上一层的特征进行交叉组合，并把上一层的原始特征重新加回来。**这样既能做特征组合，自动生成交叉组合特征，又能保留低阶原始特征，随着cross层的增加，是可以生成任意高阶的交叉组合特征（而DeepFM模型只有2阶的交叉组合特征）的，且在此过程中没有引入更多的参数（仅仅为$L_c \times d \times 2$），有效控制了模型复杂度。

同时为了缓解网络性能"**退化**"的问题，还引入了**残差**的思想:

![img](https://pic3.zhimg.com/80/v2-e252f9a168d66100125e20b0ead13ca2_1440w.png)残差思想

总结一下，DCN提出了一种新的交叉网络，在每个层上明确地应用特征交叉，能够**有效地捕获有限度的有效特征的相互作用**，学会高度非线性的相互作用，不需要人工特征工程或遍历搜索，并具有较低的计算成本。此外**Cross Network**可以看成是对**FM的泛化，**而FM则是**Cross Network**在**阶数为2时的特例。**

#### **DIN**

阿里提出的DIN模型中使用了**weighted-sum机制，**其实就是加权的sum-pooling，权重经过一个**activation unit**计算得到，即**Attention(注意力)**机制。通过它来对用户行为的两个重要特性（Diversity & Local activation）进行建模。

#### 深度模型之语义相似模型

**1). CNN(Learning to Rank Short Text Pairs with Convolutional Deep Neural Networks)**

**Paper:**[http://eecs.csuohio.edu/~sschung/CIS660/RankShortTextCNNACM2015.pdf](https://link.zhihu.com/?target=http%3A//eecs.csuohio.edu/~sschung/CIS660/RankShortTextCNNACM2015.pdf)

**Code:**[https://github.com/YETI-WU/Semantic_Analysis_NLP/blob/master/rank_TextPairs.py](https://link.zhihu.com/?target=https%3A//github.com/YETI-WU/Semantic_Analysis_NLP/blob/master/rank_TextPairs.py)

![img](https://pic3.zhimg.com/80/v2-d6fbd79428925f433757e557a5a82a4e_1440w.jpg)

Semantic analysis. GloVe word embedding + Conv1D + MaxPool, concatenated with similarity matching, forward to Softmax layer and following the logistic regression output, binary cross entropy loss function to identify the matching similarity of documentary and query in sklearn datasets fetch_20newsgroup training.



**2). DRMM(Deep Relevance Ranking Using Enhanced Document-Query Interactions)**

**Paper:**[http://nlp.cs.aueb.gr/pubs/emnlp2018.pdf](https://link.zhihu.com/?target=http%3A//nlp.cs.aueb.gr/pubs/emnlp2018.pdf)



**3). DSSM**

// TODO



## Pairwise Ranker

点级排序学习技术完全从单个物品的分类得分角度计算，没有考虑物品之间的顺序关系。而对级排序学习技术则偏重于对物品顺序关系是否合理进行判断，判断任意两个物品组成的物品对$<\text{item 1} , \text{item 2}>$，是否满足顺序关系，即，item 1是否应该排在 item 2 前面(相比于item 2, 用户更偏好于item 1)。具体构造偏序对的方法见Basic一节。支持偏序对的模型也有很多，例如支持向量机、BPR、神经网络等。

然而，无论使用哪一种机器学习方法，最终目标都是判断对于目标用户，输入的物品对之间的顺序关系是否成立，从而最终完成物品的排序任务来得到推荐列表。

### RankSVM

参考：https://www.cnblogs.com/bentuwuying/p/6683832.html

Ｒanking SVM 算法是通过在训练集构造样本有序数据对的方式将排序问题转化为应用支持向量机方法解决的分类问题，如下图所示。

![Fig. 2](https://img-blog.csdn.net/20150516112500826)

在SVM中，为实现软间隔最大化， 需要满足约束$y_i(w^Tx_i +b) > 1- \xi_i$，$y_i$代表样本$x_i$是正样本(1)还是负样本(-1)。而RankSVM是典型的pairwise方法，考虑两个有偏序关系的物品对，训练样本为$x_{u,i}-x_{u,j}$，则优化目标为：
$$
\begin{split}
& min_{w,b} \frac{1}{2}||w||^2 + C\sum_{i=1}^n \xi_{uij} \\
s.t., & y_{uij}(w^T(x_{u,i}-x_{u,j})) \geq 1-\xi_{uij}, \xi_{uij}\geq 0
\end{split}
$$
此时，$y_{uij}$代表用户$u$对$i$的喜爱程度比对$j$的喜爱程度高时，为1，否则为-1。上述优化等价于优化如下 `Hinge Loss`:
$$
L_{w}=\sum_{(u,i,j)\in \mathcal{S} \cup \mathcal{D}}[1- y_{uij}\cdot w^T x_{u,i}+ y_{uij} \cdot w^Tx_{u,j}]_{+}
$$
其中，$S$集合中，$y_{uij}=1$，$D$集合中，$y_{uij}=-1$。$[z]_{+}=max(0,z)$。

上述的改进工作一般是围绕不同pair有序对权重不同展开，即$\xi_{uij}$前会加个权重。`Hinge Ranking Loss` 是排序学习中非常重要的损失函数之一，大部分做ranking任务的模型都会采用。

### *[RankNet](https://cloud.tencent.com/developer/article/1061669)

RankNet 是 2005 年微软提出的一种经典的**Pairwise**的排序学习方法，是典型的**前向神经网络排序模型**。它从概率的角度来解决排序问题。RankNet 的核心是**提出了一种概率损失函数来学习 Ranking Function**，并应用 Ranking Function 对文档进行排序。这里的 Ranking Function 可以是任意对参数可微的模型，也就是说，该概率损失函数并不依赖于特定的机器学习模型，在论文中，RankNet 是基于神经网络实现的。除此之外，GDBT、LR、MF 等模型也可以应用于该框架。

![](https://ningshixian.github.io/resources/images/ranknet.png)

**预测相关性概率**

RankNet 网络将输入 query 的特征向量 $x∈R^n$ 映射为一个实数 $f(x)∈R$。其实它的整个过程其实与上面介绍的 BPR 十分相似。具体地，给定特定 query 下的两个文档 $U_i$ 和 $U_j$，其特征向量分别为 $x_i$ 和 $x_j$，经过 RankNet 进行前向计算得到对应的分数为 $s_i=f(x_i)$ 和 $s_j=f(x_j)$。用 $Ui⊳Uj$ 表示 $U_i$ 比 $U_j$  排序更靠前。RankNet 把排序问题转换成比较一个 $(U_i,U_j)$ 文档对的排序概率问题，它首先计算每个文档的得分，再用下面的公式来计算**同一个Query下 $U_i$ 应该比 $U_j$ 更相关的概率**：
$$
P_{ij} = P(x_i \rhd x_j) = \frac{1}{1 + \exp\bigl(-\sigma \cdot (s_i - s_j)\bigr)}
$$
这个概率实际上就是深度学习中经常使用的 sigmoid 函数，由于参数 σ 影响的是 sigmoid 函数的形状，对最终结果影响不大，因此通常默认使用 σ=1 进行简化。

RankNet 证明了如果知道一个待排序文档的排列中相邻两个文档之间的排序概率，则通过推导可以算出每两个文档之间的排序概率。因此对于一个待排序文档序列，只需计算相邻文档之间的排序概率，不需要计算所有 pair，减少计算量。

**真实相关性概率**

对于特定的 query，定义 $S_{ij} \in \{0,1,-1\}$ 为文档 $Ui$ 和文档 $Uj$ 被标记的标签之间的关联，
- 即$S_{ij}$
  - 1: 文档 i 比 文档 j 更相关
  - 0: 文档 i和文档j相关性一致
  - -1: 文档 j比文档i更相关

我们定义 $d_i$ 比 $d_j$ 更相关（排序更靠前）的真实概率为：
$$
\bar{P}_{i j}=\frac{1}{2}\left(1+S_{i j}\right)
$$
**RankNet 的损失函数**

对于一个排序，RankNet 从各个 doc 的相对关系来评价排序结果的好坏，排序的效果越好，那么有错误相对关系的 pair 就越少。所谓错误的相对关系即如果根据模型输出 $Ui$ 排在 $Uj$ 前面，但真实 label 为 $Ui$ 的相关性小于 $Uj$，那么就记一个错误 pair，RankNet 本质上就是以错误的 pair 最少为优化目标，也就是说 RankNet 的目的是优化逆序对数。而在抽象成损失函数时，RankNet 实际上是引入了概率的思想：不是直接判断 $Ui$ 排在 $Uj$ 前面，而是说 $Ui$ 以一定的概率 $P$ 排在 $Uj$ 前面，即是以预测概率与真实概率的差距最小作为优化目标。**最后，RankNet 使用交叉熵（Cross Entropy）作为损失函数，来衡量 $P_{ij}$ 对 $\bar{P}_{i j}$ 的拟合程度**：（注：这里为了与原论文保持一致，用 C 表示损失函数，但实际上，更习惯用 L。）
$$
C=-\bar{P}_{i j} \log P_{i j}-\left(1-\bar{P}_{i j}\right) \log \left(1-P_{i j}\right)
$$
Ranknet 最终目标是训练出一个打分函数 $s=f(x;w)$，使得所有 pair 的排序概率估计的损失最小，即：
$$
C=\sum_{(i, j) \in I} C_{i j}
$$
其中，$I$ 表示所有在同一 query 下，且具有不同相关性判断的 doc pair，每个 pair 有且仅有一次。

通常，这个打分函数只要是光滑可导就行，比如 $f(x)=wx$ 都可以，RankNet 使用了两层神经网络：
$$
f(x)=g^{3}\left(\sum_{j} w_{i j}^{32} g^{2}\left(\sum_{k} w_{j k}^{21} x_{k}+b_{j}^{2}\right)+b_{i}^{3}\right)
$$
**参数更新**

RankNet 采用神经网络模型优化损失函数，也就是后向传播过程，采用梯度下降法求解并更新参数：
$$
w_{k}=w_{k}-\eta \frac{\partial C}{\partial w_{k}}
$$
其中，η 是学习率，论文中实验时选取的范围是 1e-3 到 1e-5，因为 RankNet 是一种方法框架，因此这里的 wk 可以是 NN、LR、GBDT 等算法的权重。

对RankNet的梯度进行因式分解，有：
$$
\frac{\partial L}{\partial w_k} =
\sum_{(i, j) \in P}\frac{\partial L_{ij}}{\partial w_k} =
\sum_{(i, j) \in P}
\frac{\partial L_{ij}}{\partial s_i}\frac{\partial s_i}{\partial w_k}
+
\frac{\partial L_{ij}}{\partial s_j}\frac{\partial s_j}{\partial w_k}
$$
不难证明：
$$
\frac{\partial L_{ij}}{\partial s_i} = \sigma \cdot \Biggl[\frac12(1 - S_{ij}) -\frac{1}{1 + \exp\bigl(\sigma \cdot (s_i - s_j)\bigr)}\Biggr] = -\frac{\partial L_{ij}}{\partial s_j}
$$
由公式可知，结果排序最终由模型得分 $si$ 确定，它的梯度很关键，于是我们定义：$\lambda_{ij}\mathrel{\stackrel{\mbox{def}}{=}} \frac{\partial L_{ij}}{\partial s_i} = -\frac{\partial L_{ij}}{\partial s_j}$.

**在构造样本Pair时，我们可以始终令i为更相关的文档，此时始终有 $S_{ij}=1$，于是有简化，**
$$
\lambda_{i j} \stackrel{\text { def }}{=}-\frac{\sigma}{1+e^{\sigma\left(s_{i}-s_{j}\right)}}
$$
这个 $\lambda_{ij}$ 就是接下来要介绍的 LambdaRank 和 LambdaMART 中的 Lambda，称为 Lambda 梯度。

**RankNet 小结**

排序问题的评价指标一般有 NDCG、ERR、MAP、MRR 等，这些指标的特点是不平滑、不连续，无法求梯度，因此无法直接用梯度下降法求解。RankNet 的创新点在于没有直接对这些指标进行优化，而是间接把优化目标转换为可以求梯度的基于概率的交叉熵损失函数进行求解。因此任何用梯度下降法优化目标函数的模型都可以采用该方法，RankNet 采用的是神经网络模型，其他类似 boosting tree 等模型也可以使用该方法求解。

**基于Keras的RankNet实现:**

```text
# 网络参数
def create_base_network(input_dim):
    seq = Sequential()
    seq.add(Dense(input_dim, input_shape=(input_dim,), activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(64, activation='relu'))
    seq.add(Dropout(0.1))
    seq.add(Dense(32, activation='relu'))
    seq.add(Dense(1))
    return seq

# Rank Cost层
def create_meta_network(input_dim, base_network):
    input_a = Input(shape=(input_dim,))
    input_b = Input(shape=(input_dim,))

    rel_score = base_network(input_a)
    irr_score = base_network(input_b)

    # subtract scores
    diff = Subtract()([rel_score, irr_score])

    # Pass difference through sigmoid function.
    prob = Activation("sigmoid")(diff)

    # Build model.
    model = Model(inputs = [input_a, input_b], outputs = prob)
    model.compile(optimizer = "adam", loss = "binary_crossentropy")

    return model
```



## Listwise Ranker

[基于 softmax 的 listwise ranking loss 是在 GSF 等 neural ranking models 中比较常用的损失函数，特别是在基于用户行为数据训 neural ranking model 的非偏(unbiased)学习框架中](https://zhuanlan.zhihu.com/p/138436960)

Listwise方法的几个优势：

- 原始数据无需组pair，从而避免了因组pair导致的数据量、数据大小的成倍增长。这也一定程度上加快了训练过程。
- 优化目标为NDCG，通过指定NDCG截断个数，可以忽略大量尾部带噪声的样本的排序，从而集中优化前几位的排序。
- 直接利用原始数据的打分信息进行排序学习，避免了通过分数大小组pair带来的信息损失。

### *LambdaRank

以错误pair最少为优化目标的RankNet算法，然而许多时候仅以错误pair数来评价排序的好坏是不够的，像NDCG或者ERR等评价指标就只关注top k个结果的排序，当我们采用RankNet算法时，往往无法以这些指标为优化目标进行迭代，所以RankNet的优化目标和IR评价指标之间还是存在gap的。

假设 RankNet 经过两轮迭代实现下图所示的顺序优化：

![img](https://images2015.cnblogs.com/blog/995611/201704/995611-20170411090321032-1845715176.png)

如上图所示，每个线条表示文档，蓝色表示相关文档，灰色表示不相关文档，左图的cost为13，右图通过把第一个相关文档下调3个位置，第二个文档上条5个位置，将cost降为11，但是像NDCG或者ERR等评价指标只关注top k个结果的排序，在优化过程中下调前面相关文档的位置不是我们想要得到的结果。图 1右图左边黑色的箭头表示RankNet下一轮的调序方向和强度，但我们真正需要的是右边红色箭头代表的方向和强度，即**更关注靠前位置的相关文档的排序位置的提升**。LambdaRank正是基于这个思想演化而来，其中Lambda指的就是红色箭头，代表下一次迭代优化的方向和强度，也就是梯度

**RankNet 以优化逆序对数为目标，并没有考虑位置的权重，这种优化方式对 AUC 这类评价指标比较友好，但实际的排序结果与现实的排序需求不一致，现实中的排序需求更加注重头部的相关度，排序评价指标选用 NDCG 这一类的指标才更加符合实际需求。而 RankNet 这种以优化逆序对数为目的的交叉熵损失，并不能直接或者间接优化 NDCG 这样的指标。**

我们知道 NDCG 是一个处处非平滑的函数，直接以它为目标函数进行优化是不可行的。那 LambdaRank 又是如何解决这个问题的呢？

我们必须先有这样一个洞察，**对于绝大多数的优化过程来说，目标函数很多时候仅仅是为了推导梯度而存在的。而如果我们直接就得到了梯度，那自然就不需要目标函数了。**

于是，微软学者经过分析，就直接把 RankNet 最后得到的 Lambda 梯度拿来作为 LambdaRank 的梯度来用了，这也是 LambdaRank 中 Lambda 的含义。这样我们便知道了 LambdaRank 其实是一个经验算法，它不是通过显示定义损失函数再求梯度的方式对排序问题进行求解，而是分析排序问题需要的梯度的物理意义，直接定义梯度，即 Lambda 梯度。有了梯度，就不用关心损失函数是否连续、是否可微了，所以，微软学者直接把 NDCG 这个更完善的评价指标与 Lambda 梯度结合了起来，就形成了 LambdaRank。

我们来分析一下 Lambda 梯度的物理意义。

LambdaRank 中的 Lambd 其实就是 RankNet 中的梯度 $λ_{ij}$，**$λ_{ij}$可以看成是 $Ui$ 和 $Uj$ 中间的作用力，代表下一次迭代优化的方向和强度**。如果 $Ui⊳Uj$，则 $Uj$ 会给予 $Ui$ 向上的大小为 $|λij|$ 的推动力，而对应地 $Ui$ 会给予 $Uj$ 向下的大小为 $|λij|$ 的推动力。对应上面的那张图，Lambda 的物理意义可以理解为图中箭头，即优化趋势，因此有些人也会把 Lambda 梯度称之为箭头函数。

LambdaRank在RankNet的加速算法形式的基础上引入评价指标Z （如NDCG、ERR等），把交换两个文档的位置引起的**评价指标的变化 $|ΔNDCG|$ ，作为其中一个因子乘以 $λ_{ij}$ 就可以得到 LambdaRank 的 Lambda**，即：
$$
\lambda_{i j} \stackrel{\text { def }}{=} \frac{\partial C\left(s_{i}-s_{j}\right)}{\partial s_{i}}=\frac{-\sigma}{1+e^{\sigma\left(s_{i}-s_{j}\right)}}\left|\Delta_{N D C G}\right|
$$
其中 $|ΔNDCG|$ 是 $Ui$ 和 $Uj$ 交换排序位置得到的 NDCG 差值。

NDCG 倾向于将排名高并且相关性高的文档更快地向上推动，而排名低而且相关性较低的文档较慢地向上推动，这样通过引入 IR 评价指标（Information Retrieval 的评价指标包括：MRR，MAP，ERR，NDCG 等）就实现了类似上面图中红色箭头的优化趋势。**可以证明，通过此种方式构造出来的梯度经过迭代更新，最终可以达到优化 NDCG 的目的。**

当然，我们还可以反推一下 LambdaRank 的损失函数，确切地应该称为效用函数（utility function）：
$$
C_{i j}=\log \left(1+e^{-\sigma\left(s_{i}-s_{j}\right)}\right) \cdot\left|\Delta Z_{i j}\right|
$$
这里的 $Z$ 泛指可用的评价指标，但最常用的还是 NDCG。

**总结一下**，LambdaRank提供了一种思路：绕过目标函数本身，直接构造一个特殊的梯度，按照梯度的方向修正模型参数，最终能达到拟合NDCG的方法[6]。LambdaRank 的主要突破点：

- 分析了梯度的物理意义；
- 绕开损失函数，直接定义梯度。



### *LambdaMART

> 1. Mart（Multiple Additive Regression Tree，Mart就是GBDT），定义了一个框架，缺少一个梯度。
> 2. LambdaRank重新定义了梯度，赋予了梯度新的物理意义。

LambdaRank 重新定义了梯度，赋予了梯度新的物理意义，因此，所有可以使用梯度下降法求解的模型都可以使用这个梯度，基于决策树的 MART 就是其中一种，将梯度 Lambda 和 MART 结合就是大名鼎鼎的 LambdaMART。LambdaMART是LambdaRank的提升树版本，而LambdaRank又是基于pairwise的RankNet。因此LambdaMART本质上也是属于pairwise排序算法，只不过引入Lambda梯度后，还显示的考察了列表级的排序指标，如NDCG等，因此，也可以算作是listwise的排序算法。LambdaMART由微软于2010年的论文[Adapting Boosting for Information Retrieval Measures](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/LambdaMART_Final.pdf)提出，截止到2018.12.12，引用量达到308。本小节主要介绍LambdaMART，参考文章 [From RankNet to LambdaRank toLambdaMART: An Overview](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.180.634&rep=rep1&type=pdf)。

LambdaMART是基于 LambdaRank 算法和 MART (Multiple Additive Regression Tree) 算法，将排序问题转化为回归决策树问题。GBDT 的核心思想是在不断的迭代中，新一轮迭代产生的回归决策树模型拟合损失函数的梯度，最终将所有的回归决策树叠加得到最终的模型。LambdaMART 使用一个特殊的 Lambda 值来代替上述梯度，也就是将 LambdaRank 算法与 MART 算法结合起来，是一种能够支持非平滑损失函数的学习排序算法。

LambdaMART 在 GBDT 的过程中做了一个很小的修改。原始 GBDT 的原理是直接在函数空间对函数进行求解模型，结果由许多棵树组成，每棵树的拟合目标是损失函数的梯度，而在 LambdaMART 中这个梯度就换成了 Lambda 梯度，这样就使得 GBDT 并不是直接优化二分分类问题，而是一个改装了的二分分类问题，也就是在优化的时候优先考虑能够进一步改进 NDCG 的方向。

由此，LambdaMART的具体算法过程如下：

![img](http://octopuscoder.github.io/images/search_lambdaMart.png)

可以看出LambdaMART的框架其实就是MART，主要的创新在于中间计算的梯度使用的是Lambda，是pairwise的。MART需要设置的参数包括：树的数量M、叶子节点数L和学习率v，这3个参数可以通过验证集调节获取最优参数。

MART输出值的计算：

1. 首先设置每棵树的最大叶子数，基分类器通过最小平方损失进行分裂，达到最大叶子数量时停止分裂

2. 使用牛顿法得到叶子的输出，计算效用函数对模型得分的二阶导$\frac{ \partial \lambda_{i} }{\partial s_{i} }=\frac{\partial^{2} C}{\partial^{2} s_{i} }$

3. $\frac{\partial^2 C}{\partial^2 s_i} = \sum_{ {i,j }=I} \sigma^2 | \Delta Z_{ij} | \rho_{ij} \left( 1-\rho_{ij} \right)$

4. 得到第m颗树的第k个叶子的输出值:
   $$ \gamma_{km}=\frac{\sum_{x_{i} \in R_{km} } \frac{\partial C}{\partial s_{i} } }{\sum_{x_{i} \in R_{km} } \frac{\partial^{2} C}{\partial^{2} s_{i} } }=\frac{-\sum_{x_{i} \in R_{km} } \sum_{ {i, j} \rightleftharpoons I} \left| \Delta Z_{ij}\right| \rho_{ij} }{ \sum_{x_{i} \in R_{km} } \sum_{ {i,j} \rightleftharpoons I} \left| \Delta Z_{ij} \right| \sigma \rho_{ij} \left( 1- \rho_{ij} \right) } $$

5. $x_i$为第i个样本，$x_i∈R_{km}$意味着落入该叶子的样本，这些样本共同决定了该叶子的输出值。

**LambdaMART 具有很多优势：**

1. 适用于排序场景：不是传统的通过分类或者回归的方法求解排序问题，而是直接求解；
2. 损失函数可导：通过损失函数的转换，将类似于 NDCG 这种无法求导的 IR 评价指标转换成可以求导的函数，并且富有了梯度的实际物理意义，数学解释非常漂亮；
3. 增量学习：由于每次训练可以在已有的模型上继续训练，因此适合于增量学习；
4. 组合特征：因为采用树模型，因此可以学到不同特征组合情况；
5. 特征选择：因为是基于 MART 模型，因此也具有 MART 的优势，每次节点分裂选取 Gain 最⼤的特征，可以学到每个特征的重要性，因此可以做特征选择；
6. 适用于正负样本比例失衡的数据：因为模型的训练对象具有不同 label 的文档 pair，而不是预测每个文档的 label，因此对正负样本比例失衡不敏感。



PS: 需要注意的是，LmabdaRank 和 LambdaMART 虽然看起来很像配对法（Pairwise），但其实都属列表法（Listwise），确切来说是 Listwise 和 Pairwise 之间的一种混合方法。列表法从理论上和研究情况来看，都是比较理想的排序学习方法。因为列表法尝试统一排序学习的测试指标和学习目标。尽管在学术研究中，纯列表法表现优异，但是在实际中，类似于 LambdaRank 和 LambdaMART 这类思路，也就是基于配对法和列表法之间的混合方法更受欢迎。因为从总体上看，纯列表法的运算复杂度都比较高，而在工业级的实际应用中，真正的优势并不是特别大，因此纯列表法的主要贡献目前还多是学术价值。

PS: LambdaMART是LambdaRank的提升树版本，而LambdaRank又是基于pairwise的RankNet。因此LambdaMART本质上也是属于pairwise排序算法，只不过引入Lambda梯度后，还显示的考察了列表级的排序指标，如NDCG等，因此，也可以算作是listwise的排序算法。



**Summary: From RankNet to LambdaRank to LambdaMART**

|                          | RankNet                       | LambdaRank                                                   | LambdaMART                                                   |
| ------------------------ | ----------------------------- | ------------------------------------------------------------ | ------------------------------------------------------------ |
| Object                   | Cross  entropy over the pairs | Unknown                                                      | Unknown                                                      |
| Gradient   (λ  function) | Gradient of cross entropy     | Gradient of cross entropy **times** **pairwise** **change  in target metric** | Gradient of cross entropy **times** **pairwise** **change  in target metric** |
| Optimization method      | neural network                | stochastic gradient descent                                  | Multiple Additive Regression Trees  (MART)                   |



## Ranking Metric Optimization

[排序学习调研](http://xtf615.com/2018/12/25/learning-to-rank/)

前面我们提到，不管是pointwise还是pairwise都不能直接优化排序度量指标。在listwise中，我们通过定义Lambda梯度来优化排序度量指标，如LambdaRank和LambdaMART，然后Lambda梯度是一种经验性方法，缺乏理论指导。最近，有研究者提出了优化排序度量指标的概率模型框架，叫做LambdaLoss（[CIKM18: The LambdaLoss Framework for Ranking Metric Optimization](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/1e34e05e5e4bf2d12f41eb9ff29ac3da9fdb4de3.pdf)），提供了一种EM算法来优化Metric驱动的损失函数。文中提到LambdaRank中的Lambda梯度在LambdaLoss框架下，能够通过定义一种良好、特殊设定的损失函数求解，提供了一定的理论指导。

传统的pointwise或pairwise损失函数是平滑的凸函数，很容易进行优化。有些工作已经证明优化这些损失的结果是真正排序指标的界，即实际回归或分类误差是排序指标误差(取反)的上界([bound](https://papers.nips.cc/paper/3270-mcrank-learning-to-rank-using-multiple-classification-and-gradient-boosting.pdf))。但是这个上界粒度比较粗，因为优化不是metric驱动的。

然而，直接优化排序指标的挑战在于，排序指标依赖于列表的排序结果，而列表的排序结果又依赖于每个物品的得分，导致排序指标曲线要么不是平坦的，要么不是连续的，即非平滑，也非凸。因此，梯度优化方法不实用，尽管有些非梯度优化方法可以用，但是时间复杂度高，难以规模化。为了规模化，目前有3种途径，1是近似法，缺点是非凸，容易陷入局部最优；2是将排序问题转成结构化预测问题，在该方法中排序列表当做最小单元来整体对待，损失定义为实际排序列表和最理想排序列表之间的距离，缺点是排序列表排列组合数量太大，复杂度高；3是使用排序指标在迭代过程中不断调整样本的权重分布(回顾下WRAP就是这种，LambdaRank也属于这种，|ΔNDCG||ΔNDCG|就可以看做是权重。)，这个方法优点是既考虑了排序指标，同时也是凸优化问题。

本文的动机之一就是探索LambdaRank中提出的Lambda梯度真正优化的损失函数是什么。文章通过提出LambdaLoss概率框架，来解释LambdaRank可以通过EM算法优化LambdaLoss得到。进一步，可以在LambdaLoss框架下，定义基于排序和得分条件下，metric-driven的损失函数。



## Unbiased Learning-to-Rank

[排序学习调研](http://xtf615.com/2018/12/25/learning-to-rank/)

先前的[研究](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/45286.pdf)表明，受到排版的影响，给定排序的项目列表，无论其相关性如何，用户更有可能与前几个结果进行交互，这实际上是一种位置偏差（Position Bias），即，被交互和用户真正喜欢存在差距，因为靠前，人们更倾向于交互，而很多靠后未点击的物品，用户可能更感兴趣，倘若将这些靠后的物品挪到前面去，那用户交互的概率可能会更高。 这一观察激发了研究人员对无偏见的排序学习(Unbiased Learning-to-Rank)的兴趣。常见的解决途径包括，对训练实例重新赋权来缓解偏差；构造无偏差损失函数进行无偏排序学习；评估时使用无偏度量指标等。



# 总结

搜索引擎时敲定特征分的权重是非常疼的一件事儿，而LTR正好帮你定分，LTR的三种实现方法其实各有优劣：

- 难点主要在训练样本的构建(人工或者挖日志)，另外就是训练复杂
- 虽说Listwise效果最好，pointwise使用还是比较多，参考这篇文章[综述LTR](http://www.cnblogs.com/zjgtan/p/3652689.html)体验下
- 在工业界用的比较多的应该还是Pairwise，因为他构建训练样本相对方便，并且复杂度也还可以，所以Rank SVM就很火



# 未来展望

对于CTR预估和排序学习的领域，目前深度学习尚未在自动特征挖掘上对人工特征工程形成碾压之势，因此人工特征工程依然很重要（[参考](https://tech.meituan.com/2019/01/17/dianping-search-deeplearning.html)）

- 基于深度学习的CTR预估模型改造
  - 基于Lambda梯度，通过深度网络进行反向传播，训练一个优化NDCG的深度网络[LambdaDNN](https://tech.meituan.com/2019/01/17/dianping-search-deeplearning.html)；
  - **DSSM model.** This method significantly outperforms the traditional lexical ranking model, e.g., TF-IDF and BM25.
  - **Bert.** It have achieved more impressive results on the passage re-ranking task
- 模型组合（GBDT+LR，Deep&Wide等）



# 参考资料

1 [[延伸\]Tie-Yan Liu. Learning to Rank for Information Retrieval. Springer](https://lumingdong.cn/go/nnslnt)[↩](https://lumingdong.cn/learning-to-rank-in-recommendation-system.html#ref-footnote-1)

2 [[延伸\]Hang Li. Learning to rank for information retrieval and natural language processing](https://lumingdong.cn/go/rszqpc)[↩](https://lumingdong.cn/learning-to-rank-in-recommendation-system.html#ref-footnote-2)

3 [[延伸\]Li Hang. A Short Introduction to Learning to Rank](https://lumingdong.cn/go/rui0pc)[↩](https://lumingdong.cn/learning-to-rank-in-recommendation-system.html#ref-footnote-3)

4 [Learning to rank 学习基础](https://lumingdong.cn/go/69up9k)[↩](https://lumingdong.cn/learning-to-rank-in-recommendation-system.html#ref-footnote-4)

5 [达观数据. 资讯信息流 LTR 实践. 机器之心](https://lumingdong.cn/go/5ncae0) [↩](https://lumingdong.cn/learning-to-rank-in-recommendation-system.html#ref-footnote-5)

6 [深度学习在美团推荐平台排序中的运用（2017年）](https://lumingdong.cn/go/79khq5)[↩](https://lumingdong.cn/learning-to-rank-in-recommendation-system.html#ref-footnote-6)

7 [吴佳金等. 改进 Pairwise 损失函数的排序学习方法. 知网](https://lumingdong.cn/go/wunexd)[↩](https://lumingdong.cn/learning-to-rank-in-recommendation-system.html#ref-footnote-7)

8 ★ [美团技术团队. 深入浅出排序学习：写给程序员的算法系统开发实践](https://tech.meituan.com/2018/12/20/head-in-l2r.html)[↩](https://lumingdong.cn/learning-to-rank-in-recommendation-system.html#ref-footnote-6)

★ [BERT在美团搜索核心排序的探索和实践 - 业界良心总结](https://tech.meituan.com/2020/07/09/bert-in-meituan-search.html)

★ [路明东的博客. 推荐系统中的排序学习√](https://lumingdong.cn/learning-to-rank-in-recommendation-system.html)

9 [路明东的博客. 推荐系统技术演进趋势：从召回到排序再到重排](https://lumingdong.cn/technology-evolution-trend-of-recommendation-system.html)

10 [Christopher J.C. Burges. From RankNet to LambdaRank to LambdaMART: An Overview. Microsoft](https://lumingdong.cn/go/4xmnri)[↩](https://lumingdong.cn/learning-to-rank-in-recommendation-system.html#ref-footnote-10)

11 [笨兔勿应. Learning to Rank 算法介绍. cnblogs](https://lumingdong.cn/go/4zadd4)[↩](https://lumingdong.cn/learning-to-rank-in-recommendation-system.html#ref-footnote-11)

12 [RL-Learning. 机器学习排序算法：RankNet to LambdaRank to LambdaMART. cnblogs](https://lumingdong.cn/go/o0jv29)[↩](https://lumingdong.cn/learning-to-rank-in-recommendation-system.html#ref-footnote-12)

13 [huagong_adu. Learning To Rank 之 LambdaMART 的前世今生. CSDN](https://lumingdong.cn/go/4g9ur9)[↩](https://lumingdong.cn/learning-to-rank-in-recommendation-system.html#ref-footnote-13)

[小菜鸡. Learning to rank算法（2019-11-18）](https://octopuscoder.github.io/2019/11/18/Learning-to-rank算法/)

★ [小菜鸡. 搜索排序算法（2020-01-19）]([http://octopuscoder.github.io/2020/01/19/%E6%90%9C%E7%B4%A2%E6%8E%92%E5%BA%8F%E7%AE%97%E6%B3%95/](http://octopuscoder.github.io/2020/01/19/搜索排序算法/))

★ [小菜鸡. LambdaMART从放弃到入门（2020-03-27）]([https://octopuscoder.github.io/2020/03/27/LambdaMART%E4%BB%8E%E6%94%BE%E5%BC%83%E5%88%B0%E5%85%A5%E9%97%A8/](https://octopuscoder.github.io/2020/03/27/LambdaMART从放弃到入门/))

★ [排序学习调研](http://xtf615.com/2018/12/25/learning-to-rank/)

[排序学习（Learning to rank）综述](https://blog.csdn.net/anshuai_aw1/article/details/86018105)

对比各种wise的优缺点：https://blog.csdn.net/lipengcn/article/details/80373744

[【腾讯技术分享】一：搜索排序—概述](https://cloud.tencent.com/developer/article/1523867?from=10680)

[机器学习排序](https://blog.csdn.net/hguisu/article/details/7989489)

[用好学习排序 (LTR) ,资讯信息流推荐效果翻倍](https://www.jiqizhixin.com/articles/2019-03-05-13)

★ [Introduction to Learning to Rank - 可用于快速复习掌握要点和核心代码](https://everdark.github.io/k9/notebooks/ml/learning_to_rank/learning_to_rank.html)

★ [jiangnanboy / learning_to_rank 参考代码](https://github.com/jiangnanboy/learning_to_rank)

[排序优化算法Learning to Ranking](https://www.biaodianfu.com/learning-to-ranking.html)
