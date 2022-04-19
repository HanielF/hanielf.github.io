---
title: RNAi和siRNA设计基础
tags:
  - RNAi
  - siRNA
  - Note
  - Basic
  - Summary
categories:
  - Notes
comments: true
mathjax: false
date: 2019-10-01 18:40:22
urlname: basic-knowledge-of-RNAi
---

<meta name="referrer" content="no-referrer" />

{% note info %}
刚接触siRNA搜索、设计和评测这方面的研究，记录一下基本的概念和一些基础知识，还有自己的理解总结。
{% endnote %}
<!--more-->

## 基础概念
### 英文缩写
1. RNAi：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;RNA干扰
1. siRNA：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;小干扰RNA
1. dsRNA：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; Double-stranded RNA的缩写，是指双链核糖核酸
1. [shRNA](https://baike.baidu.com/item/shRNA)：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 是英文单词short hairpin RNA的缩写。翻译为“短发夹RNA”。shRNA包括两个短反向重复序列。克隆到shRNA表达载体中的shRNA包括两个短反向重复序列，中间由一茎环（loop）序列分隔的，组成发夹结构，由polⅢ启动子控制。随后再连上5-6个T作为RNA聚合酶Ⅲ的转录终止子。
1. Argonaute (AGO)：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一类庞大的蛋白质家族，是组成RISCs复合物的主要成员。AGO蛋白质主要包含两个结构域：PAZ和PIWI两个结构域，但具体功能现在尚不清楚。
1.  Dicer酶：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;是RNAase Ⅲ家族中的一员，主要切割dsRNA或者茎环结构的RNA前体成为小RNAs分子。对应地，我们将这种小RNAs分子命名为siRNAs和miRNA。Dicer有着较多的结构域，最先在果蝇中发现，并且在不同的生物体上表现出很高的保守性。

1. RISC：RISC诱导沉默复合体（全称：RNA-induced silencing complex）：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;一种由siRNA与Argonaute蛋白和Dicer酶复合形成的复合物。在RNAi中，利用siRNA的反义链切割靶mRNA，达到基因沉默。

1. BLAST (Basic Local Alignment Search Tool)：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp; 是一套在蛋白质数据库或DNA数据库中进行相似性比较的分析工具。BLAST程序能迅速与公开数据库进行相似性序列比较。BLAST结果中的得分是对一种对相似性的统计说明。

1. rRNA（核糖体RNA）是核糖体的组成成分,它和蛋白质共同组成了核糖体.  
tRNA（转运RNA）可以转运氨基酸.
mRNA（信使RNA）是由细胞核内的DNA转录来的,相当于蛋白质的设计图纸.

1. UTR：非翻译区

1. [SD序列](http://www.baike.com/wiki/SD%E5%BA%8F%E5%88%97)：mRNA起始部位的碱基序列，为mRNA与核糖体的结合位点称SD序列.在DNA上相应的位点也称SD序列，一般位于操纵基因和第一个结构基因之间，部分序列与操纵基因重叠.


### 名词概念
1. 质粒：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;小型环状DNA分子，包括三部分：遗传标记基因，复制区，目的基因.在所有的细菌类群中都可发现，它们是独立于细菌染色体外自我复制的DNA分子
1. 质粒载体：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;质粒载体是在天然质粒的基础上为适应实验室操作而进行人工构建的质粒。与天然质粒相比，质粒载体通常带有一个或一个以上的选择性标记基因（如抗生素抗性基因）和一个人工合成的含有多个限制性内切酶识别位点的多克隆位点序列，并去掉了大部分非必需序列，使分子量尽可能减少，以便于基因工程操作

1. 转染：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;是真核细胞主动或被动导入外源DNA片段而获得新的表型的过程

1. 细胞株：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;通过选择法或克隆形成法从原代培养物或细胞系中获得具有特殊性质或标志物的培养物称为细胞株。    
细胞株是用单细胞分离培养或通过筛选的方法，由单细胞增殖形成的细胞群。细胞株的特殊性质或标志必须在整个培养期间始终存在。原代培养物经首次传代成功后即为细胞系(cell line)， 由原先存在于原代培养物中的细胞世系所组成。如果不能继续传代，或传代次数有限， 可称为有限细胞系(finite cell line)， 如可以连续培养， 则称为连续细胞系(continuous cell line)， 培养50代以上并无限培养下去。 所以细胞株是通过选择法或克隆形成法从原代培养物或细胞系中获得的具有特殊性质或标志的培养细胞。从培养代数来讲，可培养到40-50代。细胞株的特殊性质或标志必须在整个培养期间始终存在。对于人类肿瘤细胞，在体外培养半年以上，生长稳定，并连续传代的即可称为连续性株或系。

1. 核酸酶：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;能够将聚核苷酸链的磷酸二酯键切断的酶，称为核酸酶。 有些核酸酶只能作用于RNA，称为核糖核酸酶（RNase），有些核酸酶只能作用于DNA，称为脱氧核糖核酸酶（DNase），有些核酸酶专一性较低，既能作用于RNA也能作用于DNA，因此统称为核酸酶（nuclease）。根据核酸酶作用的位置不同，又可将核酸酶分为核酸外切酶（exonuclease）和核酸内切酶

1. 逆转录病毒：[病毒的分类](https://www.zhihu.com/question/23745748)

1. [DNA中的3'端和5'端](https://www.zhihu.com/question/21112790)

1. 化学修饰：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;凡通过化学基团的引入或除去，而使蛋白质或核酸共价结构发生改变的现象。它以引起酶分子共价键的变化、化学结构的改变而影响酶活性。酶的化学修饰是在另一种酶的催化下完成的，是体内快速调节的另一种重要方式。

1. 正义链：
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;与转录出的mRNA序列相同的（DNA中的T在RNA中为U）那条DNA单链那条链为正义链，与之互补的为反义链。

1. PCR扩增产物：可分为长产物片段和短产物片段两部分。短产物指数倍增加，长产物算术倍增加，所以重复多次后，可以忽略长产物，得到的是目的基因。
反应体系由模板DNA、一对引物、dNTP、耐高温的DNA聚合酶、酶反应缓冲体系及必须的离子等所组成。PCR反应循环的第一步为加热变性，使双链模板DNA变性为单链；第二步为复性，每个引物将与互补的DNA序列杂交；第三步为延伸，在耐高温的DNA聚合酶作用下，以变性的单链DNA为模板，从引物3ˊ端开始按5ˊ→3ˊ方向合成DNA链。这样经过一个周期的变性——复性——延伸等三步反应就可以产生倍增的DNA，假设PCR的效率为100%,反复n周期后，理论上就能扩增2n倍。PCR反应一般30-40次循环，DNA片段可放大数百万倍。
见[PCR的扩增产物是什么，怎么扩增出来的](https://zhidao.baidu.com/question/345141221.html)

1. guide strand 和 passenger strand：
guide strand 是被整合到RISC中的，而passenger strand 被降解。
guide strand 等同于 antisense strand，passenger strand 等同于sense strand。
guide(antisense) strand和转录出的mRNA是互补配对的

{% asset_img applications-sirna-general-structure-of-sirna.jpg "General structure of siRNA. Two RNA strands form a duplex 21 bp long with 3' dinucleotide overhangs on each strand. The antisense strand is a perfect reverse complement of the intended target mRNA." %}

**设计siRNA的时候，从cDNA选19bp长度的序列，然后互补配对，`A-U/C-G/G-C/T-A`，然后反向就是得到了guide strand/antisense strand，然后再在末尾添加两个悬垂就是得到了Guide strand**


### 基本单位
1. 碱基(A，G，C，T，U)：
&nbsp;&nbsp;&nbsp;&nbsp;DNA有四种碱基对，即鸟嘌呤(G),腺嘌呤(A),胞嘧啶(C),胸腺嘧啶(T).其中G与C配对，A与T配对。RNA有A，G，C，U(尿嘧啶)这四种

1. bp：
&nbsp;&nbsp;&nbsp;&nbsp; 碱基对的数目单位,100bp即两条链上各有100个碱基.

1. nt：
&nbsp;&nbsp;&nbsp;&nbsp;核苷酸的单位，25nt RNA是指25个核苷酸碱基的小RNA


## RNAi实验相关
1. 四要素：
  - 目标基因的dsRNA，看需求
  - 转染或者将dsRNA送入细胞的方法，具体看下面的三种方法
  - 对照
  - 检测表达情况的方法，即检验RNAi效果
2. 注意点
  - siRNA长度的选择
  - shRNA表达水平太高会导致细胞毒性或脱靶，要防止过表达
3. 三种实现方法

|适用类型| 导入材料|过程 |其他材料| 沉默类型|
|:-:|:-:|:-:|:-:|:-:|
|非哺乳生物|dsRNA| 导入dsRNA，使用Dicer酶切割dsRNA得到siRNAs，获得RISC诱导沉默复合体，切割靶mRNA| Dicer酶|瞬时基因沉默3-7天|
|哺乳生物|siRNAs|直接导入siRNAs，获得RISC，切割靶mRNA|无|瞬时基因沉默3-7天|
| | shRNA的DNA表达载体| 导入shRNA表达载体，使用Dicer酶切割shRNA，得到siRNAs，后续相同|Dicer酶|长效基因沉默|

## siRNA设计
### 结构和选择标准
- 理想siRNA是23bp，双链部分是19bp，正义链和反义链为21bp，3‘端和5’端分别是两个突出核苷酸

|    |5'端|1  | 2 | 3 | 4 | 5...10|11...14|15...18|19 |20|21...23 |3'端|
| :-:|:-: |:-:|:-:|:-:|:-:|    :-:|:-:    |:-:|:-: |:-: |:-: |:-:|
|理想状态|第一个核苷酸对沉默作用非必须 |A|A|-|-|-|-|-|-|T|T|UU/TT结尾，抵抗核糖核苷酸酶|
|有效状态| |-|A|-|-|-|-|-|-|T|T| / |
|有效状态| |-|A|-|-|-|-||--|-|-| / |
|选择标准(正义链：3-23，21nt)| |/|/|5. 为A|-|6. 第十位为U |8. 第13位非G|2. 15-19位有3+个A/U碱基对|4. 为A; 7.不能为G/C|-|-| / |

{% note info %}
正义链：3-23，一般TT结尾
反义链：1-21的互补链(A-T, C-G)，一般TT开头

反义RNA和靶RNA互补配对，参与基因表达调控。分3类：
1. 直接和靶mRNA的S-D序列或部分编码区结合，抑制翻译，或者结合成双链RNA被RNA酶Ⅲ降解
2. 和非编码区结合，引起mRNA构象变化，抑制翻译
3. 直接抑制mRNA转录
{% endnote %}

**选择标准：**
1. G/C含量=30～52%
1. 在第15 ～19位核苷酸的位置个或更多的A/U碱基对（正义链） 
1. 发夹结构预测（没有内部重或回文结构）Tm<20℃ 
1. 19位核苷酸为A（正义链） 
1. 3位核苷酸为A（正义链） 
1. 10位核苷酸为U（正义链）
1. 19位核苷酸不能为G或C（正义链） 
1. 13位核苷酸不能为G（正义链）

**符合条件小于6个的舍弃**

{% note danger %}
1. 如果靶序列开始的两个核苷酸不是AA，选择基因编码区的23个核苷酸来计算G-C的百分含量
1. 大多数G-C含量为30%-52%的siRNA产生的沉默效应较高，但过高的G-C含量会降低沉默活性
1. 避免超过3个G/C重复，以及4个A重复，多G/C可能干扰siRNA沉默机制，多A会提前终止转录干扰shRNA合成
《小干扰RNA的合理设计》中，为2+个G/C降低RNA内在稳定性，3+个U/A可能终止RNA PolymeraseIII介导的转录
1. 一定要从启动子第100个核苷酸之后开始搜索
1. 3’-端可以是合适的靶序列，可特异性防止非必要保守基因的沉默
{% endnote %}


### 设计过程
选择：
1. 在所选基因的启动子`100个`碱基以后开始自5’-端开始
2. 寻找基因序列中的23个碱基， 最好是5‘- AA（N19）TT -3’ （N是任何碱基）
3. 如果找不到以上AA（N19）TT，则用 AA（N21）补足。
4. 如果找不到以上AA（N21）， 则用 NA（N21）补足。
5. 所选定序列中，G和C的数目的总和在总数（23）的35-55%.
6. 满足以上1-5项要求的片段数目如果不足四个，将G和C的数目的总和放宽至总数（23）的30-70%

确定RNA oligo：
1. 找到正义链和反义链
1. 将反义链从3‘-5’反转，变成5'-3'
1. 将除了3‘末端的两个碱基外，所有的T改成U
1. 使用BLAST确定唯一性

{% note %}
**注意点：**
- GC含量为30-50% 的siRNA最有效
- 应选2-4个靶点序列


**需排除的siRNA：**
- 有4个T/A
- 基因组数据库比较，去除和其他序列有16-17同源碱基对的靶序列
{% endnote %}


### 设置对照
所有的RNAi试验均应设立阴性对照，siRNA阴性对照序列的合理设计与siRNA序列的设计同样重要。
因为有效的对照可以充分证明siRNA只对靶基因产生特异性基因沉默，从而增强实验的可信度。
阴性对照siRNA包括碱基错配或混乱序列的siRNA。
在实验中最好设计两条siRNA对照序列

注：
  多碱基错配比单碱基错配的siRNA阴性对照序列具有更高的实际应用价值

设置对照：
- 阴性对照siRNA：1）打乱原siRNA核苷酸顺序，且和靶mRNA无同源性; 2）碱基错配
- 针对相同基因的其他siRNA：确保siRNA数据可靠的最佳方式

### Tuschl法则
原始法则：
- 21bp + 3’两个碱基，最有效

新法则：
- NA -（A/G）-（N17）-（C/U）- NN  
正义链和反义链（21nt）siRNA应该以嘌呤核苷酸(A)开头，这对于多聚酶Ⅲ启动子的表达是必须的。

- （N4）- A -（N6）- T -（N2）-（A/T/C）-（N5）-（A）-（N2）

## 构建shRNA表达载体

### 优势
1. 有抗生素标记，可建立稳定的长期基因沉默细胞株，并筛选细胞
1. 通过病毒插入基因组得到稳定的基因沉默表达细胞株

### 构建过程

#### shRNA序列设计
shRNA 由 siRNA 和环状连接序列组成。
就是目标 siRNA 与其反向互补序列之间由特定的连接序列间隔，得到的 RNA 两端反向互补退火，与连接序列形成茎环结构，类似发夹。

#### 启动子选择
多数siRNA表达载体依赖三种RNA聚合酶III 启动子(pol III)中的一种，操纵一段小的发夹RNA在哺乳动物细胞中的表达，包括U6和H1。
因为它可以在哺乳动物细胞中表达更多小分子RNA

1. 表达产量取决于启动子强弱
1. U6\>H1，表达时间长，首选，但shRNA要避免3+个U/A，防提前转录
1. RNA 聚合酶 II 类的启动子如CMV 启动子和 U1 启动子也比较常见
1. shRNA 序列有连续的 U / T 时应该优先考虑 CMV 启动子载体

#### 载体类型选择
1. 质粒载体
1. 病毒表达载体：感染细胞效率高
常用的，哺乳动物细胞,病毒载体包括：
&nbsp;&nbsp;&nbsp;&nbsp;逆转录病毒(Retrovirus)，腺病毒(Adenovirus)，腺相关病毒(Adeno-Associated Virus,AAV)，和慢病毒(Lentivirus)。
其中逆转录病毒和慢病毒等载体，还可以用于构建，整合到染色体上的,稳定的,长期基因沉默细胞株

具体见[病毒的分类](https://www.zhihu.com/question/23745748)
研究长期基因沉默，选择逆转录病毒，逆转录病毒也是逆转录类型

#### 抗生素筛选标记
最好选一个抗性标记

1. 非必选，但是可以得到稳定表达shRNA的细胞株，且可检测是否转染
1. 促进shRNA表达，防止表达减弱影响沉默效果

#### 构建、克隆和测序验证

## 结果评测
通常从两方面检测，首先是检测mRNA的表达，其次是蛋白表达。如果是移植到小鼠或其他载体上，可以再使用生物学方法检测效果。

### mRNA的表达
#### RT-PCR
最常见的是在转染后24-48小时做定量RT-PCR（逆转录PCR）
1. 纯化细胞
1. 选取合适的RNA提取试剂盒和银光RT-PCR试剂盒
1. 提取RNA，进行扩增
1. 进行RT-PCR检测，按照说明书操作，配置反应体系，于PCR仪扩增实验
1. 观察结果

#### Northern blot杂交

!['肿瘤转移抑制基因 KAil 不同转移潜能癌细胞中的表达'](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/sirna-1.png)

#### 原位杂交
它即可检测 mRNA 的表达，又可观察 mRNA 的定位，是研究细胞内基因表达及有关因素调控的有效工具Northern blot 分析和 RT-PCR 两种方法只用于检测某 特定的 RNA 片段，它们都只能证明细胞或组织中是否存在待测的核酸而不能证明核酸分子在细胞或组织中存在的部位，不具有定位性，也不能反映组织、细胞、器官的差异
 
### 蛋白水平检测
#### 免疫组化法(immunohistochemistry)
- **原理：**免疫组化，免疫组化，是应用免疫学基本原理——抗原抗体反应，即抗原与抗体特异性结合的原理，通过化学反应使标记抗体的显色剂（荧光素、酶、金属离子、同位素）显色来确定组织细胞内抗原（多肽和蛋白质），对其进行定位、定性及定量的研究，称为免疫组织化学技术(immunohistochemistry)或免疫细胞化学技术(immunocytochemistry)。
- **特点：**是融合了免疫学原理（抗原抗体特异性结合）和组织学技术（组织的取材、固定、包埋、切片、脱蜡、水化等），通过化学反应使标记抗体的显色剂(荧光素、酶、金属离子、同位素)显色，来对组织（细胞）内抗原进行定位、定性及定量的研究(主要是定位)。样本是细胞或组织，要在显微镜下观察结果，可能出现膜阳性、质阳性和核阳性。

#### 蛋白免疫印迹( Western Blot) 
- **原理：**蛋白质印迹法是将电泳分离后的细胞或组织总蛋白质从凝胶转移到固相支持物NC膜或PVDF膜上，然后用特异性抗体检测某特定抗原的一种蛋白质检测技术。
- **特点：**先要进行SDS-PAGE，然后将分离开的蛋白质样品用电转仪转移到固相载体上，而后利用抗原-抗体-标记物显色来检测样品，可以用于定性和半定量。

#### ELISA检测 
- **原理：**酶联免疫吸附剂测定法，简称酶联免疫法，或者ELISA法，它的中心就是让抗体与酶复合物结合，然后通过显色来检测。
- **特点：**用到了免疫学原理和化学反应显色，待测的样品多是血清、血浆、尿液、细胞或组织培养上清液，因而没有用到组织包埋、切片等技术，这是与免疫组化的主要区别，操作上 开始需要将抗原或抗体结合到固相载体表面，从而使后来形成的抗原-抗体-酶-底物复合物粘附在载体上，这就是“吸附”的含义。

#### 区别
具体的还是参看[蛋白表达不同检测表达方式的比较和分析](https://wenku.baidu.com/view/67f81ad7d4bbfd0a79563c1ec5da50e2524dd102.html)

![蛋白表达不同检测表达方式的比较和分析](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/sirna-2.png)


<!-- ## miRNA和tRNA -->
<!-- ### miRNA -->
## miRNA
RNAi的重要工具

这种内源性的非编码区小分子 RNA 针对 3’端非编码区，有着极高的保守性，并在组织中广泛表达，可在转录后以及翻译水平上调控基因表达，可能在不影响 mRNA 的水平下调控基因表达

**和siRNA区别：**
- 内源而非外源序列
- 针对非编码区
- 物种进化极为高度保守
- 组织中广泛表达
- miRNA 可能调控多种关键基因
- 可在转录后以及翻译水平上调控基因表达，可能在不影响 mRNA 的水平下调控基因表达

## 总结
### 《小干扰RNA的合理设计》总结
- RNAi主要是要通过siRNA和靶基因结合并使之降解，所以siRNA要和靶基因高度同源，但是不和其他基因同源。
- 搜索siRNA时，靠近基因的3‘端的比较好
- 有研究表明，5’UTR是一个高保守区，使之成为siRNA理想的靶点，RNAi作用于5‘UTR或3’UTR序列，也可以引起靶基因沉默
- 27nt或者29nt的siRNA效果更好，27nt或29nt的siRNA与21ntsiRNA相比：
> (1)其抑制活性可提高数倍以上；
> (2)不易于诱导干扰素反应和激活PKR；
> (3)一些基因对21ntsiRNA不敏感，但是可以被27ntsiRNA有效的抑制；
> (4)与21ntSiRNA相比，27ntSiRNA对靶基因的最大抑制率可在相对低的浓度下得到
- 每一个靶基因，应该设计4+条siRNA
- 只能根据标准设计出理论较高沉默效应的siRNA，最终活性要用实验验证
- 关于shRNA的发夹环长度，3-10nt都可以，Brummelkamp研究表明9nt的抑制率高，Siolas表明长度没有明显影响
- 每个RNAi实验都要设置阴性对照。具体见上文

### 《高效siRNA的设计分析》总结
1. RNAi实验一般流程如下：
  - 准备材料，包括目标基因，shRNA表达载体，限制性核酸内切酶等
  - siRNA的设计和筛选
  - 重组表达载体的构建
  - 含靶基因的细胞的分离培养和转染
  - RT-PCR的检测和靶基因的表达
  - 分析结果并总结
2. 一些结论
  - 和siRNA结合的$$AG0_2$$蛋白主要识别5‘端为A的siRNA，因此5’端为A的siRNA对其进入RISC复合体及对靶mRNA的识别具有关键作用
  - 正义链5’端具有更多的G/C对siRNA的功能提高非常重要，第19位为G/C也更具有RNAi活力
  - 反义链的第19位G/C真的对siRNA的高功能性存在相关性有待验证
  - 高效的siRNA反义链的第13位优先为 A／U，即不为G/C
  - 反义链第10位为 U 作用强于A，优先考虑
  - 综合考虑反义链第3,7,14,16,17位还是有必要的
  - 对zfy基因来说，siRNA靶点在1000bp以内，即第3-6外显子上设计高效siRNA的可能性更大，最好位于150bp-800bp
3. 这个论文主要是记录了一整个siRNA实验以及分析过程，可以作为以后siRNA实验的参考


<!-- ## Question -->
<!-- 1. 选择标准中的G-C含量在什么范围计算，如果不是23个核苷酸中，为什么选择标准注意点说开头不是AA才在23个里计算G-C含量 -->
<!-- 1. 选择标准中，3‘端可以是合适的靶序列，什么意思 -->


## 拓展阅读
1. [生物医药大词典](http://dict.bioon.com/)
1. [siRNA和shRNA:通过基因沉默抑制蛋白表达的工具](http://www.labome.cn/method/siRNAs-and-shRNAs-Tools-for-Protein-Knockdown-by-Gene-Silencing.html)
1. [Argonaute蛋白结构与功能](https://wenku.baidu.com/view/b3571b7c192e45361066f5ca.html)
1. [张美红，周克元.小干扰RNA的合理设计[J].肿瘤防治研究，2006， 33（11）：837-839](http://www.televector.com/9171223659.pdf)
1. [秦炳燕，张永生，纪俊明等.高效siRNA的设计分析[B].黑龙江畜牧兽医，2017（1）：109-113，294](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CJFD&dbname=CJFDLAST2017&filename=HLJX201701028)
1. [韩烨.基于机器学习的siRNA沉默效率预测方法研究[D].吉林省：吉林大学，2019](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CDFD&dbname=CDFDLAST2017&filename=1017152231.nh)
1. [薛婷，王黎明，焦今文等.siRNA介导RRM2基因沉默治疗人卵巢癌裸鼠移植瘤[J].山东大学学报，2019，57（10）](https://kns.cnki.net/KCMS/detail/detail.aspx?dbcode=CAPJ&dbname=CAPJLAST&filename=SDYB20190926000)
1. [李珊珊,任秀花,闫爱华,方伟岗.3种mRNA检测方法比较[J].河南医科大学学报，2000，35（2）：113-114](http://kns.cnki.net/KCMS/detail/detail.aspx?dbname=cjfd2000&filename=hnyk200002007)
1. [蛋白表达不同检测表达方式的比较和分析](https://wenku.baidu.com/view/67f81ad7d4bbfd0a79563c1ec5da50e2524dd102.html)
1. [李珊珊，方伟岗，钟销销，等 肿瘤转移抑制基因 KAil 不同转移潜能癌细胞中的表达．中华医学杂志，1999,(9): 708](http://kns.cnki.net//KXReader/Detail?TIMESTAMP=637062394829091250&DBCODE=CJFD&TABLEName=CJFD9899&FileName=ZHYX199909026&RESULT=1&SIGN=kvPorAe14Lw3IYsBq017HIfdCF4%3d)
