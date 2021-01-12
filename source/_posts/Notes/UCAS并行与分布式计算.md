---
title: UCAS并行与分布式计算
comments: true
mathjax: true
date: 2020-12-29 11:42:24
tags:
  - UCAS
  - ParallelComputing
  - DistributedSystem
  - Notes
  - Review
categories: Notes
urlname: ucas-parallel-and-distributed-computing
---

<meta name="referrer" content="no-referrer" />

{% note info %}

UCAS并行与分布式计算期末复习

{% endnote %}
<!--more-->

## 分布式

### Scalability

可伸缩性，三个尺度：
- 在规模上可伸缩/vertical scalability：可以增加更多的用户和资源
- 在地理上可伸缩/horizontal scalability：用户、资源都可以相距很远
- 在管理上可伸缩：能够很容易地管理相互独立的组织。

### Epidemic protocols

流行病协议

- 一种快速传播信息的方法仅使用本地的大型分布式系统信息
- 用于故障检测，数据聚合，资源发现和监视和数据库复制
- 节点状态：
  - 易感；已感染；已移除
  - 简单的流行病广播算法/永久感染模型：节点始终是易感性或传染性的
- 传播模型
  – 反熵
  – 谣言传播：谣言传播/闲聊
- 优势：高可扩展性和可靠性

#### Anti-Entropy

反熵算法

1. 节点P随机选择另一个Q，然后通过称为push，pull，和push-pull的三种方式之一与Q交换更新。
   - push：P给Q自己的更新
   - pull：P只从Q处拉取更新
   - push-pull：P和Q互相发送更新
2. 轮：将轮定义为一个周期，在该周期中，每个节点将至少一次主动与随机选择的其他节点交换更新。
3. 性能：传播一个更新到所有的节点需要 $O(log(n))$轮，n是系统中节点数
   1. 假设感染进程每轮试图污染 $f$ 个其他进程
      - r回合后被感染成员的预期比例为1
      - 感染整个系统所需的回合数 $R=log_{f+1}(n)+log(n)/f+O(1)$
4. 流行病理论的基本结果是，简单的流行病最终会感染整个系统。
5. push和pull的选择
   1. 假设 $p_i$表示节点在第i轮后保持未感染的概率。
   2. 对于*pull*，假设节点在第i轮后是未感染的，如果节点在第i+1轮后还是未感染，并且它在第i+1轮和一个未感染的节点进行了通信。那么 $p_{i+1} = (p_i)^2$
   3. 对于*push*，假设节点在第i轮后是未感染的，如果节在第i+1轮后还是未感染点，并且没有已经感染的节点选择在第i+1轮和它通信，那么 $p_{i+1} = p_i \cdot (1-\frac{1}{n})^{n(1-p_i)} = p_i \cdot e^{-1}$
   4. **pull或push-pull都比push更好**

#### Gossiping

闲聊算法

1. n个人，最初不活跃（易感）。
1. 然后设置一个active（感染）的人一起散布谣言，随机打给其他人并分享谣言
1. 听到谣言的每个人也变得active，同样也分享谣言的活跃
1. 当活动个体打出不必要的电话（即接收者已经知道谣言）时，活动个体以 $\frac{1}{k}$ 的概率失去了分享谣言的兴趣
   - 该个体被removed
1. 如果未感染人数 $s = \exp^{-(k+1)(1-s)}$，感染的人数可能会降到0，未感染人数s是随着k指数下降的。
2. 因此增加 k 值是一个能确保所有人都听到谣言的高效的方法。也就是降低不分享谣言的概率。
3. 闲聊算法不能确保所有节点都被更新。

### RPC

RPC远程程序调用

Middleware Communication Protocols，即中间件通信协议。

中间件是通用的用于给上层提供服务，并屏蔽下层细节的叫中间件。

1. RPC图示
   1. ![VaVh2C](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/VaVh2C.png)
2. 从常规过程调用中推断出的想法，实现透明度
   - 允许程序调用其他计算机上的过程
3. 如何实现呼叫？
   1. 客户端程序以常规方式调用client stub
   2. client stub生成消息，调用本地OS
   3. 客户端的OS向远程OS发送消息
   4. 远程OS向server stub提供消息
   5. server stub解包参数，调用服务器
   6. 服务器完成任务，将结果返回到 stub
   7. server stub将结果打包在消息中，调用本地OS
   8. 服务器的OS向客户端OS 9发送消息
   9. 客户端OS向 client stub 发送消息
   10. stub 将结果解压缩，返回给客户端
4. 参数传递机制？
   1. 不同编码：EBCDIC和ASCII
   2. 不同的字节序
   3. 引用参数传递：call-by-reference已被copy/restore替换
5. RPC语义？
   1. At-least-once：能够重发请求消息、重新执行程序
   2. At-most-once：能够重发请求消息、重新发送答复
   3. Maybe：都不能确保
6. RPC系统可能的故障和解决方案
   1. 客户端无法找到服务器。
      1. 让这个错误引发异常（例如Java）
      2. 使用信号处理程序（例如，定义信号类型SIG-NOSERVER）
      3. 缺点：并非每种语言都有异常或信号处理程序，会破坏透明性。
   2. 从客户端到服务器的请求消息丢失。
      1. 在client stub上启动计时器，如果计时器在答复或确认返回之前到期，则再次发送消息。
   3. 服务器在收到请求后崩溃
      1. 客户端设置计时器，计时器到期后，其解决方案取决于RPC语义：
         - at least once语义：client stub重新发出请求，服务器重新执行
         - at most once语义：client stub立即放弃并报告失败
         - maybe语义：不做任何保证
   4. 从服务器到客户端的回复消息丢失
      1. client stub设置计时器，计时器到期后
      2. 再次发送请求
         1. 幂等和非幂等请求
         2. 客户为每个请求分配一个序列号
      3. at-least-once语义：重新发送请求，然后重新执行
      4. at-most-once语义：重新发送请求，过滤重复的请求，重新发送回复（如果有）/重新执行
   5. 客户端在发送请求后崩溃。
      1. 孤儿和隔代孤儿：(调用了RPC又去fork了其他进程，调用别的区了)计算处于活动状态，并且没有parent在等待结果。
      1. 灭绝：客户端存根在发送RPC消息之前进行日志输入，并在重新启动后杀死该孤儿（代价高，并且要在client端发送请求傻屌服务器进程，对权限要求也高，也不一定能杀干净）
      1. 轮回：客户端在重新启动后广播消息，因此**服务器代表该客户端**终止所有远程计算
      1. 温和的轮回：客户端广播一条消息，因此服务器尝试找到其所有者并在找不到所有者的情况下将其杀死。(Server找client，找不到就kill了)
      1. 到期：每个RPC都有标准的时间片，如果它无法在指定的时间内完成，则必须明确请求另一个时间片
      1. 实际上，这些方法都不可取
   6. ![wYgtgq](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/wYgtgq.png)
7. RPC扩展：异步RPC模型
   1. 传统RPC中客户端和服务器之间的互连
   2. 异步RPC，在server收到RPC请求后，会立即将答复发送回客户端
   3. ![jT4EO3](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/jT4EO3.png)
   4. 延迟同步RPC：客户端和服务器通过两个异步RPC进行交互
   5. ![SJ9Jwl](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/SJ9Jwl.png)

### Maekawa's voting algorithm

Maekawa投票算法，进程是否能进入临界区的投票算法。

1. 每个进程 $p_i$ 关联一个选举集 $V_i$
   1. $p_i \in V_i$
   2. 任意两个选举集至少一个公共成员
   3. 每个进程的选举集大小相同
   4. 每个进程 $p_j$被包括在M个选举集中
   5. 找到 $V_i$的集合等价于找出一个有限的N点投影平面。
2. ![hhhmRb](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hhhmRb.png)
3. 如何找到 $V_i$的集合？
   1. ![x7wiDQ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/x7wiDQ.png)
4. 容易死锁
5. 满足安全性ME1
6. 优化：如果按照发生在先的顺序对待回答的请求排队，就没有死锁，并且满足ME3
7. 每次进临界区要 $2\sqrt{N}$个消息，每次退出要 $\sqrt{N}$个消息。如果 $N>4$，那么 $3\sqrt{N}$的总值要优于Ricart-Agrawala算法的 $2(N-1)$
8. 同步延迟是一个往返时间，不是单个消息传播时间。

### Impossibility in Asynchronous Systems

异步系统的不可能性

- 在一个异步系统中，即使是只有一个进程出现崩溃故障，也**没有算法能够保证达到共识**
- 在异步系统中，**没有可以确保的方法**来解决拜占庭将军问题、交互一致性问题或者全排序可靠组播问题
- 绕过不可能性结论的三个方法
   - 故障屏蔽
      - 事务系统使用持久储存保存信息
   - 利用故障检测器达到共识
      - 即使是使用不可靠的故障检测器，只要通信是可靠的，崩溃的进程不超过N / 2，那么异步系统中的共 识是可以解决的
   - 随机化进程各方面的行为，使得破坏进程者不能有效地实施他们的阻碍战术

### Checkpointing

属于恢复算法中的后向恢复算法：把系统从当前的错误状态带到之前的正确状态

后向算法的优缺点：
   - 优势：普遍适用的机制
   - 弱点：相对较高的成本，恢复循环，有些状态永远无法回滚到

算法依赖于**检查点的时间和频率**以及**检查点中保存的信息量**，有不同类型的检查点算法。
- 同步的检查点，即协调的检查点，所有进程同步地将其状态写入本地稳定存储。
  - 使用**两阶段锁协议**：**协调者多播checkpoint_request**，将它们正在执行的应用程序传递给它们的任何后续消息使用队列存储，然后将确认发送给协调者，**协调者多播checkpoint_done**
  - 使用非阻塞方法
    – 分布式快照算法（例如Chandy-Lamport的算法）可用于协调检查点
  - **增量式快照算法**
    - 在**两阶段锁协议**中，协调者将**checkpoint_request**多播到那些依赖于协调者恢复的进程，即，协调器仅将检查点请求多播到**它自上次使用检查点以来向其发送消息的那些进程**。
    - 当进程P**接收**到这样的请求时，它将转发给自最后一个检查点以来，**P本身已向其发送消息的所有那些进程**，依此类推。就是下面的p1第二次发给了p2和p4。
    - ![WnsfHX](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/WnsfHX.png)
- 异步的检查点：每个进程独立的检查点。
  - 不协调：完全独立的行为
  - ![nG2eeG](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/nG2eeG.png)
  - 出现一个故障后，failed的进程Q通过恢复其最近的检查点来recover
     - 如果Q在其最近的检查点之后未发出任何消息，则恢复完成
     - 如果Q向进程P发送消息m，则消息m已被P接收，但它之前未被Q发送（在Q恢复其检查点之后）。 这是不一致的。 因此，Q必须向已发送消息的所有进程发送一条消息，并要求它们roll back。
     - 当请求P回滚时，它通过还原到其最新检查点来回滚。
     - 如果P已将消息发送到其他进程，则这些其他受影响的进程也必须回滚。
  - Domino多米诺效应难以避免。

### Stabilizing algorithms

1. 分布式系统的所有可能配置或行为的集合可以分为两类：合法，非法。
  - 非反应系统的合法配置通常由系统整体状态的不变性表示。
  - 在无功系统中，合法配置不仅由状态谓词决定，而且还由行为决定。
2. 行为良好的系统始终处于合法配置，但是由于以下原因，此类系统可能会切换到非法配置：瞬时故障，拓扑更改，环境更改。
3. 当以下两个条件成立时，系统称为稳定化：
  - 收敛：无论初始状态如何，以及无论在每个步骤中选择执行的合格操作如何，系统**最终都会返回到合法配置**。
  - 关闭: 一旦处于合法配置，除非故障或干扰破坏了数据存储器，否则**系统将继续处于合法配置**。

建立一棵生成树的稳定算法
1. 假设故障不会对网络进行分区，而是现在从连接的无向图G =（V，E）（其中| V | = n）构造一个以r为根的生成树，并且每个节点都知道其在树中的级别。
2. 除根r之外的每个节点i都维护两个局部变量：L（i）：i的级别； P（i）：i的父亲。 根节点r具有L（i）= 0，并且没有父变量。
3. 合法状态：如果父指针构成了以根为根的G的生成树，则除根之外的每个节点的级别都等于其父级的级别加1。
4. 如果以下谓词为真，则系统达到合法状态：
   - $$GST \equiv (\forall i,p: i \ne r \land p = P(i) : L(i) = L(p)+1$$
5. ![OzvLpE](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/OzvLpE.png)
6. ![vTuETi](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/vTuETi.png)
7. ![KIVbcG](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/KIVbcG.png)

### Chord

chord characteristics 和弦特征

1. 简单，可证明的正确性和性能
2. 源自一致性哈希算法，用于解决负载平衡load balance、分散化decentralization、可扩展性scalability、availability、灵活的命名flexible naming
3. 一致性哈希需要满足的4个条件：
   1. 均衡性balance：哈希的结果能够尽可能分布到所有的缓冲中
   2. 单调性monotonicity：缓冲区大小变化的时候，应尽量保护已经分配的内容不会被重新映射到新缓冲区
   3. 分散性低：避免由于不同终端所见的缓冲范围有可能不同，从而导致哈希的结果不一致
   4. 负载低：对于一个特定的缓冲区，避免被不同用户映射为不同的内容
4. 整个哈希值空间组成一个虚拟的圆环，在一致性哈希算法中，如果增加一台服务器，则受影响的数据仅仅是新服务器到其环空间中前一台服务器（即沿着逆时针方向行走遇到的第一台服务器）之间的数据，其他数据不受影响
5. 哈希的key被分配到恰好的环中位置，或者是接下来的位置，节点被称为successor node of key k, successor(k)
6. 改进：finger table，每个节点维护一个fingertable，最多m行，第i行包含了succeeds n最少 $2^{i-1}$距离的第一个节点的标识符
   1. ![5eja12](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/5eja12.png)
   2. 节点加入：![we0tPc](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/we0tPc.png)
   3. 节点离开：![6V72yh](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/6V72yh.png)

### Raft

- 目标
  - 分布式共识：即使某些计算机出现故障，也可以使计算机集合作为一个连贯的组工作，提供连续的服务
  - 可理解性：直觉，易于解释
- 技术
  - 问题分解：领导者选举，日志复制（正常操作），安全性（保持日志顺序一致性）
  - 最小化状态空间
    - 通过单一机制处理多个问题
    - 消除特殊情况
    - 最大化连贯性
    - 最小化不确定性
- Raft基础：服务器状态和RPCs
- 看PPT期末考点那个

### Edge-chasing algorithm

- 看PPT期末考点那个
