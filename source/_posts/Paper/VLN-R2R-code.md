---
title: VLN-R2R任务源码理解
mathjax: false
date: 2021-09-02 20:11:01
tags:
  [
    VLN,
    R2R,
    Multimodal,
    DeepLearning,
    CVPR,
  ]
categories: Papers
urlname: vln-r2r-seq2seq-code
---

<meta name="referrer" content="no-referrer" />

{% note info %}
论文《[Vision-and-Language Navigation: Interpreting visually-grounded navigation instructions in real environments](https://arxiv.org/abs/1711.07280)》是VLN的开篇之作，这里记录下对它在R2R任务里面的源码理解。

github仓库地址：[VLN-R2R](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/tasks/R2R)
{% endnote %}

<!--more-->

## 模型训练

1. 程序入口：`train.py/train_val()`
2. vocab的建立和保存
3. 创建tokenizer
4. 创建训练环境train\_env，是一个R2RBatch类
   1. **R2RBatch类**初始化init
      1. 导入feature和image的信息并建立batch个Simulator()
      2. 加载数据集，保存scans信息，记录instruction信息，并通过vocab对instruction进行encode
      3. load\_nav\_graphs，加载每个scan的链接图信息
      4. all\_pairs\_dijkstra_path，计算所有的最短路径
5. 创建val_envs，即验证环境，分为seen和unseen
   1. 这里使用字典保存`val_seen: (R2RBatch(), Evaluation()), val_unseen: (R2RBatch(), Evaluation())`
      1. **Evaluation类**
         1. `_get_nearest()`是从path中找到距离goal id最近的一个节点id
         2. `_score_item()`是计算最终结果和目标点的
            1. nav error即最终点id和goal之间的距离
            2. oracle error即和目标id最接近的id，它和goal之间的距离
            3. trajectory steps即path的step个数减1
            4. trajectory lengths即path的总距离
         3. `score()`是通过和目标点的距离，验证每个agent 的轨迹
            1. 通过`score_item()`得到每个path的分数
            2. 返回每个路径的平均分数
            3. 以及两个成功率：nav error小于目标值的概率，oracle error小于目标值的概率
6. 创建模型，encoder和decoder
   1. **encoder**是EncoderLSTM类，对navigation instruction进行embedding，并用lstm进行encode，返回hidden state、用于decoder初始化的一个state、以及cell state
      1. hidden和cell state初始化都是0
      2. forward过程是embeedding->dropout->init->pack and pad->lstm->得到h\_t和c\_t->linear层处理hidden state再加上tanh得到decoder init state-> pad and packed sequence-> dropout-> return ctx，decoder init，cell state
   2. **decoder**是AttnDecoderLSTM类
      1. forward过程是：对action进行embedding->concat action embedding和feature -> dropout -> lstm -> dropout -> attention layer得到经过dot attention得到的h\_tilde和attention权重 -> h\_tilde通过linear得到logit
7. 训练过程train()
   1. agent用**Seq2SeqAgent**，基于seq2seq和attention和LSTM的agent
      1. 用三维元组表示每个方向
      2. feedback可选teacher，argmax，sample
      3. 初始化的encoder和decoder就是之前创建好的
   2. 迭代过程
      1. **Seq2SeqAgent\.train()**
         1. encoder\.train()
         2. decoder\.train()
         3. n\_iter里面
            1. optimizer梯度置0
            2. **rollout()**
               1. self\.env\.reset()，加载一个**新的mini batch数据**
               2. 把输入，按照每个observation中的**instructions**的长度降序排序，方便padding
               3. 记录开始的observation信息
               4. **encoder**得到context state和hidden state cell state
               5. **初始化**start action和ended 标识，都是batch个
               6. 用encoder得到的context state, hidden state, cell state和每个observation里面的feature，输入到**decoder**中得到输出的hidden state，cell state, attention权重，和logit
               7. 对无法forward的部分进行**mask**，即把logit[idx, index of forward action]置为负无穷
               8. \_teacher\_action()，提取ground truth的agent的方向信息，保存在**target**中
               9. 用decoder得到的logit和target计算**交叉熵损失**
               10. 根据feedback策略获得a\_t变量，即**action target**
                   1. teacher force策略，action target是ground truth
                   2. student force策略，action target是logit的argmax
                   3. sample策略，是按照概率对logit结果采样
               11. **更新结束标志**，如果结束了，后面agent就不用再继续了
               12. 对所有的observation进行遍历，如果没有end就**更新traj路径**，即`traj[i]['path'].append((ob['viewpoint'], ob['heading'], ob['elevation']))`
               13. 如果所有的agent都end了，就不用到下一个场景了，否则就继续**下一个场景**
               14. 最后所有场景迭代完，保存每个场景的平均损失，并且返回agent的**轨迹traj**
            3. loss反向传播
            4. optimizer\.step()
      2. 记录loss等
      3. 进行**validation**
         1. agent\.test(use\_dropout=True)，保持和训练时的环境一样，即encoder,decoder都train()，再进行test()
            1. 这里的test调用了BaseAgent的test()
            2. **reset_epoch()**，重置self\.id为1，即data index变成了epoch开始那会的index
            3. 一个looped标志，记录测试集是否跑完一遍
               1. 只有在出现相同的instr\_id时，才回退出循环，而想要出现相同的instr\_id，就得遍历完一遍测试集
               2. 因为rollout函数里每次都会进行一个`self.env.reset()`，这个地方会进行`_next_minibatch()`
               3. `_next_minibatch()`会在剩下的data不足一个batch时，shuffle所有的data，然后继续采样
         2. agent\.test(use\_dropout=False)，encoder和decoder都eval()，再test()
         3. 记录loss和metric
   3. `agent.env=train_env`
   4. 记录log，保存checkpoint

## 模型验证

1. 程序入口：`eval.py/eval_simple_agents()`
2. 对每个split进行验证，train, val\_seen, val\_unseen, test
   1. 创建环境 `env=R2RBatch()`，细节见[模型训练](#模型训练)
   2. 创建evaluator，`ev=Evaluation()`，细节见[模型训练](#模型训练)
   3. 遍历不同类型的agent，StopAgent，ShortestAgent，RandomAgent
      1. 创建上面指定的agent
         1. **StopAgent**是在原地不动的，用于测试数据集是否有效，就是看agent不动能不能满足要求
         2. **RandomAgent**是随机选择了一个方向，然后尝试直走5个viewpoint然后停止
         3. **Shortest**是算的最短路径
      2. agent\.test()
      3. agent\.write\_results()，保存结果
      4. ev\.score()，计算和目标点的距离，验证每个agent 的轨迹

## 可视化

略