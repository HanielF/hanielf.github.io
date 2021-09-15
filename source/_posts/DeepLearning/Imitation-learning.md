---
title: 模仿学习-Imitation-Learning
comments: true
mathjax: true
date: 2021-09-13 13:11:01
tags:
  [
      ImitationLearning,
      ReinforcementLearning,
      DeepLearning,
      Notes
  ]
categories: MachineLearning
urlname: imitation-learning
---

<meta name="referrer" content="no-referrer" />

{% note info %}
模仿学习（Imitation Learning），属于强化学习（Reinforcement Learning）的范畴。
{% endnote %}

<!--more-->

## Imitation Learning

1. 就是让模型在environment中模仿expert的行为
2. 通常的RL是有reward的，但存在很多情况是没有reward这个，或者说很难定义这个reward，比如自动驾驶。这时候就可以用Imitation learning，它通过模仿expert的action来学习。
3. Imitation learning主要有梁总方式
   1. behavior cloning
   2. inverse reinforcement learning

### Behavior Learning

1. 简单来说就是完全去模仿expert的行为，和supervised learning一样
2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Ooulsc.png" width="500">
3. 比如自动驾驶，得到很多training data，那就可以用它们直接训练一个actor，让模型去拟合ground truth
4. 缺点 1：
   1. expert看到的情况是有限的，一般expert只会做正常的action
   2. 比如自动驾驶，expert action中不会有撞墙这个行为，那actor 做出了撞墙这个行为之后，就不知道怎么做了
5. 解决方法：
   1. Dataset Aggregation
   2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/8SGWPS.png" width="500">
   3. 通过behavior cloning创建一个actor 1
   4. 让actor 1 去和环境交互，得到一堆state和action
   5. 然后让expert去对actor 1的observation和state进行标注
   6. 把标注的数据加入到dataset中再次训练一个actor 2，重复下去
6. 缺点 2：
   1. model会完全模仿expert，可能会学习到一些没用的信息
   2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/000qeo.png" width="500">
   3. 比如老师教学生的时候，会带上一些手上动作，behavior cloning会完全模仿expert，会学到手上动作这个没用的信息
7. 缺点 3:
   1. 存在test set和train set数据分布不一样的情况，因为下一个action会受到前面的state的影响，如果有一点偏差就会导致后面分布的不同
   2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/C4khTi.png" width="500">
   3. 模型actor和expert如果不存在误差，那测试集和训练集分布应该是一样的
   4. 如果存在一点误差，那训练集和测试集的分布就不一样了

### Inverse Reinforcement Learning

1. 这时候没有reward，但是还是可以和environment交互，也有expert
2. 这时候的reward只能从expert中推导出是什么样的reward
3. 因此可以用expert数据来学习出一个reward function，有了这个reward function就可以像普通的RL一样进行
4. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/mRRcuM.png" width="500">
5. 这里训练reward function的原则就是，expert 就是最好的，每次都要让expert reward大于actor reward
6. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/rhx7rU.png" width="500">
7. 有了新的reward function之后就可以训练新的actor，然后再得到新的reward functional
8. 这个流程和GAN很像，其实只是名字换了，任务换了
9. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ZuygGy.png" width="500">
10. Inverse Reinforcement Learning通常只需要**少量的数据**，因为它只是在做一个demonstration

### Third Person Imitation Learning

1. 之前的actor都是和expert的视角一样，都是第一人称视角，未来能否让agent看着我们就可以学会，这时候就设计到视角的转换，从第一人称变成第三人称
2. 其实和Imitation learning是一样的，只要把最重要的东西抽取出来就可以
3. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/8YujmQ.png" width="500">