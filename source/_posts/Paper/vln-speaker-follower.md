# 2021-9-12 论文分享记录

## speaker-follower

1. speaker相当于是为了生成更多的Instructions，任意采样两个点，生成路径，用这个路径生成Instruction
2. follower相当于是用来router
3. 创新点
   1. speaker 驱动的数据增强，后面follower训练的时候就先在增强的数据上面训练，然后在原始数据上进行微调
   2. 挑选出一堆路径，没有直接找概率最大的，而是计算了每个候选路径生成当前Instruction的概率，选择计算后得分最高的，
   3. viewpoint处的全局感知，一共有36张图的全景图信息，把低层次的turn left变成了turn 多少度数
4. speaker
   1. encoder
      1. action embedding和方向embedding都投影到256维度
      2. 让action embedding和每个方向的embedding（36个），做一个点乘，得到36个权重，
      3. 然后对原始的36个embedding进行加权和
      4. dropout后送到lstm中
   2. decoder
      1. 用lstm做action预测
5. follower
   1. encoder
      1. lstm
   2. decoder
      1. lstm