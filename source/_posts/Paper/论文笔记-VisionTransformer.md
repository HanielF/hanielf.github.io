---
title: 论文笔记 | VisionTransformer综述笔记
tags:
  - VisionTransformer
  - Transformer
  - MachineLearning
  - Paper
categories:
  - Papers
comments: true
mathjax: true
date: 2021-10-18 10:56:51
urlname: vision-transformer-survey
---

<meta name="referrer" content="no-referrer" />

{% note info %}
《Transformers in Vision: A Survey》部分笔记
{% endnote %}
<!--more-->

## SELF-ATTENTION & T RANSFORMERS IN VISION

### Single-head Self-Attention

#### Self-Attention in CNNs

#### Self-Attention as Stand-alone Primitive

### Multi-head Self-Attention (Transformers)

#### Uniform-scale Vision Transformers

vit数属于这一类，就是输入的时候用MHA，后面的stage就维持空间尺度不变。不同的stage串联起来就像是一个柱子一样...

##### IPT: Pre-trained image processing transformer

 1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/QnA2Ak.png" width="600">
 2. 用各种head提取特征，然后用transformer的encoder-decoder结构进行编码和对图片的重建，最后接多个head进行图片的生成重建。
 3. 切分成patch，位置编码用learnable，结构和transformer的encoder-decoder一样
 4. 应用到的是全尺寸的image上做预训练，然后用到下游的denoise, derain,等等，都是生成任务

##### DeiT: Training data-efﬁcient image transformers & distillation through attention

1. 第一个证明了Transformer可以用在中等大小数据集上的，一百二十万的imagenet，对比的是ViT的JFT数据集，3亿
2. 用蒸馏的方式来学习Transformer，有一个teacher模型，还有一个student模型。目标是让student模型从teacher模型中学习到相同的知识。
3. 这里的teacher模型用的是CNN结构的RegNetY-16GF，imagenet top1 acc=82.9。student模型就是纯的transformer。
4. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ZocEwW.png" width="500">
5. 有两种蒸馏方式
   1. soft distillation: 最小化teacher模型和student模型的softmax结果的KL散度。<img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/nPPxxQ.png" width="500">
   2. hard distilation: 就是默认teacher模型的结果就是ground truth，让student模型去拟合那个label，就是用交叉熵。<img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/YA24kS.png" width="500">

##### T2T: Tokens-to-Token ViT: Training Vision Transformers from Scratch on ImageNet

1. hypothesize that such performance gap roots in two main limitations of ViT: 1) the straightforward tokenization of input images by hard split makes ViT unable to model the image local structure like edges and lines, and thus it requires signiﬁcantly more training samples (like JFT-300M for pretraining) than CNNs for achieving similar performance; 2) the attention backbone of ViT is not welldesigned as CNNs for vision tasks, which contains redundancy and leads to limited feature richness and difﬁculties in model training.
2. We are then motivated to design a new full-transformer vision model to overcome above limitations.
   1. propose a progressive tokenization module to aggregate neighboring Tokens to one Token (named Tokens-to-Token module),which can model the local structure information of surrounding tokens and reduce the length of tokens iteratively.
   2. in each Token-to-Token (T2T) step, the tokens output by a transformer layer are reconstructed as an image (restructurization) which is then split into tokens with overlapping (soft split) and ﬁnally the surrounding tokens are aggregated together by ﬂattening the split patches.
   3. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hDCjN7.png" width="500">

#### Multi-scale Vision Transformers

##### DETR: End-to-End Object Detection with Transformers

1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/lOoAXc.png" width="500">

##### CoAT: Co-Scale Conv-Attentional Image Transformers

1. 用多尺度的transformer，并提出了conv-attention，就是用 depthwise convolution 做relative position embedding，这个被用在一种分解的attention module里面。
2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/BwN5wv.png" width="500">
3. 后面看着好复杂。。直接贴个博客链接吧[CoAT](https://cloud.tencent.com/developer/article/1816902)

##### Swin-Transformer: Swin transformer: Hierarchical vision transformer using shifted windows

1. 太出名了，ICCV马尔奖，就不多记了
2. 用shift window实现了全局和局部的注意力结合,

##### Focal Trans: Focal Self-attention for Local-Global Interactions in Vision Transformers

1. 就是先patch，然后让每个window作为中心，在它周围的window叫focal region $s_r$，然后focal region中的每个位置是由 $s_w$大小的subwindow进行pooling得到的。$s_p=4$是window partition的大小，中间的焦点也就是一个window大小。所以最后得到的大小是 $s_r x s_r$， 对应原本的大小是 $s_r*s_w$，然后再去掉中间的 $s_w*s_w$大小，就两边分别为 $(s_r*s_w - s_w)/2$。
2. 它是通过中间window计算query，然后对周围sub window pooling得到的区域计算key和value。进行window-wise的multihead attention
3. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/JLLHSc.png" width="500">
4. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/QnI87r.png" width="500">

#### Hybrid ViTs with Convolutions

##### CvT: CvT: Introducing Convolutions to Vision Transformers

1. 就是对token使用2d卷积来编码，就是论文里的convolutional token embedding，然后在做attention操作的时候，把计算qkv的linear层替换成空间可分离深度卷积（depth-wise separable convolution）,就是 Convolutional projection.
2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/hdEAyx.png" width="500">
3. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/Iyekbi.png" width="500">
4. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/eiEQ2E.png" width="500">

##### CoaT：Co-Scale Conv-Attentional Image Transformers

1. 就是用conv做一个relative position embedding，然后提出了两种Block的形式，一种是Serial Block，就是平常用的，另一种是Parallel Block，多个Block之间是并行的，然后在多个block之间进行cross layer attention。
2. Parallel Block有两种实现方式，一种是Direct cross-layer attention，另一种是Attention with feature interpolation。
3. Direct cross-layer attention中，不同尺度的特征都是从输入得到的，然后对相同的layer，用conv和MHA。对不同的层，因为尺度不一样，所以对key和value进行上采样或者下采样的方式进行尺度的统一。用当前层的query和其他层的key，value。最后对当前层的conv attention和不同层cross-layer attention进行求和。
4. Attention with feature interpolation中，没有直接跨层注意力，而是对输入用独立的conv attention得到不同尺度的特征，然后用上采样和下采样让不同尺度之间进行统一。通尺度的特征在parallel group里面直接求和，然后用FFN。然后再用conv-attention module对现在这个feature interpolation做跨尺度的融合。
5. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/tLAIWk.png" width="500">
6. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/qIqIq9.png" width="500">
7. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/P6VBTu.png" width="500">
8. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/nsW1Xi.png" width="500">

##### Twins: Twins: Revisiting the Design of Spatial Attention in Vision Transformers

1. 做了两个工作，一个是提出了Twins-PCPVT，其实就是结合了PVT和CAPT，把PVT中的绝对位置编码换成了CAPT中的动态位置编码。位置编码的位置在每个stage的第一个encoder block后面。所以结构和PVT基本一样。
2. 另一个工作才是重点。提出了Twins-SVT
   1. 借鉴了空间可分离深度卷积的思想，就是在transformer的block里面，先用类似deep-wise的分组，做一个local grouped attention(LSA)，然后再类似point-wise的cnn，做一个global sub-sampled attention(GSA)。总结就是先在部分空间做MHA，然后再做一个全局的MHA。
   2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/exhvXj.png" width="500">
   3. GSA是在sub-window里面用卷积的方式弄一个代表key，然后用这些key来做全局的MHA
   4. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/JfLpdV.png" width="500">


#### Shuflle Transformer

#### CrossFormer: A Versatile Visiont Ransformer Hinging on Cross-Scale Attention Versatile

1. 做了两个事，在embed那边做了多尺度的embedding，然后降低了计算量
2. 提出了CEL：Cross-scale Embedding Layer，以及LSDA：Long Short Distance Attention
3. CEL：在上一个stage的输出基础上，用不同大小的kernel得到不同尺度的embedding，然后拼接或者projection得到patch embedding
4. LSDA：将attention分为SDA和LDA，就是分别对近距离的和远距离的attention
5. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/nSuAm5.png" width="500">
6. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/1bIawh.png" width="500">
7. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/mrGW9m.png" width="500">