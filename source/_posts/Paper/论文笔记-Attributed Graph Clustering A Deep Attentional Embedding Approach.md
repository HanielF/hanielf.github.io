---
title: 论文笔记 | Attributed Graph Clustering A Deep Attentional Embedding Approach
tags:
  - GNN
  - DAEGC
  - IJCAI-2019
categories:
  - Papers
comments: true
mathjax: true
date: 2021-05-26 19:34:18
urlname: gnn-daegc-IJCAI-2019
---

<meta name="referrer" content="no-referrer" />

{% note info %}

之前研究了DAEGC模型的源码和论文，补个笔记。

论文《Attributed Graph Clustering: A Deep Attentional Embedding Approach》，模型结果在 Node Clustering on Cora 上，Acc、NMI、ARI排第5， F1排第4。

{% endnote %}

<!--more-->

## Abstract & Introduction

1. Graph clustering
   1. 在网络中挖掘communities和groups
   2. 目标是将节点划分成不想交的group
   3. Attributed graph cluster关键问题是如何捕获结构关系和节点信息
   4. 输入是一个图，输出是 ${G_1, G_2,...,G_k$，同一个cluster的节点可能离得比较近，或者有相似的attribute values
2. 近期大多数工作都是学一个graph embedding，然后用传统的聚类方法，比如k-means或者谱聚类，谱聚类在之前文章[《GNN和图聚类》](https://hanielxx.com/MachineLearning/2021-05-09-gnn-graph-clustering)中有。
3. 之前的工作主要是two-step框架下的，文章认为这种方式不是goal-directed，比如面向一些特殊的clustering任务，所以提出了一个goal-directed的方法，
4. 使用了GAT来捕获邻居特征的重要性，同时encode网络的拓扑结构和节点信息，后面用简单的inner product decoder来重建图信息。用这个GAE生成预测的邻接矩阵A_pred和graph中每个node embedding，作为后面的初始的soft label，用生成的soft label来supervise后面的self-training。
5. 主要是分两个节点，一个是GAE阶段，得到一个初始的soft label，就是一个k-menas的结果，然后用self-training的一个算法进行迭代更新聚类中心
6. 这篇文章主要针对大图的复杂度问题和计算量，在对邻居aggregate的时候是sample

## Proposed Method

### GAE

1. 用的是GAE的变种作为graph autoencoder，目标是学习每个节点的embedding，每个节点的权重是通过将相邻节点的embedding拼接在一起，然后做一个全连接+softmax就得到了。
   1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/dePxtv.png" width="500">
   2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/AKc7H7.png" width="500">
   3. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/ih9MBs.png" width="500">
2. GAT其实在拓扑信息上只考虑了1-hop邻居节点，他们为了获得更强的关系信息，用了t-orer邻居节点信息。上面的权重加入结构信息之后就成了下面这样
   1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/F6QEol.png" width="500">
   2. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/EqGkNT.png" width="500">
3. Decoder只是简单的Inner Product
   1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/LCleco.png" width="500">
4. 这部分的损失其实就是binary_cross_entropy

#### 代码

整体的代码如下，分成了几个部分方便看

##### GAE主函数

```python
import argparse
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam

import utils
from model import GAT
from evaluation import eva

from torch_geometric.datasets import Planetoid

def pretrain(dataset):
    # 这部分只是Graph Attentional Encoder的过程，只算了重构损失
    model = GAT(
        num_features=args.input_dim,
        hidden_size=args.hidden_size,
        embedding_size=args.embedding_size,
        alpha=args.alpha,
    ).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # data and label
    x = torch.Tensor(dataset.x).to(device)
    print(dataset)
    print("GAT:",model)
    print("M.shape:", M.shape)
    y = dataset.y.cpu().numpy()

    for epoch in range(args.max_epoch):
        model.train()
        # model return reconstructed structure matrix A(N, N) and encoded z(N, output_feat)
        A_pred, z = model(x, adj, M)
        loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        with torch.no_grad():
            _, z = model(x, adj, M)
            # n_init: Number of time the k-means algorithm will be run with different centroid seeds. 
            # The final results will be the best output of n_init consecutive runs in terms of inertia.
            kmeans = KMeans(n_clusters=args.n_clusters, n_init=20).fit(
                z.data.cpu().numpy()
            )
            acc, nmi, ari, f1 = eva(y, kmeans.labels_, epoch)
        if epoch % 1 == 0:
            torch.save(
                model.state_dict(), f"./pretrain/predaegc_{args.name}_{epoch}.pkl"
            )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="train", formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument("--name", type=str, default="Citeseer")
    parser.add_argument("--max_epoch", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--n_clusters", default=6, type=int)
    parser.add_argument("--hidden_size", default=256, type=int)
    parser.add_argument("--embedding_size", default=16, type=int)
    parser.add_argument("--weight_decay", type=int, default=5e-3)
    parser.add_argument(
        "--alpha", type=float, default=0.2, help="Alpha for the leaky_relu."
    )
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    datasets = utils.get_dataset(args.name)
    dataset = datasets[0]

    if args.name == "Citeseer":
        args.lr = 0.005
        args.k = None
        args.n_clusters = 6
    elif args.name == "Cora":
        args.lr = 0.005
        args.k = None
        args.n_clusters = 7
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 3
    else:
        args.k = None

    args.input_dim = dataset.num_features

    print(args)
    pretrain(dataset)
```

##### GAT

```python
class GATLayer(nn.Module):
    """
    Simple GAT layer, similar to https://arxiv.org/abs/1710.10903
    """

    def __init__(self, in_features, out_features, alpha=0.2):
        super(GATLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.alpha = alpha

        self.W = nn.Parameter(torch.zeros(size=(in_features, out_features)))
        # 均匀分布初始化输入 Tensor，gain是缩放因子，https://pytorch.apachecn.org/docs/1.0/nn_init.html?h=nn.init.xavier_uniform_
        nn.init.xavier_uniform_(self.W.data, gain=1.414)

        self.a_self = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_self.data, gain=1.414)

        self.a_neighs = nn.Parameter(torch.zeros(size=(out_features, 1)))
        nn.init.xavier_uniform_(self.a_neighs.data, gain=1.414)

        self.leakyrelu = nn.LeakyReLU(self.alpha)

    def forward(self, input, adj, M, concat=True):
        # x: [samples_cnt=N, input_feat]
        # w: [input_feat, output_feat]
        # h: [N, output_feat]
        h = torch.mm(input, self.W) 

        attn_for_self = torch.mm(h, self.a_self)  # (N,1)
        attn_for_neighs = torch.mm(h, self.a_neighs)  # (N,1)
        # >>> a
        # tensor([[1],
        #         [2],
        #         [3]])
        # >>> torch.transpose(a, 0, 1)
        # tensor([[1, 2, 3]])
        # >>> a+torch.transpose(a, 0, 1)
        # tensor([[2, 3, 4],
        #         [3, 4, 5],
        #         [4, 5, 6]])
        attn_dense = attn_for_self + torch.transpose(attn_for_neighs, 0, 1)
        # [N, N]*[N, N]=>[N, N]
        attn_dense = torch.mul(attn_dense, M) 
        attn_dense = self.leakyrelu(attn_dense)  # (N,N)

        zero_vec = -9e15 * torch.ones_like(adj) # [N, N]
        # torch.where: Return a tensor of elements selected from either x or y, depending on condition
        # torch.where(condition, x, y) → Tensor, xi if condition else yi
        adj = torch.where(adj > 0, attn_dense, zero_vec) # [N, N]
        # 对每一行的样本所有邻居softmax
        attention = F.softmax(adj, dim=1) # N, N
        h_prime = torch.matmul(attention, h) # N, output_feat

        if concat:
            # torch.nn.function.elu: Applies element-wise, ELU(x)=max(0,x)+min(0,α∗(exp(x)−1)) .
            return F.elu(h_prime)
        else:
            return h_prime

    def __repr__(self):
        return (
            self.__class__.__name__
            + " ("
            + str(self.in_features)
            + " -> "
            + str(self.out_features)
            + ")"
        )
```

```python
class GAT(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha):
        super(GAT, self).__init__()
        self.hidden_size = hidden_size
        self.embedding_size = embedding_size
        self.alpha = alpha
        self.conv1 = GATLayer(num_features, hidden_size, alpha)
        self.conv2 = GATLayer(hidden_size, embedding_size, alpha)

    def forward(self, x, adj, M):
        h = self.conv1(x, adj, M)
        h = self.conv2(h, adj, M)
        # p是Lp normalize中的p，dim是the dimension to reduce. Default: 1
        # z: [N, output_feat]
        z = F.normalize(h, p=2, dim=1)
        # decoder, A: [N, N]
        A_pred = self.dot_product_decode(z)
        return A_pred, z

    def dot_product_decode(self, Z):
        A_pred = torch.sigmoid(torch.matmul(Z, Z.t()))
        return A_pred

```

##### Evaluate

```python
import numpy as np
from munkres import Munkres

from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from scipy.optimize import linear_sum_assignment as linear

from sklearn import metrics

# similar to https://github.com/karenlatong/AGC-master/blob/master/metrics.py
def cluster_acc(y_true, y_pred):
    # 对y_pred标签重新分配，然后计算acc等指标
    # 可以用匈牙利算法 (Kuhn-Munkres or Hungarian Algorithm) 实现
    y_true = y_true - np.min(y_true)

    l1 = list(set(y_true))
    numclass1 = len(l1)

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    ind = 0
    if numclass1 != numclass2:
        for i in l1:
            if i in l2:
                pass
            else:
                y_pred[ind] = i
                ind += 1

    l2 = list(set(y_pred))
    numclass2 = len(l2)

    if numclass1 != numclass2:
        print("error")
        return

    cost = np.zeros((numclass1, numclass2), dtype=int)
    for i, c1 in enumerate(l1):
        mps = [i1 for i1, e1 in enumerate(y_true) if e1 == c1]
        for j, c2 in enumerate(l2):
            mps_d = [i1 for i1 in mps if y_pred[i1] == c2]
            cost[i][j] = len(mps_d)

    # match two clustering results by Munkres algorithm
    m = Munkres()
    cost = cost.__neg__().tolist()
    indexes = m.compute(cost)

    # get the match results
    new_predict = np.zeros(len(y_pred))
    for i, c in enumerate(l1):
        # correponding label in l2:
        c2 = l2[indexes[i][1]]

        # ai is the index with label==c2 in the pred_label list
        ai = [ind for ind, elm in enumerate(y_pred) if elm == c2]
        new_predict[ai] = c

    acc = metrics.accuracy_score(y_true, new_predict)
    f1_macro = metrics.f1_score(y_true, new_predict, average="macro")
    precision_macro = metrics.precision_score(y_true, new_predict, average="macro")
    recall_macro = metrics.recall_score(y_true, new_predict, average="macro")
    f1_micro = metrics.f1_score(y_true, new_predict, average="micro")
    precision_micro = metrics.precision_score(y_true, new_predict, average="micro")
    recall_micro = metrics.recall_score(y_true, new_predict, average="micro")
    return acc, f1_macro


def eva(y_true, y_pred, epoch=0):
    acc, f1 = cluster_acc(y_true, y_pred)
    nmi = nmi_score(y_true, y_pred, average_method="arithmetic")
    ari = ari_score(y_true, y_pred)
    print(f"epoch {epoch}:acc {acc:.4f}, nmi {nmi:.4f}, ari {ari:.4f}, f1 {f1:.4f}")
    return acc, nmi, ari, f1
```

##### Utils和Dataset

```python
def get_dataset(dataset):
    # Planetoid参考 https://pytorch-geometric.readthedocs.io/en/latest/_modules/torch_geometric/datasets/planetoid.html#Planetoid
    datasets = Planetoid('./dataset', dataset)
    return datasets

def data_preprocessing(dataset):
    # 其实就是用边构建邻接矩阵，参考 https://pytorch.apachecn.org/docs/1.0/torch_tensors.html
    dataset.adj = torch.sparse_coo_tensor(
        dataset.edge_index, torch.ones(dataset.edge_index.shape[1]), torch.Size([dataset.x.shape[0], dataset.x.shape[0]])
    ).to_dense()
    dataset.adj_label = dataset.adj

    # torch.eye: 返回二维张量，对角线上是1，其它地方是0.
    # 给邻接矩阵加上节点到自己的边
    dataset.adj += torch.eye(dataset.x.shape[0]) # (x.shape[0], x.shape[0])
    # 每个元素除以每行的l1范数，即每行元素和，如果是l2就是除以每行样本的l2范数
    # 这里的adj就是论文中的 transition matrix B_{ij}=1/d_i if e_{ij} \in E
    dataset.adj = normalize(dataset.adj, norm="l1")
    dataset.adj = torch.from_numpy(dataset.adj).to(dtype=torch.float)

    return dataset

def get_M(adj):
    adj_numpy = adj.cpu().numpy()
    # t_order
    t=2
    tran_prob = normalize(adj_numpy, norm="l1", axis=0)
    # M就是论文中的proximity matrix M
    M_numpy = sum([np.linalg.matrix_power(tran_prob, i) for i in range(1, t + 1)]) / t
    return torch.Tensor(M_numpy)
```


### Self-optimizing Embedding

1. 这部分刚开始其实一直没怎么看懂，直到看完源码，然后看了之前那篇综述，对整个套路有了大概的了解之后才懂...
2. 主要是学习两个分部，一个P分布一个Q分布
3. 用上面GAE跑出来的node embedding作为初始化，然后跑一次k-means得到初始的簇头，然后在后面的训练过程中不断更新簇头
4. Q分布是通过node embedding和簇头embedding得到的。簇头初始化通过上面的GAE+k-means得到，node embedding在GAE的基础上更新。所以后面的训练的每个epoch，$\mu$和 $z$都会更新。
5. Q分布中的每个值可以衡量每个节点和簇头有多接近，每个epoch中Q分布被认为trustworthy，被当成了soft label，相当于是当前epoch的node embedding和簇头embedding算距离，够近就认为你就是这个簇的，一个假标签。这是模型预测出来的y_pred。
   1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/mWV0oc.png" width="500">
6. 每个节点的soft label可以通过下面的argmax得到
   1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/VVpH3P.png" width="500">
7. 而目标分布P才是真的”ground-truth label"，它是通过近期的Q算出来的，所以也依赖Q。P分布按照阶段更新，被当做是这个阶段内的ground-truth，真标签。
8. 它不能像Q那样每个epoch都更新，不然目标也太不稳定了，Q都不知道朝哪里梯度下降，没法收敛。P代表了在这一个阶段内（论文中写的是5个epoch），Q应该是朝哪里更新，起的作用就是监督学习里的真标签 $Y$。
   1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/TgzYqR.png" width="500">
9. 最后，这部分的损失就是P和Q的KL散度。
   1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/5LTggJ.png" width="500">
10. 总的损失是两个部分的加权和
    1. <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/tESTdL.png" width="500">

#### 代码

这里的evaluate和上面的一样

##### 自监督模块

```python

class DAEGC(nn.Module):
    def __init__(self, num_features, hidden_size, embedding_size, alpha, num_clusters, v=1):
        super(DAEGC, self).__init__()
        self.num_clusters = num_clusters
        self.v = v

        # get pretrain model
        self.gat = GAT(num_features, hidden_size, embedding_size, alpha)
        # 初始化的时候加载pretrain的GAT模型
        self.gat.load_state_dict(torch.load(args.pretrain_path, map_location='cpu'))

        # cluster layer，簇头embedding
        self.cluster_layer = Parameter(torch.Tensor(num_clusters, embedding_size))
        torch.nn.init.xavier_normal_(self.cluster_layer.data)


    def forward(self, x, adj, M):
        # 得到reconstruct的邻接矩阵和[N, feat_size]的节点embedding Z
        A_pred, z = self.gat(x, adj, M)
        q = self.get_Q(z)

        return A_pred, z, q

    def get_Q(self, z):
        q = 1.0 / (1.0 + torch.sum(torch.pow(z.unsqueeze(1) - self.cluster_layer, 2), 2) / self.v)
        q = q.pow((self.v + 1.0) / 2.0)
        q = (q.t() / torch.sum(q, 1)).t()
        return q

def target_distribution(q):
    weight = q**2 / q.sum(0)
    return (weight.t() / weight.sum(1)).t()

def trainer(dataset):
    model = DAEGC(num_features=args.input_dim, hidden_size=args.hidden_size,
                  embedding_size=args.embedding_size, alpha=args.alpha, num_clusters=args.n_clusters).to(device)
    print(model)
    optimizer = Adam(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    # data process
    dataset = utils.data_preprocessing(dataset)
    adj = dataset.adj.to(device)
    adj_label = dataset.adj_label.to(device)
    M = utils.get_M(adj).to(device)

    # data and label
    data = torch.Tensor(dataset.x).to(device)
    y = dataset.y.cpu().numpy()

    with torch.no_grad():
        # 这里的GAT已经load了pretrain的模型
        # 相当于用那个epoch的模型做一次eval
        _, z = model.gat(data, adj, M)

    # get kmeans and pretrain cluster result
    # 这里是用pretrain的结果来初始化kmeans的中心
    kmeans = KMeans(n_clusters=args.n_clusters, n_init=20)
    y_pred = kmeans.fit_predict(z.data.cpu().numpy())
    model.cluster_layer.data = torch.tensor(kmeans.cluster_centers_).to(device)
    eva(y, y_pred, 'pretrain')

    for epoch in range(args.max_epoch):
        model.train()
        if epoch % args.update_interval == 0:
            # update_interval
            A_pred, z, Q = model(data, adj, M)

            q = Q.detach().data.cpu().numpy().argmax(1)  # Q
            eva(y, q, epoch)

        A_pred, z, q = model(data, adj, M)
        p = target_distribution(Q.detach())

        kl_loss = F.kl_div(q.log(), p, reduction='batchmean')
        re_loss = F.binary_cross_entropy(A_pred.view(-1), adj_label.view(-1))

        loss = 10 * kl_loss + re_loss

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

```

##### 调用主函数

```python
import argparse
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import normalize

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.parameter import Parameter
from torch.optim import Adam

from torch_geometric.datasets import Planetoid

import utils
from model import GAT
from evaluation import eva

if __name__ == "__main__":
    # !python3 daegc.py --update_interval 5 --name Cora --epoch 45 --max_epoch 200
    parser = argparse.ArgumentParser(
        description='train',
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--name', type=str, default='Citeseer')
    parser.add_argument('--epoch', type=int, default=30)
    parser.add_argument('--max_epoch', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--n_clusters', default=6, type=int)
    parser.add_argument('--update_interval', default=1, type=int)  # [1,3,5]
    parser.add_argument('--hidden_size', default=256, type=int)
    parser.add_argument('--embedding_size', default=16, type=int)
    parser.add_argument('--weight_decay', type=int, default=5e-3)
    parser.add_argument('--alpha', type=float, default=0.2, help='Alpha for the leaky_relu.')
    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    print("use cuda: {}".format(args.cuda))
    device = torch.device("cuda" if args.cuda else "cpu")

    datasets = utils.get_dataset(args.name)
    dataset = datasets[0]

    if args.name == 'Citeseer':
      args.lr = 0.0001
      args.k = None
      args.n_clusters = 6
    elif args.name == 'Cora':
      args.lr = 0.0001
      args.k = None
      args.n_clusters = 7
    elif args.name == "Pubmed":
        args.lr = 0.001
        args.k = None
        args.n_clusters = 3
    else:
        args.k = None

    args.pretrain_path = f'./pretrain/predaegc_{args.name}_{args.epoch}.pkl'
    args.input_dim = dataset.num_features

```

### 总结

- 仔细一想，确实是自监督的，毕竟soft label和target label都是自己的embedding算出来的，然后再更新自己的embedding。
- 主要是依赖GAE算出来的那个embedding吧，初始化的影响可能比较大...核心还是GAT，Attention 牛批
- 虽然感觉同时更新这两个分布，还用自己算出来的分布来拟合另一个自己算出来的分布，这两个分布还有依赖关系，是有点扯...但是人家的效果还就是好emmmm 玄学
- <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/tlKLqz.png" width="500">

## 实验结果

- 贴一下实验对比的baseline
- <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/8BF7AI.png" width="500">
- 贴一下在Cora，Citeseer和Pubmed上的结果对比，这里其实都是他们自己跑出来的结果，我看和AGC那篇论文，同样的baseline，效果都不一样...
- <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/n02Knc.png" width="500">
- <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/PmeTG5.png" width="500">
- 贴一下分类的效果图，t-SNE可视化的node embedding
- ![figure3](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/JbFPVC.png)
- 提到了超参数的设置，还有不同参数的效果对比，实验做的还是挺多的
- <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/RdR0WV.png" width="500">
- <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/kMzoKK.png" width="500">
- <img src="https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/xlw5Xl.png" width="500">
