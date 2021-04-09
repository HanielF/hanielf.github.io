---
title: GBM和GridSearch
tags:
  - GBM
  - GridSearch
  - GBDT
  - K-Fold
  - ML
categories:
  - MachineLearning
comments: true
mathjax: true
date: 2021-01-12 11:17:25
urlname: gbm-gridsearch
---

<meta name="referrer" content="no-referrer" />

{% note info %}

sklearn中`sklearn.ensemble.GradientBoostingRegressor`和`sklearn.model_selection.GridSearchCV`的使用

{% endnote %}
<!--more-->

## GBDT

sklearn中实现了[sklearn.ensemble.GradientBoostingRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor)和[sklearn.ensemble.GradientBoostingClassifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.GradientBoostingRegressor.html#sklearn.ensemble.GradientBoostingRegressor)，分别用于回归和分类，具体的参数看官网说明。

### 原理

[GDBT梯度提升决策树](https://hanielxx.com/Notes/2020-12-08-ucas-big-data-analysis-review.html#Gradient-Boost-Decision-Tree-GBDT)可以参考以前的笔记。

提一下Adaboost，[wikipedia](https://en.wikipedia.org/wiki/AdaBoost)讲的比较清楚，也可以看我之前的笔记：[UCAS模式识别-Adaboost](https://hanielxx.com/Notes/2020-12-14-ucas-prml-review.html#AdaBoost)

## GridSearchCV

参考链接：
- [GBM调参汇总](https://zhuanlan.zhihu.com/p/130747955)
- [刘建平-scikit-learn 梯度提升树(GBDT)调参小结](https://www.cnblogs.com/pinard/p/6143927.html)
- [Python机器学习笔记：Grid SearchCV（网格搜索)](https://www.cnblogs.com/wj-1314/p/10422159.html)

GBDT搜索参数代码：

```python
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.utils import shuffle
from sklearn import metrics
from sklearn.model_selection import train_test_split
from sklearn.model_selection import KFold
from sklearn.model_selection import GridSearchCV

def gbdt_regression_gridsearchcv(x_train, y_train):
    '''使用网格搜索GBDT参数
    '''
    # 固定learning rate=0.1, subsample=0.8, min_samples_split=1%*N=24, min_samples_leaf=default=1, max_depth=8, max_features='sqrt', random_state=SEED，得到结果n_estimaors=90最佳
    param_test1 = {
        'n_estimators': range(20, 250, 10),
    }
    # 固定其他，调整max_depth和min_sample_split，得到结果为max_depth=5, min_samples_split=44
    param_test2 = {
        'max_depth': range(3, 16, 2),
        'min_samples_split': range(12, 48, 4),  # 观测样本的0.5%-2%
    }
    # 调整min_samples_split和min_samples_leaf，得到结果min_samples_split=44,min_samples_leaf=7
    param_test3 = {
        'min_samples_split': range(12, 48, 4),
        'min_samples_leaf': range(1, 11),
    }

    gsearch1 = GridSearchCV(
        estimator=GradientBoostingRegressor(learning_rate=0.1,
                                            n_estimators=90,
                                            subsample=0.8,
                                            max_depth=5,
                                            min_samples_split=44,
                                            min_samples_leaf=7,
                                            max_features='sqrt',
                                            random_state=SEED),
        param_grid=param_test3,  # 修改要搜索的参数
        scoring='neg_mean_squared_error',
        cv=5)
    gsearch1.fit(x_train, y_train)
    print(gsearch1.scorer_, gsearch1.best_params_, gsearch1.best_score_)
```

## k-fold交叉验证

### sklearn代码

```python
def kfold_regression(model, x_train, y_train, x_test, y_test, random_seed, K=5):
    '''使用k折交叉验证对model进行训练，得到最后的平均pearsonr相关系数
    '''
    # k折交叉验证，这里要记得shuffle=true，在分之前随机打乱
    kf = KFold(n_splits=K, shuffle=True)

    avg_r = 0
    for train_index, test_index in kf.split(y_data):
        x_train, x_test = x_data.iloc[train_index, :], x_data.iloc[test_index, :]
        y_train, y_test = y_data[train_index], y_data[test_index]

        # 下面的参数是通过girdsearch得到
        model.fit(x_train, y_train)
        y_predict = model.predict(x_test)

        r, _ = scipy.stats.pearsonr(y_predict, y_test)
        avg_r += r
        print("current r = {}".format(r))
        # print("current split r = {}, oob_score = {}".format(r, model.oob_score_))
    avg_r /= K
    print("average pcc: {}".format(avg_r))
    return avg_r

gbdt_model = GradientBoostingRegressor(learning_rate=0.1,
                                        n_estimators=90,
                                        subsample=0.8,
                                        max_depth=5,
                                        min_samples_split=44,
                                        min_samples_leaf=7,
                                        max_features='sqrt',
                                        random_state=SEED)

kfold_regression(gbdt_model, x_train, y_train, x_test, y_test, SEED, K)
```

-----