---
title: AllenNLP框架学习笔记（一）
mathjax: false
date: 2021-04-14 10:11:01
tags:
  [
    AllenNLP,
    NLP,
    OpenLibrary,
    DeepLearning,
    Pytorch,
  ]
categories: MachineLearning
urlname: allennlp-notes
---

<meta name="referrer" content="no-referrer" />

{% note info %}
刚开始学习使用AllenNLP框架，记录一些笔记供参考。

感觉这个框架用起来比较优雅，从基类的抽象，到模块间的配合。但缺点是官方的guide又臭又长，讲的也不是很清楚感觉，很多东西还是得边看源码边写。这个笔记是看官方guide记录，只能做到脑子里大概有个框架，要熟悉AllenNLP还是得看源码。
{% endnote %}

<!--more-->

## Github-README

- 官方仓库：<https://github.com/allenai/allennlp>
- 当前版本：`v2.2.0`
- Guide: [AllenNLP Guide](https://guide.allennlp.org/)
- 需要train-config files的配置，参考 此[模板](https://github.com/allenai/allennlp-template-config-files)
- 直接code配置并且自己train，参考 此[模板](https://github.com/allenai/allennlp-template-python-script)
- allennlp的插件是动态加载的，会自动识别官方维护的包，其他的需要写一个本地插件配置文件`.allennlp_plugins`，存放在运行allennlp命令的目录下，或者全局配置`~/.allennlp/plugins`。一行一个。
- 用`allennlp test-install`测试是否存在插件

- Install Requirement：
  - python 3.6.1+
  - Pytorch
- 推荐用pip安装：`pip install allennlp`
- 如果python版本高于3.7，要注意不能安装pypi的dataclasses，使用`pip freeze | grep dataclasses`，如果有的话直接`pip uninstall -y dataclasses`
- `allennlp --help`查看命令

## 大体框架

### 定义输入输出

 1. 用`Instance`表示一个example对象
 2. 每个`Instance`对象包含几个`Fields`，这些`Fields`后面会被转换为tensor
 3. 对于文本分类，输入为`text: TextField`，输出为`label: LabelField`，都是类对象

### 读数据

1. 使用`DatasetReader`类
2. 会将原始数据转换为`Instances`
3. 上面的文本分类，对应的dataset文件，每行一句话，后面是label，中间用`[TAB]`分隔
4. 数据样例：`I like this movie a lot! [TAB] positive`
5. 可以通过继承`DatasetReader`类实现自己的reader，但是**必须重写`_read()`方法**
6. 对于prediction，label不用写，代码也是一样的
7. 代码样例如下
   1. 在调用`reader.read(file)`时，
   2. 会读入input file，split每一行text，
   3. 用tokenizer对text分词，得到vacab中对应的id。
   4. 要注意的是，`fields`中的keys，后面会作为参数名传入`Model`中。

 ```python
 @DatasetReader.register('classification-tsv')
 class ClassificationTsvReader(DatasetReader):
 def __init__(self):
     self.tokenizer = SpacyTokenizer()
     self.token_indexers = {'tokens': SingleIdTokenIndexer()}

 def _read(self, file_path: str) -> Iterable[Instance]:
     with open(file_path, 'r') as lines:
         for line in lines:
             text, label = line.strip().split('\t')
             text_field = TextField(self.tokenizer.tokenize(text),
                                     self.token_indexers)
             label_field = LabelField(label)
             fields = {'text': text_field, 'label': label_field}
             yield Instance(fields)
 ```

### 创建模型

1. ALLenNLP的模型是Pytorch中的`Module`，实现`forward()`方法，**要求输出是dictionary**，输出中会包含一个`loss` key，用于后面的optimization
2. batch instances->`Model.forward()`->get `loss`->backprop phase->update parameters，这个training loop在框架中已经实现了，只有在需要自定义的时候再改
3. 建议在model的`__init__`方法中，将所有的参数都作为构造器参数，可以方便更改模型而不需要改代码。
4. 建议在参数后面写注释说明，方便理解，而且会有一些神奇的用法，比如在构造模型的时候，会读取配置文件，自动创建对应的对象。通过装饰器实现的。就像`ClassyVision`中一样。
5. 模型样例如下，
   1. 通过vocab传vocabulary，
   2. `TextFieldEmbedder`会将`TextField`创建的tensors编码，输出`(batch_size, num_tokens, embedding_dim)`的embedding，
   3. 通过`Seq2VecEncoder`将一个序列的tensor转换为一个tensor，即`(batch_size, encoding_dim)`。
   4. 最后的分类器是一个fc层，输入的shape可以通过encoder的`get_output_dim()`获得，输出的shape就是label的个数，通过`vocab.get_vocab_size("labels")`得到。

 ```python
 @Model.register('simple_classifier')
 class SimpleClassifier(Model):
     def __init__(self,
                 vocab: Vocabulary,
                 embedder: TextFieldEmbedder,
                 encoder: Seq2VecEncoder):
         super().__init__(vocab)
         self.embedder = embedder
         self.encoder = encoder
         num_labels = vocab.get_vocab_size("labels")
         self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)
 ```

### 前向传播过程

 ```python
 class SimpleClassifier(Model):
     def forward(self,
                 text: TextFieldTensors,
                 label: torch.Tensor) -> Dict[str, torch.Tensor]:
         # Shape: (batch_size, num_tokens, embedding_dim)
         embedded_text = self.embedder(text)
         # Shape: (batch_size, num_tokens)
         mask = util.get_text_field_mask(text)
         # Shape: (batch_size, encoding_dim)
         encoded_text = self.encoder(embedded_text, mask)
         # Shape: (batch_size, num_labels)
         logits = self.classifier(encoded_text)
         # Shape: (batch_size, num_labels)
         probs = torch.nn.functional.softmax(logits)
         # Shape: (1,)
         loss = torch.nn.functional.cross_entropy(logits, label)
         return {'loss': loss, 'probs': probs}
 ```

 1. DatasetReader中使用的字段名称，这里是`text`和`label`，DatasetReader会生成batch具有相同字段名称的Instances，所以这里要使**用同样的方式命名参数**
 2. 要注意输入输出都是`batch torch.Tensor`
 3. 流程比较简单，embed->mask padding->encoder->classifier->softmax->loss
 4. 最后的fc层输出一个分数，一般叫`logit`，需要用softmax将它转换为在label上的概率分布

## 训练和预测

这里接上面的分类模型，讲的是怎么训练模型和做prediction。

有两种方式实现：
1. 自己写代码来构建dataset reader和model，然后进行training loop
2. 通过配置文件，使用`allennlp train`命令训练

### 用自己的代码训练

大多数情况下可以直接使用`allennlp train`来训练，也可以自定义代码。

下面使用的dataset是：[Movie Review Data](http://www.cs.cornell.edu/people/pabo/movie-review-data/)，是用户在IMDb上的评论，label是positive和negative二分类。

#### 测试dataset reader

在使用自己的dataset reader之前，建议写一写测试脚本来验证一下代码是否正确。

```python
from typing import Dict, Iterable, List

from allennlp.data import DatasetReader, Instance
from allennlp.data.fields import Field, LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer

class ClassificationTsvReader(DatasetReader):
    def __init__(
        self,
        tokenizer: Tokenizer = None,
        token_indexers: Dict[str, TokenIndexer] = None,
        max_tokens: int = None,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {"tokens": SingleIdTokenIndexer()}
        self.max_tokens = max_tokens

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, "r") as lines:
            for line in lines:
                text, sentiment = line.strip().split("\t")
                tokens = self.tokenizer.tokenize(text)
                if self.max_tokens:
                    tokens = tokens[: self.max_tokens]
                text_field = TextField(tokens, self.token_indexers)
                label_field = LabelField(sentiment)
                fields: Dict[str, Field] = {"text": text_field, "label": label_field}
                yield Instance(fields)


dataset_reader = ClassificationTsvReader(max_tokens=64)
instances = list(dataset_reader.read("quick_start/data/movie_review/train.tsv"))

for instance in instances[:10]:
    print(instance)
```

#### 给模型输入Instances

训练过程会被分为几个简单的function来分别实例化一写依赖。模型需要从数据中建立`Vocabulary`，创建的过程作为一个单独的function。为了不把模型创建的过程放在训练流程中，模型的创建也会作为一个单独的function。

```python
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.nn import util
from allennlp.training.metrics import CategoricalAccuracy

class SimpleClassifier(Model):
    def __init__(
        self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder
    ):
        super().__init__(vocab)
        self.embedder = embedder
        self.encoder = encoder
        num_labels = vocab.get_vocab_size("labels")
        self.classifier = torch.nn.Linear(encoder.get_output_dim(), num_labels)

    def forward(
        self, text: TextFieldTensors, label: torch.Tensor
    ) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        # Shape: (1,)
        loss = torch.nn.functional.cross_entropy(logits, label)
        return {"loss": loss, "probs": probs}


def run_training_loop():
    dataset_reader = ClassificationTsvReader(max_tokens=64)
    print("Reading data")
    instances = list(dataset_reader.read("quick_start/data/movie_review/train.tsv"))

    vocab = build_vocab(instances)
    model = build_model(vocab)

    outputs = model.forward_on_instances(instances[:4])
    print(outputs)


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    print("Building the vocabulary")
    return Vocabulary.from_instances(instances)


def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=10, num_embeddings=vocab_size)}
    )
    encoder = BagOfEmbeddingsEncoder(embedding_dim=10)
    return SimpleClassifier(vocab, embedder, encoder)


run_training_loop()
```

#### 训练模型

AllenNLP使用`Trainer`来训练模型。Trainer会负责关联你的model、optimizer、instances、dataloder，执行training loop等。所有的function几乎都是`build_`格式。

##### 依赖setup

- 中间实现和上面的一样，所以省略了一些代码

```python
import tempfile
from typing import Dict, Iterable, List, Tuple

import allennlp
import torch
from allennlp.data import (
    DataLoader,
    DatasetReader,
    Instance,
    Vocabulary,
    TextFieldTensors,
)
from allennlp.data.data_loaders import SimpleDataLoader
from allennlp.data.fields import LabelField, TextField
from allennlp.data.token_indexers import TokenIndexer, SingleIdTokenIndexer
from allennlp.data.tokenizers import Token, Tokenizer, WhitespaceTokenizer
from allennlp.models import Model
from allennlp.modules import TextFieldEmbedder, Seq2VecEncoder
from allennlp.modules.seq2vec_encoders import BagOfEmbeddingsEncoder
from allennlp.modules.token_embedders import Embedding
from allennlp.modules.text_field_embedders import BasicTextFieldEmbedder
from allennlp.nn import util
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer
from allennlp.training.metrics import CategoricalAccuracy

class ClassificationTsvReader(DatasetReader):
    def __init__(
    self,
    tokenizer: Tokenizer = None,
    token_indexers: Dict[str, TokenIndexer] = None,
    max_tokens: int = None,
    **kwargs
    ):

    def _read(self, file_path: str) -> Iterable[Instance]:

class SimpleClassifier(Model):
    def __init__(
        self, vocab: Vocabulary, embedder: TextFieldEmbedder, encoder: Seq2VecEncoder
    ):

    def forward(
        self, text: TextFieldTensors, label: torch.Tensor
    ) -> Dict[str, torch.Tensor]:

def build_dataset_reader() -> DatasetReader:
    # ....
    return ClassificationTsvReader()

def read_data(reader: DatasetReader) -> Tuple[List[Instance], List[Instance]]:
    print("Reading data")
    training_data = list(reader.read("quick_start/data/movie_review/train.tsv"))
    validation_data = list(reader.read("quick_start/data/movie_review/dev.tsv"))
    return training_data, validation_data


def build_vocab(instances: Iterable[Instance]) -> Vocabulary:
    # ...
    return Vocabulary.from_instances(instances)


def build_model(vocab: Vocabulary) -> Model:
    # ...
    return SimpleClassifier(vocab, embedder, encoder)
```

##### run_training_loop和Trainer构建

```python
from allennlp.training.trainer import GradientDescentTrainer, Trainer
from allennlp.training.optimizers import AdamOptimizer

def run_training_loop():
    dataset_reader = build_dataset_reader()

    train_data, dev_data = read_data(dataset_reader)

    vocab = build_vocab(train_data + dev_data)
    model = build_model(vocab)

    train_loader, dev_loader = build_data_loaders(train_data, dev_data)
    train_loader.index_with(vocab)
    dev_loader.index_with(vocab)

    # You obviously won't want to create a temporary file for your training
    # results, but for execution in binder for this guide, we need to do this.
    with tempfile.TemporaryDirectory() as serialization_dir:
        trainer = build_trainer(model, serialization_dir, train_loader, dev_loader)
        print("Starting training")
        trainer.train()
        print("Finished training")

    return model, dataset_reader


def build_data_loaders(
    train_data: List[Instance],
    dev_data: List[Instance],
) -> Tuple[DataLoader, DataLoader]:
    train_loader = SimpleDataLoader(train_data, 8, shuffle=True)
    dev_loader = SimpleDataLoader(dev_data, 8, shuffle=False)
    return train_loader, dev_loader


def build_trainer(
    model: Model,
    serialization_dir: str,
    train_loader: DataLoader,
    dev_loader: DataLoader,
) -> Trainer:
    parameters = [(n, p) for n, p in model.named_parameters() if p.requires_grad]
    optimizer = AdamOptimizer(parameters)  # type: ignore
    trainer = GradientDescentTrainer(
        model=model,
        serialization_dir=serialization_dir,
        data_loader=train_loader,
        validation_data_loader=dev_loader,
        num_epochs=5,
        optimizer=optimizer,
    )
    return trainer

run_training_loop()
```

### 用allennlp内置的训练框架

上面的所有的`build_*`方法，allennlp都有对应实现，可以直接使用，然后自定义自己的`DatasetReader`和`Model`类。

这个方法通过`json`配置文件来指定所有的参数，然后框架会创建对应的对象，然后进行training loop。

AllenNLP依赖模型构造器中的类型注释，来正确地构造这些对象。

举例：

```python
def build_model(vocab: Vocabulary) -> Model:
    print("Building the model")
    vocab_size = vocab.get_vocab_size("tokens")
    embedder = BasicTextFieldEmbedder(
        {"tokens": Embedding(embedding_dim=10, num_embeddings=vocab_size)})
    encoder = BagOfEmbeddingsEncoder(embedding_dim=10)
    return SimpleClassifier(vocab, embedder, encoder)
```

对应的`JSON`字典应该像下面这个：

```json
"model": {
    "type": "simple_classifier",
    "embedder": {
        "token_embedders": {
            "tokens": {
                "type": "embedding",
                "embedding_dim": 10
            }
        }
    },
    "encoder": {
        "type": "bag_of_embeddings",
        "embedding_dim": 10
    }
}
```

> There are two special things to note: first, to select a particular subclass of a base type (e.g., SimpleClassifier as a subclass of Model,
> or BagOfEmbeddingsEncoder as a subclass of Seq2VecEncoder) we need an additional "type": "simple_classifier" key.
> The string "simple_classifier" comes from the call to Model.register that we saw in the previous chapter.

这里的vocab没有配置在json中，因为它通过data构造出来后，直接pass进model。

通常，显示为 `build_ *` 方法的参数的对象之间的顺序依赖关系被排除在配置文件之外，因为它们以不同的方式处理。

上面的方式同样应用于dataset reader, dataloader, trainer等所有对象。

实际上，allennlp用的是`Jsonnet`，是`json`的superset，支持更优雅的特征比如变量和imports，但是能像JSON文件一样使用。

上面的classifier的配置可以像下面这样：

```json
{
    "dataset_reader" : {
        "type": "classification-tsv",
        "token_indexers": {
            "tokens": {
                "type": "single_id"
            }
        }
    },
    "train_data_path": "quick_start/data/movie_review/train.tsv",
    "validation_data_path": "quick_start/data/movie_review/dev.tsv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "tokens": {
                    "type": "embedding",
                    "embedding_dim": 10
                }
            }
        },
        "encoder": {
            "type": "bag_of_embeddings",
            "embedding_dim": 10
        }
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
    "trainer": {
        "optimizer": "adam",
        "num_epochs": 5
    }
}

```

配置文件中可以看到所有在build_ *方法参数对应的条目（词汇表除外，我们将其省略，因为我们仅使用默认参数）。 通过按名称将JSON对象中的键与构造函数参数进行匹配来读取配置文件。如果keys不匹配，会有`ConfigurationError`。

这种训练方式使用`allennlp train [config.json] -s [serialization_dir]`。

allennlp通过`.register()`方法来导入类，和其他框架差不多，都是用装饰器。

### evaluate过程

allennlp用`Metric`来追踪一些训练过程中的指标。

下面以accuracy指标为例。

只需要在模型的初始化中加入：`self.accuracy = CategoricalAccuracy()`，然后在forward path中更新metric，`self.accuracy(logits, label)`。

AllenNLP中每个metric都会维护`counts`的计数，来计算这个metric。对于上面的acc，维护的counts就是预测对的数量。每次调用acc实例的时候都会更新这个变量。通过`get_metrics()`方法可以得到。这个方法需要自己在模型中实现。

```python
class SimpleClassifier(Model):
    # ....

    def get_metrics(self, reset: bool = False) -> Dict[str, float]:
        return {"accuracy": self.accuracy.get_metric(reset)}
```

命令行就直接`allennlp evaluate`就可以，它读入`allennlp train`命令生成的模型文件路径，以及包含test instances的文件路径，最后返回计算好的metric。

### prediction过程

train、evaluate过程中的样本都是有标签的，而perdition没有。因此如果想共用代码，只需要对label这个参数设置为optional。训练阶段会包含`LabelFields`，预测阶段没有。

```python
@DatasetReader.register('classification-tsv')
class ClassificationTsvReader(DatasetReader):
    def __init__(self,
                 lazy: bool = False,
                 tokenizer: Tokenizer = None,
                 token_indexers: Dict[str, TokenIndexer] = None):
        super().__init__(lazy)
        self.tokenizer = tokenizer or WhitespaceTokenizer()
        self.token_indexers = token_indexers or {'tokens': SingleIdTokenIndexer()}

    def text_to_instance(self, text: str, label: str = None) -> Instance:
        tokens = self.tokenizer.tokenize(text)
        text_field = TextField(tokens, self.token_indexers)
        fields = {'text': text_field}
        if label:
            fields['label'] = LabelField(label)
        return Instance(fields)

    def _read(self, file_path: str) -> Iterable[Instance]:
        with open(file_path, 'r') as lines:
            for line in lines:
                text, sentiment = line.strip().split('\t')
                yield self.text_to_instance(text, sentiment)
```

同样，在model的`forward`方法中，也需要进行调整，因为没有label就没法算loss，也不需要算，只要返回结果。需要在参数中设置默认值`None`，然后在计算metrics和loss的外面加一个`if`。

```python
class SimpleClassifier(Model):
    def forward(self,
                text: Dict[str, torch.Tensor],
                label: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        # Shape: (batch_size, num_tokens, embedding_dim)
        embedded_text = self.embedder(text)
        # Shape: (batch_size, num_tokens)
        mask = util.get_text_field_mask(text)
        # Shape: (batch_size, encoding_dim)
        encoded_text = self.encoder(embedded_text, mask)
        # Shape: (batch_size, num_labels)
        logits = self.classifier(encoded_text)
        # Shape: (batch_size, num_labels)
        probs = torch.nn.functional.softmax(logits)
        output = {'probs': probs}
        if label is not None:
            self.accuracy(logits, label)
            # Shape: (1,)
            output['loss'] = torch.nn.functional.cross_entropy(logits, label)
        return output
```

AllenNLPshiyong `predictors`来进行预测，它是一个在训练模型外面的包装类。主要工作是接受`json`格式的instances输入，转换为`instance`对象，传进模型，返回`json`格式的结果。

`Predictor`需要继承基类`Predictor`并且实现`predict()`和`_json_to_instances()`方法。其他的由基类负责。

```python
@Predictor.register("sentence_classifier")
class SentenceClassifierPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        # This method is implemented in the base class.
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)
```

日常的一些任务都已经有内置的`Predictor`了，所以先检查一下有没有再自己写。

如果要自己写

```python
class SentenceClassifierPredictor(Predictor):
    def predict(self, sentence: str) -> JsonDict:
        return self.predict_json({"sentence": sentence})

    def _json_to_instance(self, json_dict: JsonDict) -> Instance:
        sentence = json_dict["sentence"]
        return self._dataset_reader.text_to_instance(sentence)


# We've copied the training loop from an earlier example, with updated model
# code, above in the Setup section. We run the training loop to get a trained
# model.
model, dataset_reader = run_training_loop()
vocab = model.vocab
predictor = SentenceClassifierPredictor(model, dataset_reader)

output = predictor.predict("A good movie!")
print(
    [
        (vocab.get_token_from_index(label_id, "labels"), prob)
        for label_id, prob in enumerate(output["probs"])
    ]
)
output = predictor.predict("This was a monstrous waste of time.")
print(
    [
        (vocab.get_token_from_index(label_id, "labels"), prob)
        for label_id, prob in enumerate(output["probs"])
    ]
)
```

会得到类似下面的输出：
> [('neg', 0.48853254318237305), ('pos', 0.511467456817627)]  
> [('neg', 0.5346643924713135), ('pos', 0.4653356373310089)]

命令行和evaluate很像，接受模型的路径，和包含预测样本instances的json文件路径。

在[对应的repo](https://github.com/allenai/allennlp-guide)中，切换到敖`quick_start`目录可以直接执行下面的命令。

```bash
$ allennlp train \
    my_text_classifier.jsonnet \
    --serialization-dir model \
    --include-package my_text_classifier

$ allennlp evaluate \
    model/model.tar.gz \
    data/movie_review/test.tsv \
    --include-package my_text_classifier

$ allennlp predict \
    model/model.tar.gz \
    data/movie_review/test.jsonl \
    --include-package my_text_classifier \
    --predictor sentence_classifier
```

## 进阶

### 预训练情景

AllenNLP把很多模块抽象化，你只需要实现内部的细节，整个流程如何连接则不需要自己管。比如`TextFieldEmbedder`和`Seq2Vector`，只要接口和他们相同，就可以使用它。

`SeqVevEncoder`可以是任何，只要是输入为一个序列向量`(batch_size, num_tokens, embedding_dim)`，输出为`(batch_size, encoding_dim)`，都可以使用它。比如bag of embedding, CNN, RNN, Transformer等。

使用bert的一个demo

```python
local bert_model = "bert-base-uncased";

{
    "dataset_reader" : {
        "type": "classification-tsv",
        "tokenizer": {
            "type": "pretrained_transformer",
            "model_name": bert_model,
        },
        "token_indexers": {
            "bert": {
                "type": "pretrained_transformer",
                "model_name": bert_model,
            }
        },
        "max_tokens": 512
    },
    "train_data_path": "quick_start/data/movie_review/train.tsv",
    "validation_data_path": "quick_start/data/movie_review/dev.tsv",
    "model": {
        "type": "simple_classifier",
        "embedder": {
            "token_embedders": {
                "bert": {
                    "type": "pretrained_transformer",
                    "model_name": bert_model
                }
            }
        },
        "encoder": {
            "type": "bert_pooler",
            "pretrained_model": bert_model
        }
    },
    "data_loader": {
        "batch_size": 8,
        "shuffle": true
    },
    "trainer": {
        "optimizer": {
            "type": "huggingface_adamw",
            "lr": 1.0e-5
        },
        "num_epochs": 5
    }
}
```

下面直接复制了一段

> Use a PretrainedTransformerTokenizer ("pretrained_transformer"), which tokenizes the string into wordpieces and adds special tokens like [CLS] and [SEP]
>
> Use a PretrainedTransformerIndexer ("pretrained_transformer"), which converts those wordpieces into ids using BERT's vocabulary
>
> Replace the embedder layer with a PretrainedTransformerEmbedder ("pretrained_transformer"), which uses a pretrained BERT model to embed the tokens, returning the top layer from BERT
>
> Replace the encoder with a BertPooler ("bert_pooler"), which adds another (pretrained) linear layer on top of the [CLS] token and returns the result

```bash
$ allennlp train \
    my_text_classifier.jsonnet \
    --serialization-dir model-bert \
    --include-package my_text_classifier
```

修改不同的transformer架构如BERT，RoBERTa，XLNet，都只需要改`model_name`就可以。上面的修改也只是改了配置文件，没有改模型代码。

### 部署网页展示demo

还可以直接部署一个网页展示，

```bash
pip install allennlp-server
python allennlp-server/server_simple.py \
    --archive-path model/model.tar.gz \
    --predictor sentence_classifier \
    --field-name sentence
    --include-package my_text_classifier
```

具体的需要再查。

### GPU相关

用allennlp不需要自己去让模型兼容gpu，或者手动把参数，模型移到gpu上。直接添加`cuda_device`选项，指定GPU的device id。

```json
    "trainer": {
        ...
        "cuda_device": 0
        ...
    }
```

或者用分布式多gpu，使用pytorch的`DistributedDataParallel`包。eval和predict的时候可以直接在命令行指定cuda device。

```json
    "trainer": {
        ...
    },
    "distributed": {
        "cuda_devices": [0, 1, 2, 3]
    }
```