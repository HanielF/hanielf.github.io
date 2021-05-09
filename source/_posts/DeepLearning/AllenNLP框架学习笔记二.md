---
title: AllenNLP框架学习笔记（二）
mathjax: false
date: 2021-04-26 20:11:01
tags:
  [
    AllenNLP,
    NLP,
    OpenLibrary,
    DeepLearning,
    Pytorch,
    Private,
  ]
categories: MachineLearning
urlname: allennlp-notes-2
---

<meta name="referrer" content="no-referrer" />

{% note info %}
进一步理解和使用AllenNLP这个框架
{% endnote %}

<!--more-->

## 数据相关

### Fields

所有的数据都被包装成`Fields`类，一个field包含一个数据样本，在模型中会被转换为tensor作为输入和输出。

有很多种类的fields类
- LabelField
- MultiLabelField
- SequenceLabelField
- SpanField
- ListField
- ArrayField

![](https://i.loli.net/2021/04/26/cezXt9NnbB1foYd.png)

用法参考：

```python
# To create fields, simply pass the data to constructor.
# NOTE: Don't worry about the token_indexers too much for now. We have a whole
# chapter on why TextFields are set up this way, and how they work.
tokens = [Token("The"), Token("best"), Token("movie"), Token("ever"), Token("!")]
token_indexers: Dict[str, TokenIndexer] = {"tokens": SingleIdTokenIndexer()}
text_field = TextField(tokens, token_indexers=token_indexers)

label_field = LabelField("pos")

sequence_label_field = SequenceLabelField(
    ["DET", "ADJ", "NOUN", "ADV", "PUNKT"], text_field
)

# You can use print() fields to see their content
print(text_field)
print(label_field)
print(sequence_label_field)

# Many of the fields implement native python methods in intuitive ways
print(len(sequence_label_field))
print(label for label in sequence_label_field)

# Fields know how to create empty fields of the same type
print(text_field.empty_field())
print(label_field.empty_field())
print(sequence_label_field.empty_field())

# You can count vocabulary items in fields
counter: Dict[str, Dict[str, int]] = defaultdict(Counter)
text_field.count_vocab_items(counter)
print(counter)

label_field.count_vocab_items(counter)
print(counter)

sequence_label_field.count_vocab_items(counter)
print(counter)

# Create Vocabulary for indexing fields
vocab = Vocabulary(counter)

# Fields know how to turn themselves into tensors
text_field.index(vocab)
# NOTE: in practice, we will batch together instances and use the maximum padding
# lengths, instead of getting them from a single instance.
# You can print this if you want to see what the padding_lengths dictionary looks
# like, but it can sometimes be a bit cryptic.
padding_lengths = text_field.get_padding_lengths()
print(text_field.as_tensor(padding_lengths))

label_field.index(vocab)
print(label_field.as_tensor(label_field.get_padding_lengths()))

sequence_label_field.index(vocab)
padding_lengths = sequence_label_field.get_padding_lengths()
print(sequence_label_field.as_tensor(padding_lengths))

# Fields know how to batch tensors
tensor1 = label_field.as_tensor(label_field.get_padding_lengths())

label_field2 = LabelField("pos")
label_field2.index(vocab)
tensor2 = label_field2.as_tensor(label_field2.get_padding_lengths())

batched_tensors = label_field.batch_tensors([tensor1, tensor2])
print(batched_tensors)
```

### instances

一个Instance是一个模型预测的原子单位，是Fields类实例的集合，也是dataset的组成单位。fields->instances->datasets。

Instances用dataset reader创建，然后可以从里面获得`Vocabulary`。vocab可以把`Fields`map到id上。然后这些instances会被转换为batch tensor输入到模型中。

![I6x77E](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/others/I6x77E.png)

通过把field name和对应fields的字典传进constructor，可以创建instances。instances可以转换为field name和对应tensors的字典，每个field name对应的tensor可以被batch组织进模型中。

而且每个Field 名字是很重要的，因为最后包含tensor的字典会和key一起传进模型，**所以要和模型forward方法中的参数保持一致。**

示例代码：

```python
# Create Fields
tokens = [Token("The"), Token("best"), Token("movie"), Token("ever"), Token("!")]
token_indexers: Dict[str, TokenIndexer] = {"tokens": SingleIdTokenIndexer()}
text_field = TextField(tokens, token_indexers=token_indexers)

label_field = LabelField("pos")

sequence_label_field = SequenceLabelField(
    ["DET", "ADJ", "NOUN", "ADV", "PUNKT"], text_field
)

# Create an Instance
fields: Dict[str, Field] = {
    "tokens": text_field,
    "label": label_field,
}
instance = Instance(fields)

# You can add fields later
instance.add_field("label_seq", sequence_label_field)

# You can simply use print() to see the instance's content
print(instance)

# Create a Vocabulary
counter: Dict[str, Dict[str, int]] = defaultdict(Counter)
instance.count_vocab_items(counter)
vocab = Vocabulary(counter)

# Convert all strings in all of the fields into integer IDs by calling index_fields()
instance.index_fields(vocab)

# Instances know how to turn themselves into a dict of tensors.  When we call this
# method in our data code, we additionally give a `padding_lengths` argument.
# We will pass this dictionary to the model as **tensors, so be sure the keys
# match what the model expects.
tensors = instance.as_tensor_dict()
print(tensors)
```

基本上是：token-> token_indexers->text_field->label_field->fields->instance->counter->instance.count_vocab_items(counter)->vocab->instance.index_fields()

待补充

## 模型相关

中间待补充

## 通用的架构

待补充

## 特征表示相关

### 文本到特征

要将文本进行编码
1. `Tokenizer`将文本拆分成独立的`Tokens`
1. 用`TextField, TokenIndexer, and Vocabulary`将Tokens转换成index
1. 用`TextFieldEmbedder`获得每个Token的编码，只有这一步的参数是learnable的

即Text->Tokens->Ids->Vectors，前两步骤有`DatasetReader`负责，最后一步由`Model`负责。

编码方式很多，常见的Glove、Word2vec、字符级别的CNN、POS tag embedding、结合Glove和CNN，以及**wordpieces级别的BERT**

### Tokenizer

主要有三种方式tokenize
- 字符（包括空格），Characters ("AllenNLP is great" → ["A", "l", "l", "e", "n", "N", "L", "P", " ", "i", "s", " ", "g", "r", "e", "a", "t"])
- Wordpieces ("AllenNLP is great" → ["Allen", "##NL", "##P", "is", "great"])
- Words ("AllenNLP is great" → ["AllenNLP", "is", "great"])

常用：
- SpacyTokenizer
- PretrainedTransformerTokenizer, which uses a tokenizer from Hugging Face's transformers library
- CharacterTokenizer, which splits a string up into individual characters, including whitespace.

每个tokenizer都实现了`tokenize()`方法，会返回Tokens列表。一个Token是一个轻量级的dataclass

### TextFields

A TextField takes a list of Tokens from a Tokenizer and represents each of them as an array that can be converted into a vector by the model

TextFields读入Tokens列表，然后把每个token表示成一个idx。

包括这些方法，主要是给TokenIndexers负责，
- counting vocabulary items
- converting strings to integers and then tensors
- batching together several tensors with proper padding

```python
tokenizer = ...  # Whatever tokenizer you want
sentence = "We are learning about TextFields"
tokens = tokenizer.tokenize(sentence)
token_indexers = {...}  # we'll talk about this in the next section
text_field = TextField(tokens, token_indexers)
...
instance = Instance({"sentence": text_field, ...})
```

### TokenIndexer

所有的Token idx都从2开始，0是padding，1是unk

可以组合使用不同的TokenIndexer，然后进行embedding的融合。

```python
# Splits text into words (instead of wordpieces or characters).
tokenizer: Tokenizer = WhitespaceTokenizer()

# Represents each token with both an id from a vocabulary and a sequence of
# characters.
token_indexers: Dict[str, TokenIndexer] = {
    "tokens": SingleIdTokenIndexer(namespace="token_vocab"),
    "token_characters": TokenCharactersIndexer(namespace="character_vocab"),
}

vocab = Vocabulary()
vocab.add_tokens_to_namespace(
    ["This", "is", "some", "text", "."], namespace="token_vocab"
)
vocab.add_tokens_to_namespace(
    ["T", "h", "i", "s", " ", "o", "m", "e", "t", "x", "."], namespace="character_vocab"
)

text = "This is some text ."
tokens = tokenizer.tokenize(text)
print("Tokens:", tokens)

# The setup here is the same as what we saw above.
text_field = TextField(tokens, token_indexers)
text_field.index(vocab)
padding_lengths = text_field.get_padding_lengths()
tensor_dict = text_field.as_tensor(padding_lengths)
# Note now that we have two entries in this output dictionary,
# one for each indexer that we specified.
print("Combined tensor dictionary:", tensor_dict)

# Now we split text into words with part-of-speech tags, using Spacy's POS tagger.
# This will result in the `tag_` variable being set on each `Token` object, which
# we will read in the indexer.
tokenizer = SpacyTokenizer(pos_tags=True)
vocab.add_tokens_to_namespace(["DT", "VBZ", "NN", "."], namespace="pos_tag_vocab")

# Represents each token with (1) an id from a vocabulary, (2) a sequence of
# characters, and (3) part of speech tag ids.
token_indexers = {
    "tokens": SingleIdTokenIndexer(namespace="token_vocab"),
    "token_characters": TokenCharactersIndexer(namespace="character_vocab"),
    "pos_tags": SingleIdTokenIndexer(namespace="pos_tag_vocab", feature_name="tag_"),
}

tokens = tokenizer.tokenize(text)
print("Spacy tokens:", tokens)
print("POS tags:", [token.tag_ for token in tokens])

text_field = TextField(tokens, token_indexers)
text_field.index(vocab)

padding_lengths = text_field.get_padding_lengths()

tensor_dict = text_field.as_tensor(padding_lengths)
print("Tensor dict with POS tags:", tensor_dict)
```

输出：

```
Spacy models 'en_core_web_sm' not found.  Downloading and installing.
Tokens: [This, is, some, text, .]
Combined tensor dictionary: {'tokens': {'tokens': tensor([2, 3, 4, 5, 6])}, 'token_characters': {'token_characters': tensor([[ 2,  3,  4,  5],
        [ 4,  5,  0,  0],
        [ 5,  7,  8,  9],
        [10,  9, 11, 10],
        [12,  0,  0,  0]])}}
✔ Download and installation successful
You can now load the package via spacy.load('en_core_web_sm')
Spacy tokens: [This, is, some, text, .]
POS tags: ['DT', 'VBZ', 'DT', 'NN', '.']
Tensor dict with POS tags: {'tokens': {'tokens': tensor([2, 3, 4, 5, 6])}, 'token_characters': {'token_characters': tensor([[ 2,  3,  4,  5],
        [ 4,  5,  0,  0],
        [ 5,  7,  8,  9],
        [10,  9, 11, 10],
        [12,  0,  0,  0]])}, 'pos_tags': {'tokens': tensor([2, 3, 2, 4, 5])}}
```

### TextFieldEmbedders

#### 单个Indexer

allennlp数据处理的时候，会把instances中的TextFiled转换成TextFiledTensors数据结构，即：` Dict[str, Dict[str, torch.Tensor]]`，外围str对应每个TokenIndexers，里面对应TokenIndexer生成的idx。这个会被输入到TextFieldEmbedder中，用TokenEmbedder来embeds or encodes。

下面的例子是两个单独的embedding，分别是普通的embedding和cnn encoder。
```python
# This is what gets created by TextField.as_tensor with a SingleIdTokenIndexer;
# Note that we added the batch dimension at the front.  You choose the 'indexer1'
# name when you configure your data processing code.
token_tensor = {
    "indexer1": {
        "tokens": torch.LongTensor(
            [[1, 3, 2, 9, 4, 3]]
         )
     }
}

# You would typically get the number of embeddings here from the vocabulary;
# if you use `allennlp train`, there is a separate process for instantiating the
# Embedding object using the vocabulary that you don't need to worry about for
# now.
embedding = Embedding(num_embeddings=10, embedding_dim=3)

# This 'indexer1' key must match the 'indexer1' key in the `token_tensor` above.
# We use these names to align the TokenIndexers used in the data code with the
# TokenEmbedders that do the work on the model side.
embedder = BasicTextFieldEmbedder(token_embedders={"indexer1": embedding})

embedded_tokens = embedder(token_tensor)
print("Using the TextFieldEmbedder:", embedded_tokens)

# As we've said a few times, what's going on inside is that we match keys between
# the token tensor and the token embedders, then pass the inner dictionary to the
# token embedder.  The above lines perform the following logic:
embedded_tokens = embedding(**token_tensor["indexer1"])
print("Using the Embedding directly:", embedded_tokens)

# This is what gets created by TextField.as_tensor with a TokenCharactersIndexer
# Note that we added the batch dimension at the front. Don't worry too much
# about the magic 'token_characters' key - that is hard-coded to be produced
# by the TokenCharactersIndexer, and accepted by TokenCharactersEncoder;
# you don't have to produce those yourself in normal settings, it's done for you.
token_tensor = {
    "indexer2": {
        "token_characters": torch.LongTensor(
            [[[1, 3, 0], [4, 2, 3], [1, 9, 5], [6, 0, 0]]]
        )
    }
}

character_embedding = Embedding(num_embeddings=10, embedding_dim=3)
cnn_encoder = CnnEncoder(embedding_dim=3, num_filters=4, ngram_filter_sizes=(3,))
token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)

# Again here, the 'indexer2' key is arbitrary. It just has to match whatever key
# you gave to the corresponding TokenIndexer in your data code, which ends up
# as the top-level key in the token_tensor dictionary.
embedder = BasicTextFieldEmbedder(token_embedders={"indexer2": token_encoder})

embedded_tokens = embedder(token_tensor)
print("With a character CNN:", embedded_tokens)
```

输出：

```python
Using the TextFieldEmbedder: tensor([[[ 0.6184,  0.3636, -0.6774],
         [-0.0317, -0.5588,  0.6220],
         [ 0.2992, -0.2631, -0.4046],
         [ 0.4240,  0.2915,  0.6677],
         [-0.6025,  0.2038, -0.0412],
         [-0.0317, -0.5588,  0.6220]]], grad_fn=<CatBackward>)
Using the Embedding directly: tensor([[[ 0.6184,  0.3636, -0.6774],
         [-0.0317, -0.5588,  0.6220],
         [ 0.2992, -0.2631, -0.4046],
         [ 0.4240,  0.2915,  0.6677],
         [-0.6025,  0.2038, -0.0412],
         [-0.0317, -0.5588,  0.6220]]], grad_fn=<EmbeddingBackward>)
With a character CNN: tensor([[[0.0000, 0.0000, 0.0000, 0.0000],
         [0.0000, 0.3550, 0.5636, 0.3409],
         [0.1199, 0.0000, 0.2336, 0.2741],
         [0.0000, 0.0000, 0.0000, 0.0000]]], grad_fn=<CatBackward>)
```

注意到上面的都是用`BasicTextFieldEmbedder(token_embedders={"indexer_name": embedding or encoder})`。

#### 多个Indexer

下面的例子是多个Indexer，其中一个是单个word，另一个是sequence of characters per token，最后一个是different single ID,对应不同的 speech tags。

```python
# This is what gets created by TextField.as_tensor with a SingleIdTokenIndexer
# and a TokenCharactersIndexer; see the code snippet above. This time we're using
# more intuitive names for the indexers and embedders.
token_tensor = {
    "tokens": {
        "tokens": torch.LongTensor([[2, 4, 3, 5]])
    },
    "token_characters": {
        "token_characters": torch.LongTensor(
            [[[2, 5, 3], [4, 0, 0], [2, 1, 4], [5, 4, 0]]]
        )
    },
}

# This is for embedding each token.
embedding = Embedding(num_embeddings=6, embedding_dim=3)

# This is for encoding the characters in each token.
character_embedding = Embedding(num_embeddings=6, embedding_dim=3)
cnn_encoder = CnnEncoder(embedding_dim=3, num_filters=4, ngram_filter_sizes=(3,))
token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)

# 用名字来表示不同的namespace
embedder = BasicTextFieldEmbedder(
    token_embedders={"tokens": embedding, "token_characters": token_encoder}
)

embedded_tokens = embedder(token_tensor)

print("没有 speech tag")
print(embedded_tokens)

# This is what gets created by TextField.as_tensor with a SingleIdTokenIndexer,
# a TokenCharactersIndexer, and another SingleIdTokenIndexer for PoS tags;
# see the code above.
token_tensor = {
    "tokens": {
        "tokens": torch.LongTensor([[2, 4, 3, 5]])
    },
    "token_characters": {
        "token_characters": torch.LongTensor(
            [[[2, 5, 3], [4, 0, 0], [2, 1, 4], [5, 4, 0]]]
        )
    },
    "pos_tag_tokens": {
        "tokens": torch.LongTensor([[2, 5, 3, 4]])
    },
}

vocab = Vocabulary()
vocab.add_tokens_to_namespace(
    ["This", "is", "some", "text", "."], namespace="token_vocab"
)
vocab.add_tokens_to_namespace(
    ["T", "h", "i", "s", " ", "o", "m", "e", "t", "x", "."], namespace="character_vocab"
)
vocab.add_tokens_to_namespace(["DT", "VBZ", "NN", "."], namespace="pos_tag_vocab")

# Notice below how the 'vocab_namespace' parameter matches the name used above.
# We're showing here how the code works when we're constructing the Embedding from
# a configuration file, where the vocabulary object gets passed in behind the
# scenes (but the vocab_namespace parameter must be set in the config). If you are
# using a `build_model` method (see the quick start chapter) or instantiating the
# Embedding yourself directly, you can just grab the vocab size yourself and pass
# in num_embeddings, as we do above.

# This is for embedding each token.
# 这里用vocab_namespace的名字和对应的vocab匹配
embedding = Embedding(
    embedding_dim=3, vocab_namespace="token_vocab", vocab=vocab
)

# This is for encoding the characters in each token.
character_embedding = Embedding(
    embedding_dim=4, vocab_namespace="character_vocab", vocab=vocab
)
cnn_encoder = CnnEncoder(embedding_dim=4, num_filters=5, ngram_filter_sizes=(3,))
token_encoder = TokenCharactersEncoder(character_embedding, cnn_encoder)

# This is for embedding the part of speech tag of each token.
pos_tag_embedding = Embedding(
    embedding_dim=6, vocab_namespace="pos_tag_vocab", vocab=vocab
)

# Notice how these keys match the keys in the token_tensor dictionary above;
# these are the keys that you give to your TokenIndexers when constructing
# your TextFields in the DatasetReader.
embedder = BasicTextFieldEmbedder(
    token_embedders={
        "tokens": embedding,
        "token_characters": token_encoder,
        "pos_tag_tokens": pos_tag_embedding,
    }
)

embedded_tokens = embedder(token_tensor)
print("有 speech tag")
print(embedded_tokens)
```

输出

```python
没有 speech tag
tensor([[[ 0.3090,  0.0000,  0.4446,  0.0000, -0.0025,  0.4985,  0.6270],
         [ 0.0000,  0.0000,  0.0000,  0.0000,  0.3697,  0.7020, -0.6689],
         [ 0.2016,  0.0000,  0.7770,  0.0000, -0.4487, -0.6927,  0.1282],
         [ 0.0000,  0.0000,  0.0000,  0.0000, -0.0455,  0.0369,  0.0783]]],
       grad_fn=<CatBackward>)
有 speech tag
tensor([[[-0.5869,  0.1731,  0.4276, -0.6160,  0.5848,  0.6462,  0.0000,
           0.0000,  0.2893,  0.0415,  0.0000,  0.1478,  0.1721, -0.2290],
         [-0.2442, -0.0461, -0.3557, -0.1157, -0.0065, -0.1078,  0.0000,
           0.0000,  0.0000,  0.0000,  0.0000, -0.0331, -0.4360,  0.6749],
         [-0.0437,  0.7053, -0.5893, -0.1253, -0.4747,  0.0396,  0.0131,
           0.2123,  0.2087,  0.0000,  0.0000, -0.6189,  0.3971, -0.6693],
         [-0.2701, -0.3194,  0.0756,  0.6921,  0.4557,  0.5086,  0.0000,
           0.0000,  0.0000,  0.0000,  0.0000, -0.1958,  0.4875, -0.6018]]],
       grad_fn=<CatBackward>)
```

要注意，有两个地方的key要匹配：
1. vocab 的 namespace，在TokenIndexers和TokenEmbedders中需要匹配，即`embedding = Embedding(embedding_dim=3, vocab_namespace="token_vocab", vocab=vocab)`
2. TextField中用于TokenIndexer词典的key需要与BasicTextFieldEmbedder中用于TokenEmbedder词典的key匹配，就是`embedder = BasicTextFieldEmbedder(token_embedders={"token_characters": token_encoder})`里面的token_characters。

### Tokenizer, TokenIndexer, ToeknEmbedders的配合

你需要配置代码以选择要用作具体Tokenizer，TokenIndexers和TokenEmbedders。需要确保选择适合的组件，否则代码将无法正常工作。

比如，选择一个CharacterTokenizer和一个TokenCharactersIndexer并没有任何意义，因为Indexer假定您已将其标记为单词。

并且，TokenIndexer的输出会输入到TokenEmbedder中，通过使用key值。通常会有一对一的关系在 TokenIndexer 和 TokenEmbedder，并且一种 TokenIndexer 可能只对一个 Tokenizer 起作用

> Using a word-level tokenizer (such as SpacyTokenizer or WhitespaceTokenizer):
> 
> - SingleIdTokenIndexer → Embedding (for things like GloVe or other simple embeddings, including learned POS tag embeddings)  
> - TokenCharactersIndexer → TokenCharactersEncoder (for things like a character CNN)  
> - ElmoTokenIndexer → ElmoTokenEmbedder (for ELMo)  
> - PretrainedTransformerMismatchedIndexer → PretrainedTransformerMismatchedEmbedder (for using a transformer like BERT when you really want to do modeling at the word level, e.g., for a tagging task; more on what this does below)  
> 
> Using a character-level tokenizer (such as CharacterTokenizer):  
> 
> - SingleIdTokenIndexer → Embedding  
> 
> Using a wordpiece tokenizer (such as PretrainedTransformerTokenizer):  
> 
> - PretrainedTransformerIndexer → PretrainedTransformerEmbedder  
> - SingleIdTokenIndexer → Embedding (if you don't want contextualized wordpieces for some reason)  

### 用预训练编码

这里主要讲如何使用类似 ELMo, BERT 的预训练上下文编码，只要选择不同的Tokenizer、Indexer、Embedder.

#### 获取text_fields

```python
# Splits text into words (instead of wordpieces or characters).  For ELMo, you can
# just use any word-level tokenizer that you like, though for best results you
# should use the same tokenizer that was used with ELMo, which is an older version
# of spacy.  We're using a whitespace tokenizer here for ease of demonstration
# with binder.
tokenizer: Tokenizer = WhitespaceTokenizer()

# Represents each token with an array of characters in a way that ELMo expects.
token_indexer: TokenIndexer = ELMoTokenCharactersIndexer()

# Both ELMo and BERT do their own thing with vocabularies, so we don't need to add
# anything, but we do need to construct the vocab object so we can use it below.
# (And if you have any labels in your data that need indexing, you'll still need
# this.)
vocab = Vocabulary()

text = "This is some text ."
tokens = tokenizer.tokenize(text)
print("ELMo tokens:", tokens)

text_field = TextField(tokens, {"elmo_tokens": token_indexer})
text_field.index(vocab)

# We typically batch things together when making tensors, which requires some
# padding computation.  Don't worry too much about the padding for now.
padding_lengths = text_field.get_padding_lengths()

tensor_dict = text_field.as_tensor(padding_lengths)
print("ELMo tensors:", tensor_dict)

# Any transformer model name that huggingface's transformers library supports will
# work here.  Under the hood, we're grabbing pieces from huggingface for this
# part.
transformer_model = "bert-base-cased"

# To do modeling with BERT correctly, we can't use just any tokenizer; we need to
# use BERT's tokenizer.
tokenizer = PretrainedTransformerTokenizer(model_name=transformer_model)

# Represents each wordpiece with an id from BERT's vocabulary.
token_indexer = PretrainedTransformerIndexer(model_name=transformer_model)

text = "Some text with an extraordinarily long identifier."
tokens = tokenizer.tokenize(text)
print("BERT tokens:", tokens)

text_field = TextField(tokens, {"bert_tokens": token_indexer})
text_field.index(vocab)

tensor_dict = text_field.as_tensor(text_field.get_padding_lengths())
print("BERT tensors:", tensor_dict)

# Now we'll do an example with paired text, to show the right way to handle [SEP]
# tokens in AllenNLP.  We have built-in ways of handling this for two text pieces.
# If you have more than two text pieces, you'll have to manually add the special
# tokens.  The way we're doing this requires that you use a
# PretrainedTransformerTokenizer, not the abstract Tokenizer class.

# Splits text into wordpieces, but without adding special tokens.
tokenizer = PretrainedTransformerTokenizer(
    model_name=transformer_model,
    add_special_tokens=False,
)

context_text = "This context is frandibulous."
question_text = "What is the context like?"
context_tokens = tokenizer.tokenize(context_text)
question_tokens = tokenizer.tokenize(question_text)
print("Context tokens:", context_tokens)
print("Question tokens:", question_tokens)

combined_tokens = tokenizer.add_special_tokens(context_tokens, question_tokens)
print("Combined tokens:", combined_tokens)

text_field = TextField(combined_tokens, {"bert_tokens": token_indexer})
text_field.index(vocab)

tensor_dict = text_field.as_tensor(text_field.get_padding_lengths())
print("Combined BERT tensors:", tensor_dict)
```

输出：

```
ELMo tokens: [This, is, some, text, .]
ELMo tensors: {'elmo_tokens': {'elmo_tokens': tensor([[259,  85, 105, 106, 116, 260, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261],
        [259, 106, 116, 260, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261],
        [259, 116, 112, 110, 102, 260, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261],
        [259, 117, 102, 121, 117, 260, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261],
        [259,  47, 260, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261]])}}
Downloading: 100%|██████████| 570/570 [00:00<00:00, 164kB/s]
Downloading: 100%|██████████| 213k/213k [00:01<00:00, 203kB/s]
Downloading: 100%|██████████| 436k/436k [00:01<00:00, 300kB/s]
Downloading: 100%|██████████| 29.0/29.0 [00:00<00:00, 10.5kB/s]
BERT tokens: [[CLS], Some, text, with, an, extra, ##ord, ##ina, ##rily, long, id, ##ent, ##ifier, ., [SEP]]
BERT tensors: {'bert_tokens': {'token_ids': tensor([  101,  1789,  3087,  1114,  1126,  3908,  6944,  2983, 11486,  1263,
        25021,  3452, 17792,   119,   102]), 'mask': tensor([True, True, True, True, True, True, True, True, True, True, True, True,
        True, True, True]), 'type_ids': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])}}
Context tokens: [This, context, is, f, ##rand, ##ib, ##ulous, .]
Question tokens: [What, is, the, context, like, ?]
Combined tokens: [[CLS], This, context, is, f, ##rand, ##ib, ##ulous, ., [SEP], What, is, the, context, like, ?, [SEP]]
Combined BERT tensors: {'bert_tokens': {'token_ids': tensor([  101,  1188,  5618,  1110,   175, 13141, 13292, 14762,   119,   102,
         1327,  1110,  1103,  5618,  1176,   136,   102]), 'mask': tensor([True, True, True, True, True, True, True, True, True, True, True, True,
        True, True, True, True, True]), 'type_ids': tensor([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1])}}
```

注意，`tokenizer.add_special_tokens()`这个方法，只能用于`PretrainedTransformerTokenizer`。并且，`TokenEmbedder`在这种情况下是作用于整个tokens序列的，而不是单个token，是在`TextFieldEmbedder`里面做self-attention算embedding，但是在代码上没啥影响。

#### embedding

```python
# It's easiest to get ELMo input by just running the data code.  See the
# exercise above for an explanation of this code.
tokenizer: Tokenizer = WhitespaceTokenizer()
token_indexer: TokenIndexer = ELMoTokenCharactersIndexer()
vocab = Vocabulary()
text = "This is some text."
tokens = tokenizer.tokenize(text)
print("ELMo tokens:", tokens)
text_field = TextField(tokens, {"elmo_tokens": token_indexer})
text_field.index(vocab)
token_tensor = text_field.as_tensor(text_field.get_padding_lengths())
print("ELMo tensors:", token_tensor)

# We're using a tiny, toy version of ELMo to demonstrate this.
elmo_options_file = (
    "https://allennlp.s3.amazonaws.com/models/elmo/test_fixture/options.json"
)
elmo_weight_file = (
    "https://allennlp.s3.amazonaws.com/models/elmo/test_fixture/lm_weights.hdf5"
)
elmo_embedding = ElmoTokenEmbedder(
    options_file=elmo_options_file, weight_file=elmo_weight_file
)

embedder = BasicTextFieldEmbedder(token_embedders={"elmo_tokens": elmo_embedding})

tensor_dict = text_field.batch_tensors([token_tensor])
embedded_tokens = embedder(tensor_dict)
print("ELMo embedded tokens:", embedded_tokens)


# Again, it's easier to just run the data code to get the right output.

# We're using the smallest transformer model we can here, so that it runs on
# binder.
transformer_model = "google/reformer-crime-and-punishment"
tokenizer = PretrainedTransformerTokenizer(model_name=transformer_model)
token_indexer = PretrainedTransformerIndexer(model_name=transformer_model)
text = "Some text with an extraordinarily long identifier."
tokens = tokenizer.tokenize(text)
print("Transformer tokens:", tokens)
text_field = TextField(tokens, {"bert_tokens": token_indexer})
text_field.index(vocab)
token_tensor = text_field.as_tensor(text_field.get_padding_lengths())
print("Transformer tensors:", token_tensor)

embedding = PretrainedTransformerEmbedder(model_name=transformer_model)

embedder = BasicTextFieldEmbedder(token_embedders={"bert_tokens": embedding})

tensor_dict = text_field.batch_tensors([token_tensor])
embedded_tokens = embedder(tensor_dict)
print("Transformer embedded tokens:", embedded_tokens)
```

输出：

```

ELMo tokens: [This, is, some, text.]
ELMo tensors: {'elmo_tokens': {'elmo_tokens': tensor([[259,  85, 105, 106, 116, 260, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261],
        [259, 106, 116, 260, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261],
        [259, 116, 112, 110, 102, 260, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261],
        [259, 117, 102, 121, 117,  47, 260, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261, 261,
         261, 261, 261, 261, 261, 261, 261, 261]])}}
ELMo embedded tokens: tensor([[[ 0.0000,  0.0000, -0.0000, -0.0000, -0.8512, -0.0000, -0.4313,
          -1.4576,  0.0000, -0.0000,  0.0271,  0.0000, -0.0000,  0.0000,
          -0.0000, -0.1257, -0.4585,  0.0000, -1.1158, -0.2194, -1.5000,
          -1.4474, -0.0000, -0.0000,  0.0000, -0.0000, -0.2561, -0.0000,
          -0.0000,  0.0740, -0.0000, -0.0000],
         [ 0.0000,  0.0000, -0.0000, -0.8952, -0.8725,  0.3791, -0.5978,
          -0.3816,  0.0000, -0.0000,  0.0000, -0.1073, -0.0000,  0.2156,
           0.0582,  0.0000,  0.3820,  0.5719, -0.0000, -0.6818, -0.7399,
          -1.2560, -1.4208,  0.3838,  0.0000, -0.0000, -0.7528,  0.1510,
          -1.5196, -0.0000, -0.0000,  0.6609],
         [ 0.7935,  1.2918, -0.1949, -0.7476, -0.0000, -0.2793, -0.8381,
          -1.2474,  0.1516, -0.0000,  0.0000,  0.0000, -0.7108,  0.0000,
          -0.0000,  0.4098, -0.3095,  0.0000, -1.7565,  0.4459, -0.6707,
          -1.4732, -0.0000,  0.0000,  0.0000, -0.6564,  0.1276,  0.0000,
          -0.0000,  0.5801,  0.0000,  0.2791],
         [-0.0000, -0.0000, -6.5606,  2.1171, -2.1024,  2.5919, -4.8008,
           0.0000, -0.0000, -0.0000,  0.0000,  1.9269,  0.0000, -0.6810,
           0.0000,  3.7315, -0.0000,  0.0000, -3.3836, -0.0000, -0.0000,
           1.8097, -7.0459,  0.0000,  2.7400, -0.0000, -1.6098,  2.8753,
           0.0000,  0.0000, -0.0000, -0.4789]]], grad_fn=<CatBackward>)
```

### word-level模型同时使用wordpiece的transformer

在part-of-speech tagging或者named entity recognition中，数据集是在word level的，因此模型的loss、output都应该是word level，但是可能会需要用transformer，这是在wordpiece level的。

有两种主要的方式：
1. 在transformer运行之后，在wordpiece-level上做一些pooling
2. 把label扩散到wordpiece-level上

#### pooling over Wordpieces

> The first step is tokenization, and here we tokenize at the word level (typically the tokenization will be already given to you, so you don't need to run a tokenizer at all). 
> 
> In the second step (indexing), we need to further tokenize each word into subword units, getting a list of wordpieces that will be indexed and passed to the transformer in the third step (embedding). 
> 
> The embedding step has to run the transformer, then perform pooling to undo the subword tokenization that was done in the indexing subwordtep, so that we end up with one vector per original token.

只需要使用`PretrainedTransformerMismatchedIndexer`和`PretrainedTransformerMismatchedEmbedder`就可以使用任何 Hugging Face 的 transformers。

```python
# This pattern is typically used in cases where your input data is already
# tokenized, so we're showing that here.
text_tokens = ["This", "is", "some", "frandibulous", "text", "."]
tokens = [Token(x) for x in text_tokens]
print(tokens)

# We're using a very small transformer here so that it runs quickly in binder. You
# can change this to any transformer model name supported by Hugging Face.
transformer_model = "google/reformer-crime-and-punishment"

# Represents the list of word tokens with a sequences of wordpieces as determined
# by the transformer's tokenizer.  This actually results in a pretty complex data
# type, which you can see by running this.  It's complicated because we need to
# know how to combine the wordpieces back into words after running the
# transformer.
indexer = PretrainedTransformerMismatchedIndexer(model_name=transformer_model)

text_field = TextField(tokens, {"transformer": indexer})
text_field.index(Vocabulary())
token_tensor = text_field.as_tensor(text_field.get_padding_lengths())

# There are two key things to notice in this output.  First, there are two masks:
# `mask` is a word-level mask that gets used in the utility functions described in
# the last section of this chapter.  `wordpiece_mask` gets used by the `Embedder`
# itself.  Second, there is an `offsets` tensor that gives start and end wordpiece
# indices for the original tokens.  In the embedder, we grab these, average all of
# the wordpieces for each token, and return the result.
print("Indexed tensors:", token_tensor)

embedding = PretrainedTransformerMismatchedEmbedder(model_name=transformer_model)

embedder = BasicTextFieldEmbedder(token_embedders={"transformer": embedding})

tensor_dict = text_field.batch_tensors([token_tensor])
embedded_tokens = embedder(tensor_dict)
print("Embedded tokens size:", embedded_tokens.size())
print("Embedded tokens:", embedded_tokens)
```

#### 扩散标签到wordpiece

没有直接的函数可以实现这个功能，但是直接自己写个方法，把word的label转换为wordpiece的labels也比较简单。做完之后，就可以用`PretrainedTransformerMismatchedIndexer`和`PretrainedTransformerMismatchedEmbedder`来处理word-piece level的数据。

另一种选择是为非初始单词提供空标签，并为任何带有空标签的单词掩盖损失计算。但是，从建模角度来看，这是有问题的，因为它打破了CRF局部性假设（您不会从CRF转换概率中获得任何用处），并且通常会使建模更加困难。我们不推荐这种方法。

### padding和mask

由于要batch computation，所以不一样长的序列需要padding，allennlp里面用的是`text_field.get_padding_lengths()`。`collate_function`方法会在batch中找到最长的dimension，然后将这个最大值传给`text_field.as_tensor()`，因此每个相同维度的tensor在被创建之前就会被batched。

mask的过程在`TextFieldEmbedder`中，但是也要确保模型代码做了对应的masking computation。

`allennlp.nn.util`中提供了很多masked版本的pytorch的工具类方法，比如`masked_softmax`和`masked_log_softmax`和`masked_topk`

### 使用TextField输出的TextFieldTensors

TextField会返回一个TextFieldTensors对象，是一个复杂的字典结构。

不要直接编写访问TextfieldTensors对象内部的代码。allennlp中有几个方法可以访问textfieldtensors。TextFieldEmbedder对象会把TextFieldTensors对象转换为每个输入token对应一个embedding。一般会通过mask，并获得token id来把他们转换为字符串。allennlp.nn.util中提供了get_text_field_mask和get_token_ids_from_text_field_tesors.

```python
# We're following the logic from the "Combining multiple TokenIndexers" example
# above.
tokenizer = SpacyTokenizer(pos_tags=True)

vocab = Vocabulary()
vocab.add_tokens_to_namespace(
    ["This", "is", "some", "text", "."], namespace="token_vocab"
)
vocab.add_tokens_to_namespace(
    ["T", "h", "i", "s", " ", "o", "m", "e", "t", "x", "."], namespace="character_vocab"
)
vocab.add_tokens_to_namespace(["DT", "VBZ", "NN", "."], namespace="pos_tag_vocab")

text = "This is some text."
text2 = "This is some text with more tokens."
tokens = tokenizer.tokenize(text)
tokens2 = tokenizer.tokenize(text2)
print("Tokens:", tokens)
print("Tokens 2:", tokens2)


# Represents each token with (1) an id from a vocabulary, (2) a sequence of
# characters, and (3) part of speech tag ids.
token_indexers = {
    "tokens": SingleIdTokenIndexer(namespace="token_vocab"),
    "token_characters": TokenCharactersIndexer(namespace="character_vocab"),
    "pos_tags": SingleIdTokenIndexer(namespace="pos_tag_vocab", feature_name="tag_"),
}

text_field = TextField(tokens, token_indexers)
text_field.index(vocab)
text_field2 = TextField(tokens2, token_indexers)
text_field2.index(vocab)

# We're using the longer padding lengths here; we'd typically be relying on our
# collate function to figure out what the longest values are to use.
padding_lengths = text_field2.get_padding_lengths()
tensor_dict = text_field.as_tensor(padding_lengths)
tensor_dict2 = text_field2.as_tensor(padding_lengths)
print("Combined tensor dictionary:", tensor_dict)
print("Combined tensor dictionary 2:", tensor_dict2)

text_field_tensors = text_field.batch_tensors([tensor_dict, tensor_dict2])
print("Batched tensor dictionary:", text_field_tensors)

# We've seen plenty of examples of using a TextFieldEmbedder, so we'll just show
# the utility methods here.
mask = nn_util.get_text_field_mask(text_field_tensors)
print("Mask:", mask)
print("Mask size:", mask.size())
token_ids = nn_util.get_token_ids_from_text_field_tensors(text_field_tensors)
print("Token ids:", token_ids)

# We can also handle getting masks when you have lists of TextFields, but there's
# an important parameter that you need to pass, which we'll show here.  The
# difference in output that you see between here and above is just that there's an
# extra dimension in this output.  Where shapes used to be (batch_size=2, ...),
# now they are (batch_size=1, list_length=2, ...).
list_field = ListField([text_field, text_field2])
tensor_dict = list_field.as_tensor(list_field.get_padding_lengths())
text_field_tensors = list_field.batch_tensors([tensor_dict])
print("Batched tensors for ListField[TextField]:", text_field_tensors)

# The num_wrapping_dims argument tells get_text_field_mask how many nested lists
# there are around the TextField, which we need for our heuristics that guess
# which tensor to use when computing a mask.
mask = nn_util.get_text_field_mask(text_field_tensors, num_wrapping_dims=1)
print("Mask:", mask)
print("Mask:", mask.size())
```

输出：

```
Tokens: [This, is, some, text, .]
Tokens 2: [This, is, some, text, with, more, tokens, .]
Combined tensor dictionary: {'tokens': {'tokens': tensor([2, 3, 4, 5, 6, 0, 0, 0])}, 'token_characters': {'token_characters': tensor([[ 2,  3,  4,  5,  0,  0],
        [ 4,  5,  0,  0,  0,  0],
        [ 5,  7,  8,  9,  0,  0],
        [10,  9, 11, 10,  0,  0],
        [12,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0],
        [ 0,  0,  0,  0,  0,  0]])}, 'pos_tags': {'tokens': tensor([2, 3, 2, 4, 5, 0, 0, 0])}}
Combined tensor dictionary 2: {'tokens': {'tokens': tensor([2, 3, 4, 5, 1, 1, 1, 6])}, 'token_characters': {'token_characters': tensor([[ 2,  3,  4,  5,  0,  0],
        [ 4,  5,  0,  0,  0,  0],
        [ 5,  7,  8,  9,  0,  0],
        [10,  9, 11, 10,  0,  0],
        [ 1,  4, 10,  3,  0,  0],
        [ 8,  7,  1,  9,  0,  0],
        [10,  7,  1,  9,  1,  5],
        [12,  0,  0,  0,  0,  0]])}, 'pos_tags': {'tokens': tensor([2, 3, 2, 4, 1, 1, 1, 5])}}
Batched tensor dictionary: {'tokens': {'tokens': tensor([[2, 3, 4, 5, 6, 0, 0, 0],
        [2, 3, 4, 5, 1, 1, 1, 6]])}, 'token_characters': {'token_characters': tensor([[[ 2,  3,  4,  5,  0,  0],
         [ 4,  5,  0,  0,  0,  0],
         [ 5,  7,  8,  9,  0,  0],
         [10,  9, 11, 10,  0,  0],
         [12,  0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0,  0],
         [ 0,  0,  0,  0,  0,  0]],

        [[ 2,  3,  4,  5,  0,  0],
         [ 4,  5,  0,  0,  0,  0],
         [ 5,  7,  8,  9,  0,  0],
         [10,  9, 11, 10,  0,  0],
         [ 1,  4, 10,  3,  0,  0],
         [ 8,  7,  1,  9,  0,  0],
         [10,  7,  1,  9,  1,  5],
         [12,  0,  0,  0,  0,  0]]])}, 'pos_tags': {'tokens': tensor([[2, 3, 2, 4, 5, 0, 0, 0],
        [2, 3, 2, 4, 1, 1, 1, 5]])}}
Mask: tensor([[ True,  True,  True,  True,  True, False, False, False],
        [ True,  True,  True,  True,  True,  True,  True,  True]])
Mask size: torch.Size([2, 8])
Token ids: tensor([[2, 3, 4, 5, 6, 0, 0, 0],
        [2, 3, 4, 5, 1, 1, 1, 6]])
Batched tensors for ListField[TextField]: {'tokens': {'tokens': tensor([[[2, 3, 4, 5, 6, 0, 0, 0],
         [2, 3, 4, 5, 1, 1, 1, 6]]])}, 'token_characters': {'token_characters': tensor([[[[ 2,  3,  4,  5,  0,  0],
          [ 4,  5,  0,  0,  0,  0],
          [ 5,  7,  8,  9,  0,  0],
          [10,  9, 11, 10,  0,  0],
          [12,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0],
          [ 0,  0,  0,  0,  0,  0]],

         [[ 2,  3,  4,  5,  0,  0],
          [ 4,  5,  0,  0,  0,  0],
          [ 5,  7,  8,  9,  0,  0],
          [10,  9, 11, 10,  0,  0],
          [ 1,  4, 10,  3,  0,  0],
          [ 8,  7,  1,  9,  0,  0],
          [10,  7,  1,  9,  1,  5],
          [12,  0,  0,  0,  0,  0]]]])}, 'pos_tags': {'tokens': tensor([[[2, 3, 2, 4, 5, 0, 0, 0],
         [2, 3, 2, 4, 1, 1, 1, 5]]])}}
Mask: tensor([[[ True,  True,  True,  True,  True, False, False, False],
         [ True,  True,  True,  True,  True,  True,  True,  True]]])
Mask: torch.Size([1, 2, 8])
```

