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