---
title: BERT-预训练源码理解
comments: true
mathjax: false
date: 2021-02-23 16:11:01
tags: [BERT, Pretrain, DeepLearning]
categories: MachineLearning
urlname: bert-create-pretrain-data-analysis
---

<meta name="referrer" content="no-referrer" />

{% note info %}
BERT 作为一个里程碑式的预训练模型，很多时候我们都是直接用训练好的 model 直接 fine-tune，对它的理解只停留在 MLM 和 NSP 上。后续的很多 SOTA 模型都是在 BERT 的基础上发展来，比如 ALBERT、RoBERTa、XLNet 之类。

这里对 BERT 创建预训练数据的源码：`create_pretraining_data.py`和`run_pretrain.py`进行分析和理解。

[当前 BERT 对应的 commit](https://github.com/google-research/bert/tree/eedf5716ce1268e56f0a50264a88cafad334ac61)

部分参考[预训练模型-BERT预训练源码解读笔记](https://carlos9310.github.io/2019/09/30/pre-trained-bert/)

{% endnote%}

<!--more-->

## 原始数据格式

1. 每行一句话，每个文档中间用空格分开。
2. 可以输入多个文件，也可以输出多个 tfrecord 文件
3. 参考样例可以看 bert 中附带的`sample_text.txt`

## 数据生成 tfrecord

主要为`create_pretraining_data.py`分析。

### 生成 tfrecord 命令

```bash
python create_pretraining_data.py \
  --input_file=./sample_text.txt \
  --output_file=/tmp/tf_examples.tfrecord \
  --vocab_file=$BERT_BASE_DIR/vocab.txt \
  --do_lower_case=True \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --masked_lm_prob=0.15 \
  --random_seed=12345 \
  --dupe_factor=5
```

这里并不是所有的参数，所有的参数和说明可以看`create_pretraining_data.py`中的[`flags.DEFINE_string`](#参数说明)。

### 参数说明

```python
flags.DEFINE_string("input_file", None,
                    "Input raw text file (or comma-separated list of files).")

flags.DEFINE_string(
    "output_file", None,
    "Output TF example file (or comma-separated list of files).")

flags.DEFINE_string("vocab_file", None,
                    "The vocabulary file that the BERT model was trained on.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_bool(
    "do_whole_word_mask", False,
    "Whether to use whole word masking rather than per-WordPiece masking.")

flags.DEFINE_integer("max_seq_length", 128, "Maximum sequence length.")

flags.DEFINE_integer("max_predictions_per_seq", 20,
                     "Maximum number of masked LM predictions per sequence.")

flags.DEFINE_integer("random_seed", 12345, "Random seed for data generation.")

flags.DEFINE_integer(
    "dupe_factor", 10,
    "Number of times to duplicate the input data (with different masks).")

flags.DEFINE_float("masked_lm_prob", 0.15, "Masked LM probability.")

flags.DEFINE_float(
    "short_seq_prob", 0.1,
    "Probability of creating sequences which are shorter than the "
    "maximum length.")
```

其中：

- input_file：是输入文件，就按照上面说的格式。如果有多个文件可以用逗号分开
- output_file：是输出的 tfrecord 文件，多个可以用逗号分开
- vocab_file：是词表，可以直接用 bert 模型里面的 vocab。如果重新训练也可以用自己的词表
- do_lower_case：是表示是否把输入小写
- do_whole_word_mask：表示是否要进行整个单词的 mask，而不是 word piece 的 mask。word piece 会把单词拆分，非单词首部的用##开头
- max_seq_length：表示拼接后的句子对组成的序列中包含 Wordpiece 级别的 token 数的上限，超过部分，需将较长的句子进行首尾截断
- max_predictions_per_seq：表示每个序列中需要预测的 token 的上限
- masked_lm_prob：表示生成的序列中被 masked 的 token 占总 token 数的比例。(这里的 masked 是广义的 mask，即将选中的 token 替换成[mask]或保持原词汇或随机替换成词表中的另一个词)，且有如下关系`max_predictions_per_seq = max_seq_length * masked_lm_prob`
- random_seed：用于复现结果，每次保持一样
- dupe_factor：对输入使用不同的 mask 的次数，会重复创建 TrainingInstance
- masked_lm_prob：mask LM 的概率，一般按照上面的公式确定
- short_seq_prob：会按照这个概率创建比最大长度短的句子

### main of pretraining data

```python
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  tokenizer = tokenization.FullTokenizer(
      vocab_file=FLAGS.vocab_file, do_lower_case=FLAGS.do_lower_case)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Reading from input files ***")
  for input_file in input_files:
    tf.logging.info("  %s", input_file)

  # 这个rng会一直用下去，作为参数传递
  rng = random.Random(FLAGS.random_seed)
  # 得到的是一个一维instance列表，每个instance是一行预处理得到
  instances = create_training_instances(
      input_files, tokenizer, FLAGS.max_seq_length, FLAGS.dupe_factor,
      FLAGS.short_seq_prob, FLAGS.masked_lm_prob, FLAGS.max_predictions_per_seq,
      rng)

  # 所以输出文件也是可以为多个，用逗号分割
  output_files = FLAGS.output_file.split(",")
  tf.logging.info("*** Writing to output files ***")
  for output_file in output_files:
    tf.logging.info("  %s", output_file)

  # 写入文件
  write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length,
                                  FLAGS.max_predictions_per_seq, output_files)
```

可以看到，这个模块的流程大概是：

1. 创建 tokenizer，使用到了 vocab 和 do_lower_case 这两个参数
2. 将 input files 整理到数组中
3. 创建 training instances，见[create_training_instances](#create_training_instances)
4. 将生成的每一个 TrainingInstance 对象依此转成 tf.train.Example 对象后
5. 将上述生成的对象序列化到.tfrecord 格式的文件中。最终生成的.tfrecord 格式的文件是 BERT 预训练时的数据源，见[write_instance_to_example_files](#write_instance_to_example_files)

下面先看如何[创建 training instances](#create_training_instances)

### create_training_instances

直接在代码上写注释了

```python
def create_training_instances(input_files, tokenizer, max_seq_length,
                              dupe_factor, short_seq_prob, masked_lm_prob,
                              max_predictions_per_seq, rng):
  """Create `TrainingInstance`s from raw text."""
  all_documents = [[]]

  # Input file format:
  # (1) One sentence per line. These should ideally be actual sentences, not
  # entire paragraphs or arbitrary spans of text. (Because we use the
  # sentence boundaries for the "next sentence prediction" task).
  # (2) Blank lines between documents. Document boundaries are needed so
  # that the "next sentence prediction" task doesn't span between documents.

  # 读取文件并保存在all_documents二维列表中。
  # 每个文件按行读取，每读一行，转换成Unicode，如果没有了就break
  # 如果读到空行，就再all_documents中加入一个list表示下一个文件的开始。
  # 每行使用tokenizer进行tokenize，加到最后一个文档中。
  # 最终形成如下形式的all_documents:
  # [[[d1_s1],[d1_s2],…,[d2_sn]],…,[d2_sm]],…,[[dk_s1],…,[dk_sz]]]。
  # 上述表示一个语料中有k个文档，第一个文档有n句话，第二个文档有m句话，
  # 第k个文档有z句话，d1_s1表示第一个文档中的第一句话被分割成wordpiece级别的list。
  for input_file in input_files:
    with tf.gfile.GFile(input_file, "r") as reader:
      while True:
        line = tokenization.convert_to_unicode(reader.readline())
        if not line:
          break
        line = line.strip()

        # Empty lines are used as document delimiters
        if not line:
          all_documents.append([])
        tokens = tokenizer.tokenize(line)
        if tokens:
          all_documents[-1].append(tokens)

  # Remove empty documents
  # 过滤空文档，并且随机打乱文档顺序
  all_documents = [x for x in all_documents if x]
  rng.shuffle(all_documents)

  vocab_words = list(tokenizer.vocab.keys())
  instances = []
  # 重复dupe_factor次，为每个文档创建创建instances列表
  for _ in range(dupe_factor):
    # 对all_documents中的每一个文档生成由TrainingInstance对象组成的instances列表
    # 即create_instances_from_document，并拼接(extend)所有的instances到一个instances中
    for document_index in range(len(all_documents)):
      instances.extend(
          # 一个文档是一个instances列表，里面每个instance对象是一行预处理的结果
          create_instances_from_document(
              all_documents, document_index, max_seq_length, short_seq_prob,
              masked_lm_prob, max_predictions_per_seq, vocab_words, rng))

  # 随机打乱instances并返回，这时候所有的instance对象都在一个列表里
  rng.shuffle(instances)
  return instances
```

这里引申出一个问题，怎么从文档生成 TrainingInstance 对象组成的 instances 列表

见[create_instances_from_document](#create_instances_from_document)

debug 截图如下所示

![BOBnSQ](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/BOBnSQ.png)

### create_instances_from_document

同样是注释的形式

```python
def create_instances_from_document(
    all_documents, document_index, max_seq_length, short_seq_prob,
    masked_lm_prob, max_predictions_per_seq, vocab_words, rng):
  """Creates `TrainingInstance`s for a single document."""
  # 当前文档
  document = all_documents[document_index]

  # Account for [CLS], [SEP], [SEP]
  max_num_tokens = max_seq_length - 3

  # We *usually* want to fill up the entire sequence since we are padding
  # to `max_seq_length` anyways, so short sequences are generally wasted
  # computation. However, we *sometimes*
  # (i.e., short_seq_prob == 0.1 == 10% of the time) want to use shorter
  # sequences to minimize the mismatch between pre-training and fine-tuning.
  # The `target_seq_length` is just a rough target however, whereas
  # `max_seq_length` is a hard limit.
  # 通过short_seq_prob随机生成小于max_num_tokens的短句
  target_seq_length = max_num_tokens
  if rng.random() < short_seq_prob:
    target_seq_length = rng.randint(2, max_num_tokens)

  # We DON'T just concatenate all of the tokens from a document into a long
  # sequence and choose an arbitrary split point because this would make the
  # next sentence prediction task too easy. Instead, we split the input into
  # segments "A" and "B" based on the actual "sentences" provided by the user
  # input.
  instances = []
  current_chunk = []
  current_length = 0
  i = 0
  while i < len(document):
    # document是二维的，document[i]就是其中的一句话，被tokenize之后的wordpiece
    segment = document[i]
    # chunk一直保存当前的segment
    current_chunk.append(segment)
    current_length += len(segment)
    # 只有当前chunk保存的token数达到了目标序列长度（不一定是max_num_tokens)，
    # 或者是文档的最后一行了，才进入这个if
    # 否则就直接i+1，继续把下一个segment加入到current_chunk中
    # 所以current_chunk有可能是一句话，也有可能使得多句话
    # 同理，segment A和segment B也是有可能一句，也可能多句
    if i == len(document) - 1 or current_length >= target_seq_length:
      # 当前chunk非空就得进，防止chunk被置空之后没有新的segment加进去
      if current_chunk:
        # `a_end` is how many segments from `current_chunk` go into the `A`
        # (first) sentence.
        # a_end是用于确定把current_chunk中的多少token分给segment A的
        a_end = 1
        if len(current_chunk) >= 2:
          a_end = rng.randint(1, len(current_chunk) - 1)

        tokens_a = []
        for j in range(a_end):
          tokens_a.extend(current_chunk[j])

        # 现在开始确定segment B，有两种方式，一种是random_next，
        # 另一种是依次拼接current_chunk中剩下的token
        tokens_b = []
        # Random next
        is_random_next = False
        # 如果current_chunk长度为1，那么就肯定被分给A，segment B只能random next
        # random next的概率是50%
        # 此时segment B的长度就被确定为：target_seq_length - len(tokens_a)，
        # 因为A和B最后要拼在一起训练
        if len(current_chunk) == 1 or rng.random() < 0.5:
          is_random_next = True
          target_b_length = target_seq_length - len(tokens_a)

          # This should rarely go for more than one iteration for large
          # corpora. However, just to be careful, we try to make sure that
          # the random document is not the same as the document
          # we're processing.
          # 确保random next的不是当前文档
          for _ in range(10):
            random_document_index = rng.randint(0, len(all_documents) - 1)
            if random_document_index != document_index:
              break

          # 随机确定segment B在random doc中的开始位置
          random_document = all_documents[random_document_index]
          random_start = rng.randint(0, len(random_document) - 1)
          for j in range(random_start, len(random_document)):
            tokens_b.extend(random_document[j])
            if len(tokens_b) >= target_b_length:
              break
          # We didn't actually use these segments so we "put them back" so
          # they don't go to waste.
          # 剩下的没有用上的segment，直接回退回去，不能浪费，i控制的是document[i]
          num_unused_segments = len(current_chunk) - a_end
          i -= num_unused_segments
        # Actual next，segment B直接就是current chunk中剩下的内容
        else:
          is_random_next = False
          for j in range(a_end, len(current_chunk)):
            tokens_b.extend(current_chunk[j])
        # 确保segment A和B拼起来不会超过num_tokens
        # 每次哪个长去掉哪个，并且按照50%的几率从头部去掉or从尾部去掉，直到符合要求
        truncate_seq_pair(tokens_a, tokens_b, max_num_tokens, rng)

        assert len(tokens_a) >= 1
        assert len(tokens_b) >= 1

        # 最终拼接后的两个segments形成的tokens的基础上做mask操作，
        # 生成masked LM任务需要的tokens形式
        tokens = []
        segment_ids = []
        # token和segment ids数量相同，segment A对应的ids是0，B对应的是1
        tokens.append("[CLS]")
        segment_ids.append(0)
        for token in tokens_a:
          tokens.append(token)
          segment_ids.append(0)

        # segment A和B之间用[SEP]分开
        tokens.append("[SEP]")
        segment_ids.append(0)

        for token in tokens_b:
          tokens.append(token)
          segment_ids.append(1)
        # 最后也要加一个[SEP]
        tokens.append("[SEP]")
        segment_ids.append(1)

        # 创建Masked LM
        (tokens, masked_lm_positions,
         masked_lm_labels) = create_masked_lm_predictions(
             tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)

        # 一个pair(A, B)创建一个instance
        instance = TrainingInstance(
            tokens=tokens,
            segment_ids=segment_ids,
            is_random_next=is_random_next,
            masked_lm_positions=masked_lm_positions,
            masked_lm_labels=masked_lm_labels)
        # 每一行是一个instance，每个文档是instances列表，最后返回这个列表
        instances.append(instance)
      # 一个current_chunk创建一个instance，用完清空
      current_chunk = []
      current_length = 0
    i += 1

  return instances
```

debug 截图如下所示：

![LaGZ7B](https://cdn.jsdelivr.net/gh/HanielF/ImageRepo@main/blog/LaGZ7B.png)

上面主要是解决了 BERT 中的 NSP 问题，然后又引出了两个问题：

1. MLM 问题，就是那个[create_masked_lm_predictions()](#create_masked_lm_predictions)
2. 生成 TrainingInstance 问题，即[TrainingInstance()](#traininginstance)

### create_masked_lm_predictions

```python
def create_masked_lm_predictions(tokens, masked_lm_prob,
                                 max_predictions_per_seq, vocab_words, rng):
  """Creates the predictions for the masked LM objective.

  输入的tokens：['[CLS]', 'ancient', 'sage', '-', 'un', '##im', '##port', '##ant', ...]
  输入的rng: <random.Random object at 0x7fbcdc9e8020>
  """

  cand_indexes = []
  # 这个for循环的作用是创建candidate indexes列表，如果要whole word mask，需要进行拼接处理
  # wordpiece 会把一些词拆成多个，单词的第一个word piece没有任何标记，
  # 后续的word piece会在开头加上##标记
  for (i, token) in enumerate(tokens):
    if token == "[CLS]" or token == "[SEP]":
      continue
    # Whole Word Masking means that if we mask all of the wordpieces
    # corresponding to an original word. When a word has been split into
    # WordPieces, the first token does not have any marker and any subsequence
    # tokens are prefixed with ##. So whenever we see the ## token, we
    # append it to the previous set of word indexes.
    #
    # Note that Whole Word Masking does *not* change the training code
    # at all -- we still predict each WordPiece independently, softmaxed
    # over the entire vocabulary.
    if (FLAGS.do_whole_word_mask and len(cand_indexes) >= 1 and
        token.startswith("##")):
      cand_indexes[-1].append(i)
    else:
      cand_indexes.append([i])

  # 打乱indexs
  rng.shuffle(cand_indexes)

  # output_tokens 是 tokens输入的copy
  output_tokens = list(tokens)

  # mask掉的token数量不可以超过max_predictions_per_seq
  num_to_predict = min(max_predictions_per_seq,
                       max(1, int(round(len(tokens) * masked_lm_prob))))

  masked_lms = []
  covered_indexes = set()
  for index_set in cand_indexes:
    if len(masked_lms) >= num_to_predict:
      break
    # If adding a whole-word mask would exceed the maximum number of
    # predictions, then just skip this candidate.
    if len(masked_lms) + len(index_set) > num_to_predict:
      continue
    # 没看懂下面这段什么意思index不应该都是唯一的吗，为什么可能index in covered_indexes == True的情况
    is_any_index_covered = False
    for index in index_set:
      if index in covered_indexes:
        is_any_index_covered = True
        break
    if is_any_index_covered:
      continue

    # 针对whole word mask，
    for index in index_set:
      covered_indexes.add(index)

      masked_token = None
      # 80% of the time, replace with [MASK]
      if rng.random() < 0.8:
        masked_token = "[MASK]"
      else:
        # 10% of the time, keep original
        if rng.random() < 0.5:
          masked_token = tokens[index]
        # 10% of the time, replace with random word
        else:
          masked_token = vocab_words[rng.randint(0, len(vocab_words) - 1)]

      output_tokens[index] = masked_token

      # 为这个token创建MaskedLmInstance
      # 声明：MaskedLmInstance = collections.namedtuple("MaskedLmInstance", ["index", "label"])
      # 类型namedtuple: Returns a new subclass of tuple with named fields.
      # 用于创建position embedding和real label
      masked_lms.append(MaskedLmInstance(index=index, label=tokens[index]))
  assert len(masked_lms) <= num_to_predict
  # 因为之前对cand_indexes进行了shuffle，所以返回前要进行sort
  masked_lms = sorted(masked_lms, key=lambda x: x.index)

  masked_lm_positions = []
  masked_lm_labels = []
  for p in masked_lms:
    masked_lm_positions.append(p.index)
    masked_lm_labels.append(p.label)

  return (output_tokens, masked_lm_positions, masked_lm_labels)
```

最后返回给`create_instances_from_document`中的

```python
(tokens, masked_lm_positions,
  masked_lm_labels) = create_masked_lm_predictions(
      tokens, masked_lm_prob, max_predictions_per_seq, vocab_words, rng)
```

### TrainingInstance

在结束 Masked LM 之后就可以来创建 TrainingInstance。

在`create_instances_from_document`中调用：

```python
instance = TrainingInstance(
    tokens=tokens,
    segment_ids=segment_ids,
    is_random_next=is_random_next,
    masked_lm_positions=masked_lm_positions,
    masked_lm_labels=masked_lm_labels)
instances.append(instance)
```

TrainingInstance 类别源码：

```python
class TrainingInstance(object):
  """A single training instance (sentence pair)."""

  def __init__(self, tokens, segment_ids, masked_lm_positions, masked_lm_labels,
               is_random_next):
    self.tokens = tokens
    self.segment_ids = segment_ids
    self.is_random_next = is_random_next
    self.masked_lm_positions = masked_lm_positions
    self.masked_lm_labels = masked_lm_labels

  # 用于print函数调用的，一般都是return一个什么东西
  def __str__(self):
    s = ""
    s += "tokens: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.tokens]))
    s += "segment_ids: %s\n" % (" ".join([str(x) for x in self.segment_ids]))
    s += "is_random_next: %s\n" % self.is_random_next
    s += "masked_lm_positions: %s\n" % (" ".join(
        [str(x) for x in self.masked_lm_positions]))
    s += "masked_lm_labels: %s\n" % (" ".join(
        [tokenization.printable_text(x) for x in self.masked_lm_labels]))
    s += "\n"
    return s

  # __str__()用于显示给用户，而__repr__()用于显示给开发人员
  def __repr__(self):
    return self.__str__()

```

### write_instance_to_example_files

返回到[main 函数](#main-of-pretraining-data)中可以看到，还需要把 TrainingInstance 对象写入到输出文件中，即
`write_instance_to_example_files(instances, tokenizer, FLAGS.max_seq_length, FLAGS.max_predictions_per_seq, output_files)`那段。

```python
def write_instance_to_example_files(instances, tokenizer, max_seq_length,
                                    max_predictions_per_seq, output_files):
  """Create TF example files from `TrainingInstance`s."""
  writers = []
  for output_file in output_files:
    writers.append(tf.python_io.TFRecordWriter(output_file))

  writer_index = 0

  total_written = 0
  for (inst_index, instance) in enumerate(instances):
    input_ids = tokenizer.convert_tokens_to_ids(instance.tokens)
    input_mask = [1] * len(input_ids)
    segment_ids = list(instance.segment_ids)
    assert len(input_ids) <= max_seq_length

    # 保证各个数据的长度都是max_seq_length，不足补0
    while len(input_ids) < max_seq_length:
      input_ids.append(0)
      input_mask.append(0)
      segment_ids.append(0)

    assert len(input_ids) == max_seq_length
    assert len(input_mask) == max_seq_length
    assert len(segment_ids) == max_seq_length

    masked_lm_positions = list(instance.masked_lm_positions)
    masked_lm_ids = tokenizer.convert_tokens_to_ids(instance.masked_lm_labels)
    # masked_lm_weights和input_mask 的作用一样，标注masked_lm_ids
    # 和 masked_lm_positions 哪些是真实值，哪些是补全值
    masked_lm_weights = [1.0] * len(masked_lm_ids)

    while len(masked_lm_positions) < max_predictions_per_seq:
      masked_lm_positions.append(0)
      masked_lm_ids.append(0)
      masked_lm_weights.append(0.0)

    next_sentence_label = 1 if instance.is_random_next else 0

    # 关于创建feature的函数
    # def create_int_feature(values):
    #   feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
    #   return feature
    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(input_ids)
    features["input_mask"] = create_int_feature(input_mask)
    features["segment_ids"] = create_int_feature(segment_ids)
    features["masked_lm_positions"] = create_int_feature(masked_lm_positions)
    features["masked_lm_ids"] = create_int_feature(masked_lm_ids)
    features["masked_lm_weights"] = create_float_feature(masked_lm_weights)
    features["next_sentence_labels"] = create_int_feature([next_sentence_label])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    # 保存为tf.train.Example(features字典[key: 特征, value: tf.train.Feature对象])保存
    writers[writer_index].write(tf_example.SerializeToString())
    writer_index = (writer_index + 1) % len(writers)

    total_written += 1

    # 打印前20个样本
    if inst_index < 20:
      tf.logging.info("*** Example ***")
      tf.logging.info("tokens: %s" % " ".join(
          [tokenization.printable_text(x) for x in instance.tokens]))

      for feature_name in features.keys():
        feature = features[feature_name]
        values = []
        if feature.int64_list.value:
          values = feature.int64_list.value
        elif feature.float_list.value:
          values = feature.float_list.value
        tf.logging.info(
            "%s: %s" % (feature_name, " ".join([str(x) for x in values])))

  for writer in writers:
    writer.close()

  tf.logging.info("Wrote %d total instances", total_written)
```

## 预训练 run_pretrain

### 预训练命令

```bash
python run_pretraining.py \
  --input_file=/tmp/tf_examples.tfrecord \
  --output_dir=/tmp/pretraining_output \
  --do_train=True \
  --do_eval=True \
  --bert_config_file=$BERT_BASE_DIR/bert_config.json \
  --init_checkpoint=$BERT_BASE_DIR/bert_model.ckpt \
  --train_batch_size=32 \
  --max_seq_length=128 \
  --max_predictions_per_seq=20 \
  --num_train_steps=20 \
  --num_warmup_steps=10 \
  --learning_rate=2e-5
```

### 参数说明

```python
## Required parameters
flags.DEFINE_string(
    "bert_config_file", None,
    "The config json file corresponding to the pre-trained BERT model. "
    "This specifies the model architecture.")

flags.DEFINE_string(
    "input_file", None,
    "Input TF example files (can be a glob or comma separated).")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

## Other parameters
flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_integer(
    "max_seq_length", 128,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded. Must match data generation.")

flags.DEFINE_integer(
    "max_predictions_per_seq", 20,
    "Maximum number of masked LM predictions per sequence. "
    "Must match data generation.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")

flags.DEFINE_integer("num_train_steps", 100000, "Number of training steps.")

flags.DEFINE_integer("num_warmup_steps", 10000, "Number of warmup steps.")

flags.DEFINE_integer("save_checkpoints_steps", 1000,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("iterations_per_loop", 1000,
                     "How many steps to make in each estimator call.")

flags.DEFINE_integer("max_eval_steps", 100, "Maximum number of eval steps.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")
```

其中：

- bert_config_file: 预训练模型的配置文件，直接使用对应大小 bert model 里的 config，或者自己调
- input_file: tfrecord 格式的文件
- output_dir: 预训练生成的模型文件路径，会自动创建
- init_checkpoint: 预训练模型的初始检查点，从头开始训练就不需要这个参数，fine-tune 的话就加载 bert 预训练的 ckpt
- max_seq_length: 最大序列长度，超过这个的会被阶段，不足的会补齐。要和数据生成 tfrecord 过程的一致。类似于 RNN 中的最大时间步，每次可动态调整。针对某一特定领域的语料，可在通用的语言模型的基础上，每次通过设置不同长度的专业领域的句子对微调语言模型，使最终生成的预训练的语言模型更适合某一特定领域
- train_batch_size: 训练的 mini batch 大小。如果出现内存不够的问题，那么调小 max_seq_length 或者 batch 大小就可以。
- do_train: 如果不训练只是预测 or 验证，可以设置为 false
- do_eval: 是否进行 eval 验证
- eval_batch_size: 验证的时候的 batch 大小
- learning_rate: Adam 的学习率，有论文表明越小越好，一般是 2e-5 级别
- num_train_steps: 训练的步数，如果是自己从头训练，这个步数要根据语料大小看，一般设置 w 级别。
- num_warmup_steps: warmup 步数，学习率从 0 逐渐增加到初始学习率所需的步数，以后的步数保持固定学习率。参考[github-issue](https://github.com/google-research/bert/issues/529)
- save_checkpoints_steps: 每隔多少步保存一次模型
- iterations_per_loop: 每次调用 estimator 的步数
- max_eval_steps: 最大的 eval 步数
- tup 相关参数看说明

### main of pretrain

在源码中写了注释说明。

```python
def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  # 可以选择不训练的，不改变模型参数
  if not FLAGS.do_train and not FLAGS.do_eval:
    raise ValueError("At least one of `do_train` or `do_eval` must be True.")

  # 读取模型参数
  bert_config = modeling.BertConfig.from_json_file(FLAGS.bert_config_file)

  tf.gfile.MakeDirs(FLAGS.output_dir)

  input_files = []
  for input_pattern in FLAGS.input_file.split(","):
    input_files.extend(tf.gfile.Glob(input_pattern))

  tf.logging.info("*** Input Files ***")
  for input_file in input_files:
    tf.logging.info("  %s" % input_file)

  # TPU相关
  tpu_cluster_resolver = None
  if FLAGS.use_tpu and FLAGS.tpu_name:
    tpu_cluster_resolver = tf.contrib.cluster_resolver.TPUClusterResolver(
        FLAGS.tpu_name, zone=FLAGS.tpu_zone, project=FLAGS.gcp_project)

  is_per_host = tf.contrib.tpu.InputPipelineConfig.PER_HOST_V2
  run_config = tf.contrib.tpu.RunConfig(
      cluster=tpu_cluster_resolver,
      master=FLAGS.master,
      model_dir=FLAGS.output_dir,
      save_checkpoints_steps=FLAGS.save_checkpoints_steps,
      tpu_config=tf.contrib.tpu.TPUConfig(
          iterations_per_loop=FLAGS.iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))

  # 正式创建模型，其实返回的是output_spec = tf.contrib.tpu.TPUEstimatorSpec()
  model_fn = model_fn_builder(
      bert_config=bert_config,
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=FLAGS.num_train_steps,
      num_warmup_steps=FLAGS.num_warmup_steps,
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = tf.contrib.tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size)

  if FLAGS.do_train:
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    # 从tfrecord解析出BERT的输入数据
    train_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=True)
    estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

  if FLAGS.do_eval:
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    # 如果不训练只是eval，输入也是一样的tfrecord
    eval_input_fn = input_fn_builder(
        input_files=input_files,
        max_seq_length=FLAGS.max_seq_length,
        max_predictions_per_seq=FLAGS.max_predictions_per_seq,
        is_training=False)

    # evaluate操作得到结果
    result = estimator.evaluate(
        input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)

    output_eval_file = os.path.join(FLAGS.output_dir, "eval_results.txt")
    with tf.gfile.GFile(output_eval_file, "w") as writer:
      tf.logging.info("***** Eval results *****")
      for key in sorted(result.keys()):
        tf.logging.info("  %s = %s", key, str(result[key]))
        writer.write("%s = %s\n" % (key, str(result[key])))
```

这里引出几个问题：

- 模型的创建: [model_fn_builder](#model_fn_builder)
- 数据的解析: [input_fn_builder](#input_fn_builder)
- 模型的训练：estimator.train
- 验证集的测试: estimator.evaluate

### input_fn_builder

从 tfrecord 中解析出 bert 的输入数据

```python
def input_fn_builder(input_files,
                     max_seq_length,
                     max_predictions_per_seq,
                     is_training,
                     num_cpu_threads=4):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""

  def input_fn(params):
    """The actual input function."""
    batch_size = params["batch_size"]

    name_to_features = {
        "input_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "input_mask": # input_mask=0表示是补齐的部分
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "segment_ids":
            tf.FixedLenFeature([max_seq_length], tf.int64),
        "masked_lm_positions":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_ids":
            tf.FixedLenFeature([max_predictions_per_seq], tf.int64),
        "masked_lm_weights":
            tf.FixedLenFeature([max_predictions_per_seq], tf.float32),
        "next_sentence_labels":
            tf.FixedLenFeature([1], tf.int64),
    }

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      # 得到dataset对象
      d = tf.data.Dataset.from_tensor_slices(tf.constant(input_files))
      # 如果 repeat 转换在 shuffle 转换之前应用，则迭代次数边界将变的不确定。
      # 也就是说，某些元素可以在其他元素出现之前重复一次。
      # 另一方面，如果在 repeat 转换之前应用 shuffle 转换，则在涉及 shuffle
      # 转换的内部状态初始化的每个迭代次数开始时性能可能会下降。
      # 换句话说，前者（在 shuffle 之前 repeat）提供了更好的性能，
      # 而后者（在 repeat 之前 shuffle）提供了更确定性的排序。
      d = d.repeat()
      d = d.shuffle(buffer_size=len(input_files))

      # `cycle_length` is the number of parallel files that get read.
      cycle_length = min(num_cpu_threads, len(input_files))

      # `sloppy` mode means that the interleaving is not exact. This adds
      # even more randomness to the training pipeline.
      # 这里的说明可以看https://tensorflow.juejin.im/performance/datasets_performance.html
      # sloppy设置为True，转换可能会偏离其确定性顺序，但是这样可以让训练数据更加随机化
      d = d.apply(
          tf.contrib.data.parallel_interleave(
              tf.data.TFRecordDataset,
              sloppy=is_training,
              cycle_length=cycle_length))
      d = d.shuffle(buffer_size=100)
    else:
      # evaluate模式就无所谓顺序和并行了
      d = tf.data.TFRecordDataset(input_files)
      # Since we evaluate for a fixed number of steps we don't want to encounter
      # out-of-range exceptions.
      d = d.repeat()

    # We must `drop_remainder` on training because the TPU requires fixed
    # size dimensions. For eval, we assume we are evaluating on the CPU or GPU
    # and we *don't* want to drop the remainder, otherwise we wont cover
    # every sample.
    d = d.apply(
        # 这个方法相当于先map，然后batch
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            num_parallel_batches=num_cpu_threads,
            drop_remainder=True))
    # 将解析后的值分成多组batch，作为模型的输入数据(model_fn中的features)。
    return d

  return input_fn
```

### model_fn_builder

用于构造 Estimator 使用的 model_fn。包含了特征提取、模型创建、计算损失、加载 checkpoint、计算 acc，并返回一个 EstimatorSpec。

定义好了`get_masked_lm_output`和`get_next_sentence_output`两个训练任务后，就可以写出训练过程，之后将训练集传入自动训练。

```python
def model_fn_builder(bert_config, init_checkpoint, learning_rate,
                     num_train_steps, num_warmup_steps, use_tpu,
                     use_one_hot_embeddings):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):  # pylint: disable=unused-argument
    """The `model_fn` for TPUEstimator."""

    tf.logging.info("*** Features ***")
    for name in sorted(features.keys()):
      tf.logging.info("  name = %s, shape = %s" % (name, features[name].shape))

    input_ids = features["input_ids"]
    input_mask = features["input_mask"]
    segment_ids = features["segment_ids"]
    masked_lm_positions = features["masked_lm_positions"]
    masked_lm_ids = features["masked_lm_ids"]
    masked_lm_weights = features["masked_lm_weights"]
    next_sentence_labels = features["next_sentence_labels"]

    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    # 重点是这里的创建BERTModel，看下一部分的模型代码
    model = modeling.BertModel(
        config=bert_config,
        is_training=is_training,
        input_ids=input_ids,
        input_mask=input_mask,
        token_type_ids=segment_ids,
        use_one_hot_embeddings=use_one_hot_embeddings)

    # 计算MaskedLM的损失
    (masked_lm_loss,
     masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
         bert_config, model.get_sequence_output(), model.get_embedding_table(),
         masked_lm_positions, masked_lm_ids, masked_lm_weights)

    # 计算NSP的损失
    (next_sentence_loss, next_sentence_example_loss,
     next_sentence_log_probs) = get_next_sentence_output(
         bert_config, model.get_pooled_output(), next_sentence_labels)

    # MLM的损失和NSP损失相加
    total_loss = masked_lm_loss + next_sentence_loss

    # 获得所有可训练的variable
    tvars = tf.trainable_variables()

    # 加载checkpoint
    initialized_variable_names = {}
    scaffold_fn = None
    if init_checkpoint:
      (assignment_map, initialized_variable_names
      ) = modeling.get_assignment_map_from_checkpoint(tvars, init_checkpoint)
      if use_tpu:

        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()

        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # 打印log
    tf.logging.info("**** Trainable Variables ****")
    for var in tvars:
      init_string = ""
      if var.name in initialized_variable_names:
        init_string = ", *INIT_FROM_CKPT*"
      tf.logging.info("  name = %s, shape = %s%s", var.name, var.shape,
                      init_string)

    # 训练过程，获得spec
    output_spec = None
    if mode == tf.estimator.ModeKeys.TRAIN:
      # 创建优化器optimizer
      train_op = optimization.create_optimizer(
          total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)

      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn)
    elif mode == tf.estimator.ModeKeys.EVAL:

      # 下面是计算损失和acc的函数
      def metric_fn(masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
                    masked_lm_weights, next_sentence_example_loss,
                    next_sentence_log_probs, next_sentence_labels):
        """Computes the loss and accuracy of the model."""
        # [batch_size*max_predictions_per_seq=640, vocab_size=30522]
        masked_lm_log_probs = tf.reshape(masked_lm_log_probs,
                                         [-1, masked_lm_log_probs.shape[-1]])
        masked_lm_predictions = tf.argmax(
            masked_lm_log_probs, axis=-1, output_type=tf.int32)
        # [batch_size*max_predictions_per_seq=640, ]
        masked_lm_example_loss = tf.reshape(masked_lm_example_loss, [-1])
        masked_lm_ids = tf.reshape(masked_lm_ids, [-1])
        masked_lm_weights = tf.reshape(masked_lm_weights, [-1])
        # tf.metrics.accuracy返回两个值，accuracy为到上一个batch为止的准确度，
        # update_op为更新本批次后的准确度。
        # masked_lm_weights用于标记哪些是真实值哪些是补全值
        masked_lm_accuracy = tf.metrics.accuracy(
            labels=masked_lm_ids,
            predictions=masked_lm_predictions,
            weights=masked_lm_weights)
        # 就是per_example_loss交叉熵损失
        masked_lm_mean_loss = tf.metrics.mean(
            values=masked_lm_example_loss, weights=masked_lm_weights)

        next_sentence_log_probs = tf.reshape(
            next_sentence_log_probs, [-1, next_sentence_log_probs.shape[-1]])
        next_sentence_predictions = tf.argmax(
            next_sentence_log_probs, axis=-1, output_type=tf.int32)
        next_sentence_labels = tf.reshape(next_sentence_labels, [-1])
        next_sentence_accuracy = tf.metrics.accuracy(
            labels=next_sentence_labels, predictions=next_sentence_predictions)
        next_sentence_mean_loss = tf.metrics.mean(
            values=next_sentence_example_loss)

        return {
            "masked_lm_accuracy": masked_lm_accuracy,
            "masked_lm_loss": masked_lm_mean_loss,
            "next_sentence_accuracy": next_sentence_accuracy,
            "next_sentence_loss": next_sentence_mean_loss,
        }

      eval_metrics = (metric_fn, [
          masked_lm_example_loss, masked_lm_log_probs, masked_lm_ids,
          masked_lm_weights, next_sentence_example_loss,
          next_sentence_log_probs, next_sentence_labels
      ])
      output_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=total_loss,
          eval_metrics=eval_metrics,
          scaffold_fn=scaffold_fn)
    else:
      raise ValueError("Only TRAIN and EVAL modes are supported: %s" % (mode))

    return output_spec

  return model_fn
```

基于上述搭建好的模型结构及相应的损失函数，在训练阶段，利用相应的优化器(AdamWeightDecayOptimizer)优化损失函数，使其减小，并保存不同训练步数对应的模型参数，直到跑完所有步数，从而确定最终的模型结构与参数。

从这里引出几个问题：

1. Bert 模型的创建，见[model = modeling.BertModel(···)](#bertmodel)
2. 计算 MaskedLM 的损失，见[(masked_lm_loss, masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output()](#get_masked_lm_output)
3. 计算 NSP 的损失，见[(next_sentence_loss, next_sentence_example_loss, next_sentence_log_probs) = get_next_sentence_output()](#get_next_sentence_output)
4. 创建优化器，用来更新模型(权重)参数，见[create_optimizer()](#create_optimizer)

### 文件说明

由于 BERT 在预训练中使用了 estimator 这种高级 API 形式，在训练完成后会自动生成 ckpt 格式的模型文件(结构和数据是分开的) 及可供 tensorboard 查看的事件文件。具体文件说明如下：

1. `checkpoint`: 记录了模型文件的路径信息列表，可以用来迅速查找最近一次的 ckpt 文件。(每个 ckpt 文件对应一个模型)其内容如下所示
   - model_checkpoint_path: "model.ckpt-20"
   - all_model_checkpoint_paths: "model.ckpt-0"
   - all_model_checkpoint_paths: "model.ckpt-20"
2. `events.out.tfevents.1570029823.04c93f97d224`：事件文件，tensorboard 可加载显示
3. `graph.pbtxt` : 以 Protobuffer 格式描述的模型结构文件(text 格式的图文件(.pbtext),二进制格式的图文件为(.pb))，记录了模型中所有的节点信息，内容大致如下：

```python
  node {
    name: "global_step/Initializer/zeros"
    op: "Const"
    attr {
      key: "_class"
      value {
        list {
          s: "loc:@global_step"
        }
      }
    }
    attr {
      key: "_output_shapes"
      value {
        list {
          shape {
          }
        }
      }
    }
    attr {
      key: "dtype"
      value {
        type: DT_INT64
      }
    }
    attr {
      key: "value"
      value {
        tensor {
          dtype: DT_INT64
          tensor_shape {
          }
          int64_val: 0
        }
      }
    }
  }
```

4. `model.ckpt-20.data-00000-of-00001` : 模型文件中的数据(the values of all variables)部分 (二进制文件)
5. `model.ckpt-20.index` : 模型文件中的映射表( Each key is a name of a tensor and its value is a serialized BundleEntryProto. Each BundleEntryProto describes the metadata of a tensor: which of the “data” files contains the content of a tensor, the offset into that file, checksum, some auxiliary data, etc.)部分 (二进制文件)
6. `model.ckpt-20.meta` : 模型文件中的(图)结构(由 GraphDef, SaverDef, MateInfoDef,SignatureDef,CollectionDef 等组成的 MetaGraphDef)部分 (二进制文件，内容和 graph.pbtxt 基本一样，其是一个序列化的 MetaGraphDef protocol buffer)

在评估阶段，直接加载训练好的模型结构与参数，对预测样本进行预测即可。

### BertModel

```python
class BertModel(object):
  """BERT model ("Bidirectional Encoder Representations from Transformers").

  Example usage:

  # python
  # Already been converted into WordPiece token ids
  input_ids = tf.constant([[31, 51, 99], [15, 5, 0]])
  input_mask = tf.constant([[1, 1, 1], [1, 1, 0]])
  token_type_ids = tf.constant([[0, 0, 1], [0, 2, 0]])

  config = modeling.BertConfig(vocab_size=32000, hidden_size=512,
    num_hidden_layers=8, num_attention_heads=6, intermediate_size=1024)

  model = modeling.BertModel(config=config, is_training=True,
    input_ids=input_ids, input_mask=input_mask, token_type_ids=token_type_ids)

  label_embeddings = tf.get_variable(...)
  pooled_output = model.get_pooled_output()
  logits = tf.matmul(pooled_output, label_embeddings)
  ...
  """

  def __init__(self,
               config,
               is_training,
               input_ids,
               input_mask=None,
               token_type_ids=None,
               use_one_hot_embeddings=False,
               scope=None):
    """Constructor for BertModel.

    Args:
      config: `BertConfig` instance.
      is_training: bool. true for training model, false for eval model. Controls
        whether dropout will be applied.
      input_ids: int32 Tensor of shape [batch_size, seq_length].
      input_mask: (optional) int32 Tensor of shape [batch_size, seq_length].
      token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      use_one_hot_embeddings: (optional) bool. Whether to use one-hot word
        embeddings or tf.embedding_lookup() for the word embeddings.
      scope: (optional) variable scope. Defaults to "bert".

    Raises:
      ValueError: The config is invalid or one of the input tensor shapes
        is invalid.
    """
    config = copy.deepcopy(config)
    if not is_training:
      config.hidden_dropout_prob = 0.0
      config.attention_probs_dropout_prob = 0.0

    # get_shape_list
    # Returns a list of the shape of tensor, preferring static dimensions.
    input_shape = get_shape_list(input_ids, expected_rank=2)
    batch_size = input_shape[0]
    seq_length = input_shape[1]

    if input_mask is None:
      input_mask = tf.ones(shape=[batch_size, seq_length], dtype=tf.int32)

    if token_type_ids is None:
      token_type_ids = tf.zeros(shape=[batch_size, seq_length], dtype=tf.int32)

    with tf.variable_scope(scope, default_name="bert"):
      with tf.variable_scope("embeddings"):
        # Perform embedding lookup on the word ids.
        # embedding_output 是input_ids对应的embedding输出
        # embedding_table就是整个的embedding table
        (self.embedding_output, self.embedding_table) = embedding_lookup(
            input_ids=input_ids,
            vocab_size=config.vocab_size,
            embedding_size=config.hidden_size,
            initializer_range=config.initializer_range,
            word_embedding_name="word_embeddings",
            use_one_hot_embeddings=use_one_hot_embeddings)

        # Add positional embeddings and token type embeddings, then layer
        # normalize and perform dropout.
        # 上面得到的embedding输出，先加上token_type_embedding，
        # 然后加上position_embedding。shape相同，对应位置相加
        # token_type_embedding和position_embedding都是通过构建一个look up table得到
        # 最后进行layer_norm_and_dropout
        self.embedding_output = embedding_postprocessor(
            input_tensor=self.embedding_output,
            use_token_type=True,
            token_type_ids=token_type_ids,
            token_type_vocab_size=config.type_vocab_size,
            token_type_embedding_name="token_type_embeddings",
            use_position_embeddings=True,
            position_embedding_name="position_embeddings",
            initializer_range=config.initializer_range,
            max_position_embeddings=config.max_position_embeddings,
            dropout_prob=config.hidden_dropout_prob)

      with tf.variable_scope("encoder"):
        # This converts a 2D mask of shape [batch_size, seq_length] to a 3D
        # mask of shape [batch_size, seq_length, seq_length] which is used
        # for the attention scores.
        # input_ids, input_mask: (32, 128)
        # attention_mask: (32, 128, 128)
        attention_mask = create_attention_mask_from_input_mask(
            input_ids, input_mask)

        # Run the stacked transformer.
        # `sequence_output` shape = [batch_size, seq_length, hidden_size].
        # encoder部分，由num_hidden_layers(12)个transformer encoder组成的
        self.all_encoder_layers = transformer_model(
            input_tensor=self.embedding_output,
            attention_mask=attention_mask,
            hidden_size=config.hidden_size,
            num_hidden_layers=config.num_hidden_layers,
            num_attention_heads=config.num_attention_heads,
            intermediate_size=config.intermediate_size,
            intermediate_act_fn=get_activation(config.hidden_act),
            hidden_dropout_prob=config.hidden_dropout_prob,
            attention_probs_dropout_prob=config.attention_probs_dropout_prob,
            initializer_range=config.initializer_range,
            do_return_all_layers=True)

      self.sequence_output = self.all_encoder_layers[-1]
      # The "pooler" converts the encoded sequence tensor of shape
      # [batch_size, seq_length, hidden_size] to a tensor of shape
      # [batch_size, hidden_size]. This is necessary for segment-level
      # (or segment-pair-level) classification tasks where we need a fixed
      # dimensional representation of the segment.
      # 两种输出，一种是最后一层transformer encoder的sequence_output，
      # token级别的embedding，用于masked LM任务训练
      # 另一种输出是取sequence_output的第一个token，然后接一个带有tanh的全连接层最为输出，
      # 句子级别的embedding，用于NSP任务的训练
      with tf.variable_scope("pooler"):
        # We "pool" the model by simply taking the hidden state corresponding
        # to the first token. We assume that this has been pre-trained
        first_token_tensor = tf.squeeze(self.sequence_output[:, 0:1, :], axis=1)
        self.pooled_output = tf.layers.dense(
            first_token_tensor,
            config.hidden_size,
            activation=tf.tanh,
            kernel_initializer=create_initializer(config.initializer_range))

  def get_pooled_output(self):
    return self.pooled_output

  def get_sequence_output(self):
    """Gets final hidden layer of encoder.

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the final hidden of the transformer encoder.
    """
    return self.sequence_output

  def get_all_encoder_layers(self):
    return self.all_encoder_layers

  def get_embedding_output(self):
    """Gets output of the embedding lookup (i.e., input to the transformer).

    Returns:
      float Tensor of shape [batch_size, seq_length, hidden_size] corresponding
      to the output of the embedding layer, after summing the word
      embeddings with the positional embeddings and the token type embeddings,
      then performing layer normalization. This is the input to the transformer.
    """
    return self.embedding_output

  def get_embedding_table(self):
    return self.embedding_table
```

这里引出几个部分：

1. embedding 表的构建，即[embedding_lookup](#embedding_lookup)
2. 在 word_embeddings 的基础上增加 segment_id 和 position 信息，最后将叠加后 embedding 分别进行 layer_norm，batch_norm 和 dropout 操作。见[embedding_postprocessor](#embedding_postprocessor)
3. transformer 模型的构建，即[transformer_model](#transformer_model)
4. attention 自注意力层的构建，即[attention_layer](#attention_layer)

### embedding_lookup

构建一个 embedding lookup 表，用于生成每个 token 的表示，同时返回 input_ids 对应的 embedding。

这里的 embedding 只包括 word_embedding，token embedding 和 position embedding 在 embedding_postprocessor 中处理

```python
def embedding_lookup(input_ids,
                     vocab_size,
                     embedding_size=128,
                     initializer_range=0.02,
                     word_embedding_name="word_embeddings",
                     use_one_hot_embeddings=False):
  """Looks up words embeddings for id tensor.

  Args:
    input_ids: int32 Tensor of shape [batch_size, seq_length] containing word
      ids.
    vocab_size: int. Size of the embedding vocabulary.
    embedding_size: int. Width of the word embeddings.
    initializer_range: float. Embedding initialization range.
    word_embedding_name: string. Name of the embedding table.
    use_one_hot_embeddings: bool. If True, use one-hot method for word
      embeddings. If False, use `tf.gather()`.

  Returns:
    float Tensor of shape [batch_size, seq_length, embedding_size].
  """
  # This function assumes that the input is of shape [batch_size, seq_length,
  # num_inputs].
  #
  # If the input is a 2D tensor of shape [batch_size, seq_length], we
  # reshape to [batch_size, seq_length, 1].
  if input_ids.shape.ndims == 2:
    input_ids = tf.expand_dims(input_ids, axis=[-1])

  embedding_table = tf.get_variable(
      name=word_embedding_name,
      shape=[vocab_size, embedding_size],
      initializer=create_initializer(initializer_range))

  # flat_input_ids shape: (4096, )
  flat_input_ids = tf.reshape(input_ids, [-1])
  if use_one_hot_embeddings:
    one_hot_input_ids = tf.one_hot(flat_input_ids, depth=vocab_size)
    output = tf.matmul(one_hot_input_ids, embedding_table)
  else:
    output = tf.gather(embedding_table, flat_input_ids)

  # input_shape = [32, 128, 1]
  input_shape = get_shape_list(input_ids)

  # output shape: [4096=32*128, 512] -> [batch, seq_len, embedding_size=512]
  output = tf.reshape(output,
                      input_shape[0:-1] + [input_shape[-1] * embedding_size])
  return (output, embedding_table)
```

### embedding_postprocessor

在 word_embeddings 的基础上增加 segment_id 和 position 信息，最后将叠加后 embedding 分别进行 layer_norm(对每个样本的不同维度进行归一化操作)，batch_norm(是对不同样本的同一特征进行归一化操作)和 dropout(一个张量中某几个位置的值变成 0)操作。

token_type_table 与 full_position_embeddings 为模型待学习参数。它们和 word_embedding 是对应位置相加，不改变 shape

```python
def embedding_postprocessor(input_tensor,
                            use_token_type=False,
                            token_type_ids=None,
                            token_type_vocab_size=16,
                            token_type_embedding_name="token_type_embeddings",
                            use_position_embeddings=True,
                            position_embedding_name="position_embeddings",
                            initializer_range=0.02,
                            max_position_embeddings=512,
                            dropout_prob=0.1):
  """Performs various post-processing on a word embedding tensor.

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length,
      embedding_size].
    use_token_type: bool. Whether to add embeddings for `token_type_ids`.
    token_type_ids: (optional) int32 Tensor of shape [batch_size, seq_length].
      Must be specified if `use_token_type` is True.
    token_type_vocab_size: int. The vocabulary size of `token_type_ids`.
    token_type_embedding_name: string. The name of the embedding table variable
      for token type ids.
    use_position_embeddings: bool. Whether to add position embeddings for the
      position of each token in the sequence.
    position_embedding_name: string. The name of the embedding table variable
      for positional embeddings.
    initializer_range: float. Range of the weight initialization.
    max_position_embeddings: int. Maximum sequence length that might ever be
      used with this model. This can be longer than the sequence length of
      input_tensor, but cannot be shorter.
    dropout_prob: float. Dropout probability applied to the final output tensor.

  Returns:
    float tensor with same shape as `input_tensor`.

  Raises:
    ValueError: One of the tensor shapes or input values is invalid.
  """
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  width = input_shape[2]

  output = input_tensor

  if use_token_type:
    if token_type_ids is None:
      raise ValueError("`token_type_ids` must be specified if"
                       "`use_token_type` is True.")
    token_type_table = tf.get_variable(
        name=token_type_embedding_name,
        shape=[token_type_vocab_size, width],
        initializer=create_initializer(initializer_range))
    # This vocab will be small so we always do one-hot here, since it is always
    # faster for a small vocabulary.
    flat_token_type_ids = tf.reshape(token_type_ids, [-1])
    one_hot_ids = tf.one_hot(flat_token_type_ids, depth=token_type_vocab_size)
    token_type_embeddings = tf.matmul(one_hot_ids, token_type_table)
    token_type_embeddings = tf.reshape(token_type_embeddings,
                                       [batch_size, seq_length, width])
    output += token_type_embeddings

  if use_position_embeddings:
    assert_op = tf.assert_less_equal(seq_length, max_position_embeddings)
    with tf.control_dependencies([assert_op]):
      full_position_embeddings = tf.get_variable(
          name=position_embedding_name,
          shape=[max_position_embeddings, width],
          initializer=create_initializer(initializer_range))
      # Since the position embedding table is a learned variable, we create it
      # using a (long) sequence length `max_position_embeddings`. The actual
      # sequence length might be shorter than this, for faster training of
      # tasks that do not have long sequences.
      #
      # So `full_position_embeddings` is effectively an embedding table
      # for position [0, 1, 2, ..., max_position_embeddings-1], and the current
      # sequence has positions [0, 1, 2, ... seq_length-1], so we can just
      # perform a slice.
      position_embeddings = tf.slice(full_position_embeddings, [0, 0],
                                     [seq_length, -1])
      num_dims = len(output.shape.as_list())

      # Only the last two dimensions are relevant (`seq_length` and `width`), so
      # we broadcast among the first dimensions, which is typically just
      # the batch size.
      position_broadcast_shape = []
      for _ in range(num_dims - 2):
        position_broadcast_shape.append(1)
      position_broadcast_shape.extend([seq_length, width])
      position_embeddings = tf.reshape(position_embeddings,
                                       position_broadcast_shape)
      output += position_embeddings

  output = layer_norm_and_dropout(output, dropout_prob)
  return output
```

### transformer_model

```python
def transformer_model(input_tensor,
                      attention_mask=None,
                      hidden_size=768,
                      num_hidden_layers=12,
                      num_attention_heads=12,
                      intermediate_size=3072,
                      intermediate_act_fn=gelu,
                      hidden_dropout_prob=0.1,
                      attention_probs_dropout_prob=0.1,
                      initializer_range=0.02,
                      do_return_all_layers=False):
  """Multi-headed, multi-layer Transformer from "Attention is All You Need".

  This is almost an exact implementation of the original Transformer encoder.

  See the original paper:
  https://arxiv.org/abs/1706.03762

  Also see:
  https://github.com/tensorflow/tensor2tensor/blob/master/tensor2tensor/models/transformer.py

  Args:
    input_tensor: float Tensor of shape [batch_size, seq_length, hidden_size].
    attention_mask: (optional) int32 Tensor of shape [batch_size, seq_length,
      seq_length], with 1 for positions that can be attended to and 0 in
      positions that should not be.
    hidden_size: int. Hidden size of the Transformer.
    num_hidden_layers: int. Number of layers (blocks) in the Transformer.
    num_attention_heads: int. Number of attention heads in the Transformer.
    intermediate_size: int. The size of the "intermediate" (a.k.a., feed
      forward) layer.
    intermediate_act_fn: function. The non-linear activation function to apply
      to the output of the intermediate/feed-forward layer.
    hidden_dropout_prob: float. Dropout probability for the hidden layers.
    attention_probs_dropout_prob: float. Dropout probability of the attention
      probabilities.
    initializer_range: float. Range of the initializer (stddev of truncated
      normal).
    do_return_all_layers: Whether to also return all layers or just the final
      layer.

  Returns:
    float Tensor of shape [batch_size, seq_length, hidden_size], the final
    hidden layer of the Transformer.

  Raises:
    ValueError: A Tensor shape or parameter is invalid.
  """
  # hidden size需要是注意力头个数的倍数
  if hidden_size % num_attention_heads != 0:
    raise ValueError(
        "The hidden size (%d) is not a multiple of the number of attention "
        "heads (%d)" % (hidden_size, num_attention_heads))

  attention_head_size = int(hidden_size / num_attention_heads)
  input_shape = get_shape_list(input_tensor, expected_rank=3)
  batch_size = input_shape[0]
  seq_length = input_shape[1]
  input_width = input_shape[2]

  # The Transformer performs sum residuals on all layers so the input needs
  # to be the same as the hidden size.
  if input_width != hidden_size:
    raise ValueError("The width of the input tensor (%d) != hidden size (%d)" %
                     (input_width, hidden_size))

  # We keep the representation as a 2D tensor to avoid re-shaping it back and
  # forth from a 3D tensor to a 2D tensor. Re-shapes are normally free on
  # the GPU/CPU but may not be free on the TPU, so we want to minimize them to
  # help the optimizer.
  prev_output = reshape_to_matrix(input_tensor)

  all_layer_outputs = []
  for layer_idx in range(num_hidden_layers):
    with tf.variable_scope("layer_%d" % layer_idx):
      layer_input = prev_output

      with tf.variable_scope("attention"):
        attention_heads = []
        with tf.variable_scope("self"):
          # 这里引入attention_layer，输入是上一次的输出，输出是context Z矩阵
          # attention_head: [B*F, N*H] if do_return_2d_tensor=True
          attention_head = attention_layer(
              from_tensor=layer_input,
              to_tensor=layer_input,
              attention_mask=attention_mask,
              num_attention_heads=num_attention_heads,
              size_per_head=attention_head_size,
              attention_probs_dropout_prob=attention_probs_dropout_prob,
              initializer_range=initializer_range,
              do_return_2d_tensor=True,
              batch_size=batch_size,
              from_seq_length=seq_length,
              to_seq_length=seq_length)
          attention_heads.append(attention_head)

        attention_output = None
        # 调试过程中确定每次len(attention_heads)都是1
        if len(attention_heads) == 1:
          attention_output = attention_heads[0]
        else:
          # In the case where we have other sequences, we just concatenate
          # them to the self-attention head before the projection.
          # 把所有的Z拼接在一起，然后使用Wo线性变换
          attention_output = tf.concat(attention_heads, axis=-1)

        # Run a linear projection of `hidden_size` then add a residual
        # with `layer_input`.
        # 上面是self-attention部分，下面是和layer_input进行残差连接
        # 使用线性变换到hidden_size维，然后dropout，
        # 然后加上layer_input之后layer_norm（对一个样本的特征进行归一化）
        # [B*F, N*H] -> [B*F, hidden_size]
        with tf.variable_scope("output"):
          attention_output = tf.layers.dense(
              attention_output,
              hidden_size,
              kernel_initializer=create_initializer(initializer_range))
          attention_output = dropout(attention_output, hidden_dropout_prob)
          attention_output = layer_norm(attention_output + layer_input)

      # The activation is only applied to the "intermediate" hidden layer.
      # 再进行一个带激活函数的线性变换，激活函数只用在这，用的是smooth版的relu：gelu
      with tf.variable_scope("intermediate"):
        intermediate_output = tf.layers.dense(
            attention_output,
            intermediate_size,
            activation=intermediate_act_fn,
            kernel_initializer=create_initializer(initializer_range))

      # Down-project back to `hidden_size` then add the residual.
      # 再次有一个layer_output和attention_output的残差连接
      with tf.variable_scope("output"):
        layer_output = tf.layers.dense(
            intermediate_output,
            hidden_size,
            kernel_initializer=create_initializer(initializer_range))
        layer_output = dropout(layer_output, hidden_dropout_prob)
        layer_output = layer_norm(layer_output + attention_output)
        prev_output = layer_output
        all_layer_outputs.append(layer_output)

  # bert的配置是True，返回所有层
  if do_return_all_layers:
    final_outputs = []
    for layer_output in all_layer_outputs:
      # [4096, 512] -> [32, 128, 512]
      final_output = reshape_from_matrix(layer_output, input_shape)
      final_outputs.append(final_output)
    return final_outputs
  else:
    final_output = reshape_from_matrix(prev_output, input_shape)
    return final_output
```

### attention_layer

```python
def attention_layer(from_tensor,
                    to_tensor,
                    attention_mask=None,
                    num_attention_heads=1,
                    size_per_head=512,
                    query_act=None,
                    key_act=None,
                    value_act=None,
                    attention_probs_dropout_prob=0.0,
                    initializer_range=0.02,
                    do_return_2d_tensor=False,
                    batch_size=None,
                    from_seq_length=None,
                    to_seq_length=None):
  """Performs multi-headed attention from `from_tensor` to `to_tensor`.

  This is an implementation of multi-headed attention based on "Attention
  is all you Need". If `from_tensor` and `to_tensor` are the same, then
  this is self-attention. Each timestep in `from_tensor` attends to the
  corresponding sequence in `to_tensor`, and returns a fixed-with vector.

  This function first projects `from_tensor` into a "query" tensor and
  `to_tensor` into "key" and "value" tensors. These are (effectively) a list
  of tensors of length `num_attention_heads`, where each tensor is of shape
  [batch_size, seq_length, size_per_head].

  Then, the query and key tensors are dot-producted and scaled. These are
  softmaxed to obtain attention probabilities. The value tensors are then
  interpolated by these probabilities, then concatenated back to a single
  tensor and returned.

  In practice, the multi-headed attention are done with transposes and
  reshapes rather than actual separate tensors.

  Args:
    from_tensor: float Tensor of shape [batch_size, from_seq_length,
      from_width].
    to_tensor: float Tensor of shape [batch_size, to_seq_length, to_width].

    attention_mask: (optional) int32 Tensor of shape [batch_size,
      from_seq_length, to_seq_length]. The values should be 1 or 0. The
      attention scores will effectively be set to -infinity for any positions in
      the mask that are 0, and will be unchanged for positions that are 1.
    num_attention_heads: int. Number of attention heads.
    size_per_head: int. Size of each attention head. 传入的是 int(hidden_size / num_attention_heads)

    query_act: (optional) Activation function for the query transform.
    key_act: (optional) Activation function for the key transform.
    value_act: (optional) Activation function for the value transform.

    attention_probs_dropout_prob: (optional) float. Dropout probability of the
      attention probabilities.
    initializer_range: float. Range of the weight initializer.
    do_return_2d_tensor: bool. If True, the output will be of shape [batch_size
      * from_seq_length, num_attention_heads * size_per_head]. If False, the
      output will be of shape [batch_size, from_seq_length, num_attention_heads
      * size_per_head].

    batch_size: (Optional) int. If the input is 2D, this might be the batch size
      of the 3D version of the `from_tensor` and `to_tensor`.
    from_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `from_tensor`.
    to_seq_length: (Optional) If the input is 2D, this might be the seq length
      of the 3D version of the `to_tensor`.

  Returns:
    float Tensor of shape [batch_size, from_seq_length,
      num_attention_heads * size_per_head]. (If `do_return_2d_tensor` is
      true, this will be of shape [batch_size * from_seq_length,
      num_attention_heads * size_per_head]).

  Raises:
    ValueError: Any of the arguments or tensor shapes are invalid.
  """

  def transpose_for_scores(input_tensor, batch_size, num_attention_heads,
                           seq_length, width):
    output_tensor = tf.reshape(
        input_tensor, [batch_size, seq_length, num_attention_heads, width])

    output_tensor = tf.transpose(output_tensor, [0, 2, 1, 3])
    return output_tensor

  from_shape = get_shape_list(from_tensor, expected_rank=[2, 3])
  to_shape = get_shape_list(to_tensor, expected_rank=[2, 3])

  if len(from_shape) != len(to_shape):
    raise ValueError(
        "The rank of `from_tensor` must match the rank of `to_tensor`.")

  if len(from_shape) == 3:
    batch_size = from_shape[0]
    from_seq_length = from_shape[1]
    to_seq_length = to_shape[1]
  elif len(from_shape) == 2:
    if (batch_size is None or from_seq_length is None or to_seq_length is None):
      raise ValueError(
          "When passing in rank 2 tensors to attention_layer, the values "
          "for `batch_size`, `from_seq_length`, and `to_seq_length` "
          "must all be specified.")

  # Scalar dimensions referenced here:
  #   B = batch size (number of sequences)
  #   F = `from_tensor` sequence length
  #   T = `to_tensor` sequence length
  #   N = `num_attention_heads`
  #   H = `size_per_head`
  #   N * H 就相当于hidden size

  # from_tensor or to tensor: rank >=2 -> [batch * seq_len, width]
  # 相当于batch * seq_len的二维按行顺序排列成了一维
  from_tensor_2d = reshape_to_matrix(from_tensor)
  to_tensor_2d = reshape_to_matrix(to_tensor)

  # 下面是计算Q、K和V，实现不是原本的三维矩阵乘，而是压缩成了二维
  # 得到的结果相当于是每个attention-head的Q按照列拼接在一起，所以输出的unit
  # 是num_attention_heads*size_per_head，每个输出节点都有一组权重，相当于W^Q的一列
  # 返回的是一个tensor
  # `query_layer` = [B*F, width]->[B*F, N*H]
  query_layer = tf.layers.dense(
      from_tensor_2d,
      num_attention_heads * size_per_head,
      activation=query_act,
      name="query",
      # kernel_initializer用于初始化权重w
      kernel_initializer=create_initializer(initializer_range))

  # `key_layer` = [B*T, N*H]
  key_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=key_act,
      name="key",
      kernel_initializer=create_initializer(initializer_range))

  # `value_layer` = [B*T, N*H]
  value_layer = tf.layers.dense(
      to_tensor_2d,
      num_attention_heads * size_per_head,
      activation=value_act,
      name="value",
      kernel_initializer=create_initializer(initializer_range))

  # `query_layer` = [B, N, F, H]
  # 先reshape到[B, F, N, H]，然后transpose到[B, N, F, H]
  query_layer = transpose_for_scores(query_layer, batch_size,
                                     num_attention_heads, from_seq_length,
                                     size_per_head)

  # `key_layer` = [B, N, T, H]
  key_layer = transpose_for_scores(key_layer, batch_size, num_attention_heads,
                                   to_seq_length, size_per_head)

  # Take the dot product between "query" and "key" to get the raw
  # attention scores.
  # `attention_scores` = [B, N, F, T]
  # transpose_b=True表示b=key_layer在乘法之前进行转置，其实是最后两个维度进行转置
  # tf.matmul()在高维矩阵乘法中，其实是对高维矩阵的每个二维矩阵相乘
  # 所以转制后变成[B, N, F, H] * [B, N, H, T] => [B, N, F, T]
  attention_scores = tf.matmul(query_layer, key_layer, transpose_b=True)
  # size_per_head就是score除以的\sqrt(d_k)
  attention_scores = tf.multiply(attention_scores,
                                 1.0 / math.sqrt(float(size_per_head)))

  if attention_mask is not None:
    # `attention_mask` = [B, 1, F, T]
    attention_mask = tf.expand_dims(attention_mask, axis=[1])

    # Since attention_mask is 1.0 for positions we want to attend and 0.0 for
    # masked positions, this operation will create a tensor which is 0.0 for
    # positions we want to attend and -10000.0 for masked positions.
    # 如果mask显示为1，那么不改变score，否则会变得很小。这个mask是attention的可视域
    # 这个mask追溯上去得到的是全1的矩阵，对padding部分进行mask
    adder = (1.0 - tf.cast(attention_mask, tf.float32)) * -10000.0

    # Since we are adding it to the raw scores before the softmax, this is
    # effectively the same as removing these entirely.
    attention_scores += adder

  # Normalize the attention scores to probabilities.
  # `attention_probs` = [B, N, F, T]
  attention_probs = tf.nn.softmax(attention_scores)

  # This is actually dropping out entire tokens to attend to, which might
  # seem a bit unusual, but is taken from the original Transformer paper.
  # base bert_config中attention_probs_dropout_prob=0.1，所以实际上dropout了0.9的token
  # dropout函数输出的是output = tf.nn.dropout(input_tensor, 1.0 - dropout_prob)
  attention_probs = dropout(attention_probs, attention_probs_dropout_prob)

  # `value_layer` = [B, T, N, H]
  value_layer = tf.reshape(
      value_layer,
      [batch_size, to_seq_length, num_attention_heads, size_per_head])

  # `value_layer` = [B, N, T, H]
  value_layer = tf.transpose(value_layer, [0, 2, 1, 3])

  # `context_layer` = [B, N, F, H]
  context_layer = tf.matmul(attention_probs, value_layer)

  # `context_layer` = [B, F, N, H]
  # context其实就是图解中的Z值
  context_layer = tf.transpose(context_layer, [0, 2, 1, 3])

  if do_return_2d_tensor:
    # `context_layer` = [B*F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size * from_seq_length, num_attention_heads * size_per_head])
  else:
    # `context_layer` = [B, F, N*H]
    context_layer = tf.reshape(
        context_layer,
        [batch_size, from_seq_length, num_attention_heads * size_per_head])

  # 返回的是整个的Z矩阵
  return context_layer
```

### get_masked_lm_output

从 BertModel 部分返回到[model_fn_builder](#model_fn_builder)。

搞定了`modeling.BertModel`，下面开始计算 Masked LM 和 NSP 任务。

两个任务的本质都是分类任务，一个是二分类，即两个 segment 是否是连贯的；一个是多分类，即输入序列中被 mask 的 token 为词表中某个 token 的概率。它们的损失函数都是**交叉熵损失**。

NSP 问题中 0 是连续的，1 是随机的。

在`model_fn_builder`中是通过下面代码进行调用的：

```python
(masked_lm_loss,
  masked_lm_example_loss, masked_lm_log_probs) = get_masked_lm_output(
      bert_config, model.get_sequence_output(), model.get_embedding_table(),
      masked_lm_positions, masked_lm_ids, masked_lm_weights
```

源码：

```python
def get_masked_lm_output(bert_config, input_tensor, output_weights, positions,
                         label_ids, label_weights):
  """Get loss and log probs for the masked LM."""
  # input_tensor是model.get_sequence_output()得到，是encoder layer的最后一层，shape=(32, 128, 512)
  # output_weights是emebdding table
  # position是masked_lm_positions，ids和weights都是masked_
  # gather_indexes是从input_tensor中取出positions位置的tensor
  # 此时input_tensor.shape = （4096, 512)->(640, 512)，即
  # [batch_size*max_predictions_per_seq=20, hidden_size]
  input_tensor = gather_indexes(input_tensor, positions)

  with tf.variable_scope("cls/predictions"):
    # We apply one more non-linear transformation before the output layer.
    # This matrix is not used after pre-training.
    # 对input做了一个fc+ac+norm的操作
    with tf.variable_scope("transform"):
      input_tensor = tf.layers.dense(
          input_tensor,
          units=bert_config.hidden_size,
          activation=modeling.get_activation(bert_config.hidden_act),
          kernel_initializer=modeling.create_initializer(
              bert_config.initializer_range))
      input_tensor = modeling.layer_norm(input_tensor)

    # The output weights are the same as the input embeddings, but there is
    # an output-only bias for each token.
    output_bias = tf.get_variable(
        "output_bias",
        shape=[bert_config.vocab_size],
        initializer=tf.zeros_initializer())
    # output_bias: [30522]
    # input_tensor: [640, 512]
    # output_weights: [30522, 512]
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    # transpose_b表示转置一下
    # logits: [batch_size*max_predictions_per_seq=640, vocab_size=30522]
    # log_probs: [batch_size*max_predictions_per_seq, vocab_size]
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)

    # label_ids: [640, ]
    # label_weights: [640, ]
    label_ids = tf.reshape(label_ids, [-1])
    label_weights = tf.reshape(label_weights, [-1])

    # [batch_size*max_predictions_per_seq, vocab_size]
    one_hot_labels = tf.one_hot(
        label_ids, depth=bert_config.vocab_size, dtype=tf.float32)

    # The `positions` tensor might be zero-padded (if the sequence is too
    # short to have the maximum number of predictions). The `label_weights`
    # tensor has a value of 1.0 for every real prediction and 0.0 for the
    # padding predictions.
    # [640, 30522]*[640, 30522]
    # reduce_sum: 调用reduce_sum(arg1, arg2)时，arg1为要求和的数据，arg2为0或1，
    # 通常用reduction_indices=[0]或reduction_indices=[1]来传递。
    # arg2 = 0时，是纵向求和，当arg2 = 1时，是横向求和；省略arg2参数时，默认对所有元素进行求和。
    # reduce就是“对矩阵降维”的含义，在reduce_sum()中就是按照求和的方式对矩阵降维。
    # 那么其他reduce前缀的函数也举一反三了，比如reduce_mean()就是按照某个维度求平均值，等等。
    # per_example_loss->reduce_sum->[640, ]，每个位置预测对的log概率*(-1)
    per_example_loss = -tf.reduce_sum(log_probs * one_hot_labels, axis=[-1])
    # label_weights即masked_lm_weights
    # 和input_mask的作用一样，标注masked_lm_ids 和 masked_lm_positions 哪些是真实值，哪些是补全值
    # label_weights: [640, ]
    numerator = tf.reduce_sum(label_weights * per_example_loss)
    # 计算非0的真实位置个数，+1e-5防止分子为0，
    denominator = tf.reduce_sum(label_weights) + 1e-5
    loss = numerator / denominator

  return (loss, per_qiumple_loss, log_probs)
```

### get_next_sentence_output

`model_fn_builder`中计算完了`get_masked_lm_output`之后，计算`get_next_sentence_output`。

调用方式：

```python
(next_sentence_loss, next_sentence_example_loss,
  next_sentence_log_probs) = get_next_sentence_output(
      bert_config, model.get_pooled_output(), next_sentence_labels)
```

源码：

```python
def get_next_sentence_output(bert_config, input_tensor, labels):
  """Get loss and log probs for the next sentence prediction."""

  # Simple binary classification. Note that 0 is "next sentence" and 1 is
  # "random sentence". This weight matrix is not used after pre-training.
  # input_tensor: [batch_size, hidden_size]
  # labels: [batch_size, 1]
  with tf.variable_scope("cls/seq_relationship"):
    output_weights = tf.get_variable(
        "output_weights",
        shape=[2, bert_config.hidden_size],
        initializer=modeling.create_initializer(bert_config.initializer_range))
    output_bias = tf.get_variable(
        "output_bias", shape=[2], initializer=tf.zeros_initializer())

    # logits: [batch_size, 2]，相当于全连接层+softmax分类
    logits = tf.matmul(input_tensor, output_weights, transpose_b=True)
    logits = tf.nn.bias_add(logits, output_bias)
    log_probs = tf.nn.log_softmax(logits, axis=-1)
    labels = tf.reshape(labels, [-1])
    # one_hot_labels: [batch_size, 2]
    one_hot_labels = tf.one_hot(labels, depth=2, dtype=tf.float32)
    # 交叉熵损失：-\frac{1}{N} \sum_i (y_i log(p_i) + (1-y_i) log(1-p_i))
    per_example_loss = -tf.reduce_sum(one_hot_labels * log_probs, axis=-1)
    loss = tf.reduce_mean(per_example_loss)
    return (loss, per_example_loss, log_probs)
```

### create_optimizer

调用部分：

```python
if mode == tf.estimator.ModeKeys.TRAIN:
  # 创建优化器optimizer
  train_op = optimization.create_optimizer(
      total_loss, learning_rate, num_train_steps, num_warmup_steps, use_tpu)
```

实现部分：

```python
def create_optimizer(loss, init_lr, num_train_steps, num_warmup_steps, use_tpu):
  """Creates an optimizer training op."""
  global_step = tf.train.get_or_create_global_step()

  learning_rate = tf.constant(value=init_lr, shape=[], dtype=tf.float32)

  # Implements linear decay of the learning rate.
  learning_rate = tf.train.polynomial_decay(
      learning_rate,
      global_step,
      num_train_steps,
      end_learning_rate=0.0,
      power=1.0,
      cycle=False)

  # Implements linear warmup. I.e., if global_step < num_warmup_steps, the
  # learning rate will be `global_step/num_warmup_steps * init_lr`.
  if num_warmup_steps:
    global_steps_int = tf.cast(global_step, tf.int32)
    warmup_steps_int = tf.constant(num_warmup_steps, dtype=tf.int32)

    global_steps_float = tf.cast(global_steps_int, tf.float32)
    warmup_steps_float = tf.cast(warmup_steps_int, tf.float32)

    warmup_percent_done = global_steps_float / warmup_steps_float
    warmup_learning_rate = init_lr * warmup_percent_done

    is_warmup = tf.cast(global_steps_int < warmup_steps_int, tf.float32)
    # 小trick，is_warmup为1时，算后面的
    learning_rate = (
        (1.0 - is_warmup) * learning_rate + is_warmup * warmup_learning_rate)

  # It is recommended that you use this optimizer for fine tuning, since this
  # is how the model was trained (note that the Adam m/v variables are NOT
  # loaded from init_checkpoint.)
  optimizer = AdamWeightDecayOptimizer(
      learning_rate=learning_rate,
      weight_decay_rate=0.01,
      beta_1=0.9,
      beta_2=0.999,
      epsilon=1e-6,
      exclude_from_weight_decay=["LayerNorm", "layer_norm", "bias"])

  if use_tpu:
    optimizer = tf.contrib.tpu.CrossShardOptimizer(optimizer)

  tvars = tf.trainable_variables()
  grads = tf.gradients(loss, tvars)

  # This is how the model was pre-trained.
  (grads, _) = tf.clip_by_global_norm(grads, clip_norm=1.0)

  train_op = optimizer.apply_gradients(
      zip(grads, tvars), global_step=global_step)

  # Normally the global step update is done inside of `apply_gradients`.
  # However, `AdamWeightDecayOptimizer` doesn't do this. But if you use
  # a different optimizer, you should probably take this line out.
  new_global_step = global_step + 1
  train_op = tf.group(train_op, [global_step.assign(new_global_step)])
  return train_op

class AdamWeightDecayOptimizer(tf.train.Optimizer):
  """A basic Adam optimizer that includes "correct" L2 weight decay."""

  def __init__(self,
               learning_rate,
               weight_decay_rate=0.0,
               beta_1=0.9,
               beta_2=0.999,
               epsilon=1e-6,
               exclude_from_weight_decay=None,
               name="AdamWeightDecayOptimizer"):
    """Constructs a AdamWeightDecayOptimizer."""
    super(AdamWeightDecayOptimizer, self).__init__(False, name)

    self.learning_rate = learning_rate
    self.weight_decay_rate = weight_decay_rate
    self.beta_1 = beta_1
    self.beta_2 = beta_2
    self.epsilon = epsilon
    self.exclude_from_weight_decay = exclude_from_weight_decay

  def apply_gradients(self, grads_and_vars, global_step=None, name=None):
    """See base class."""
    assignments = []
    for (grad, param) in grads_and_vars:
      if grad is None or param is None:
        continue

      param_name = self._get_variable_name(param.name)

      m = tf.get_variable(
          name=param_name + "/adam_m",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())
      v = tf.get_variable(
          name=param_name + "/adam_v",
          shape=param.shape.as_list(),
          dtype=tf.float32,
          trainable=False,
          initializer=tf.zeros_initializer())

      # Standard Adam update.
      next_m = (
          tf.multiply(self.beta_1, m) + tf.multiply(1.0 - self.beta_1, grad))
      next_v = (
          tf.multiply(self.beta_2, v) + tf.multiply(1.0 - self.beta_2,
                                                    tf.square(grad)))

      update = next_m / (tf.sqrt(next_v) + self.epsilon)

      # Just adding the square of the weights to the loss function is *not*
      # the correct way of using L2 regularization/weight decay with Adam,
      # since that will interact with the m and v parameters in strange ways.
      #
      # Instead we want ot decay the weights in a manner that doesn't interact
      # with the m/v parameters. This is equivalent to adding the square
      # of the weights to the loss with plain (non-momentum) SGD.
      if self._do_use_weight_decay(param_name):
        update += self.weight_decay_rate * param

      update_with_lr = self.learning_rate * update

      next_param = param - update_with_lr

      assignments.extend(
          [param.assign(next_param),
           m.assign(next_m),
           v.assign(next_v)])
    return tf.group(*assignments, name=name)

  def _do_use_weight_decay(self, param_name):
    """Whether to use L2 weight decay for `param_name`."""
    if not self.weight_decay_rate:
      return False
    if self.exclude_from_weight_decay:
      for r in self.exclude_from_weight_decay:
        if re.search(r, param_name) is not None:
          return False
    return True

  def _get_variable_name(self, param_name):
    """Get the variable name from the tensor name."""
    m = re.match("^(.*):\\d+$", param_name)
    if m is not None:
      param_name = m.group(1)
    return param_name
```

首先是学习率部分，将学习率设置为线性衰减的形式，接着根据global_step是否达到num_warmup_steps，在原来线性衰减的基础上将学习率进一步分成warmup_learning_rate和learning_rate两种方式。然后是优化器的构建。

先是实例化AdamWeightDecayOptimizer(其是梯度下降法的一种变种，也由待更新参数、学习率和参数更新方向三大要素组成)，接着通过tvars = tf.trainable_variables()解析出模型中所有待训练的参数变量，并给出loss关于所有参数变量的梯度表示grads = tf.gradients(loss, tvars)，同时限制梯度的大小。最后基于上述描述的梯度与变量，进行参数更新操作。更新时，依此遍历每一个待更新的参数，根据标准的Adam更新公式(参考Adam和学习率衰减（learning rate decay）)，先确定参数更新方向，接着在方向的基础上增加衰减参数(这个操作叫纠正的L2 weight decay)，然后在纠正后的方向上移动一定距离(learning_rate * update)后，更新现有的参数。 以上更新步骤随着训练步数不断进行，直到走完所有训练步数。

### Estimator 类

tf 提供了很多预创建的 Estimator，也可以自己定义 Estimator 类。但都是基于`tf.estimator.Estimator`。

#### 介绍

Estimator 类，用来训练和验证 TensorFlow 模型：

- Estimator 对象包含了一个模型 model_fn，这个模型给定输入和参数，会返回训练、验证或者预测等所需要的操作节点。
- 所有的输出（检查点、事件文件等）会写入到 model_dir，或者其子文件夹中。如果 model_dir 为空，则默认为临时目录。
- config 参数为 tf.estimator.RunConfig 对象，包含了执行环境的信息。如果没有传递 config，则它会被 Estimator 实例化，使用的是默认配置。
- params 包含了超参数。Estimator 只传递超参数，不会检查超参数，因此 params 的结构完全取决于开发者。
- Estimator 的所有方法都不能被子类覆盖（它的构造方法强制决定的）。子类应该使用 model_fn 来配置母类，或者增添方法来实现特殊的功能。
- Estimator 不支持 Eager Execution（eager execution 能够使用 Python 的 debug 工具、数据结构与控制流。并且无需使用 placeholder、session，计算结果能够立即得出）。

#### 初始化

`__init__(self, model_fn, model_dir=None, config=None, params=None, warm_start_from=None)`

构造一个 Estimator 的实例。

参数：

1. model_fn: 模型函数。
   1. 参数：
      1. features: 这是 input_fn 返回的第一项（input_fn 是 train, evaluate 和 predict 的参数）。类型应该是单一的 Tensor 或者 dict。
      2. labels: 这是 input_fn 返回的第二项。类型应该是单一的 Tensor 或者 dict。如果 mode 为 ModeKeys.PREDICT，则会默认为 labels=None。如果 model_fn 不接受 mode，model_fn 应该仍然可以处理 labels=None。
      3. mode: 可选。指定是训练、验证还是测试。参见 ModeKeys。
      4. params: 可选，超参数的 dict。 可以从超参数调整中配置 Estimators。
      5. config: 可选，配置。如果没有传则为默认值。可以根据 num_ps_replicas 或 model_dir 等配置更新 model_fn。
   2. 返回：EstimatorSpec
2. model_dir:
   1. 保存模型参数、图等的地址，也可以用来将路径中的检查点加载至 estimator 中来继续训练之前保存的模型。
   2. 如果是 PathLike， 那么路径就固定为它了。
   3. 如果是 None，那么 config 中的 model_dir 会被使用（如果设置了的话）
   4. 如果两个都设置了，那么必须相同；如果两个都是 None，则会使用临时目录。
3. config: 配置类。
4. params: 超参数的 dict，会被传递到 model_fn。keys 是参数的名称，values 是基本 python 类型。
5. warm_start_from:
   1. 可选，字符串，检查点的文件路径，用来指示从哪里开始热启动。
   2. 或者是 tf.estimator.WarmStartSettings 类来全部配置热启动。
   3. 如果是字符串路径，则所有的变量都是热启动，并且需要 Tensor 和词汇的名字都没有变。
6. 异常：
   1. RuntimeError： 开启了 eager execution
   2. ValueError：model_fn 的参数与 params 不匹配
   3. ValueError：这个函数被 Estimator 的子类所覆盖

#### train

`train(self, input_fn, hooks=None, steps=None, max_steps=None, saving_listeners=None)`

根据所给数据 input_fn， 对模型进行训练。

参数：

1. input_fn：一个函数，提供由小 batches 组成的数据， 供训练使用。必须返回以下之一：
   - 一个 'tf.data.Dataset'对象：Dataset 的输出必须是一个元组 (features, labels)，元组要求如下。
   - 一个元组 (features, labels)：features 是一个 Tensor 或者一个字典（特征名为 Tensor），labels 是一个 Tensor 或者一个字典（特征名为 Tensor）。features 和 labels 都被 model_fn 所使用，应该符合 model_fn 输入的要求。
2. hooks：SessionRunHook 子类实例的列表。用于在训练循环内部执行。
3. steps：模型训练的步数。
   1. 如果是 None， 则一直训练，直到 input_fn 抛出了超过界限的异常。
   2. steps 是递进式进行的。如果执行了两次训练（steps=10），则总共训练了 20 次。如果中途抛出了越界异常，则训练在 20 次之前就会停止。
   3. 如果你不想递进式进行，请换为设置 max_steps。如果设置了 steps，则 max_steps 必须是 None。
4. max_steps：模型训练的最大步数。
   1. 如果为 None，则一直训练，直到 input_fn 抛出了超过界限的异常。
   2. 如果设置了 max_steps， 则 steps 必须是 None。
   3. 如果中途抛出了越界异常，则训练在 max_steps 次之前就会停止。
   4. 执行两次 train(steps=100) 意味着 200 次训练；但是，执行两次 train(max_steps=100) 意味着第二次执行不会进行任何训练，因为第一次执行已经做完了所有的 100 次。
5. saving_listeners：CheckpointSaverListener 对象的列表。用于在保存检查点之前或之后立即执行的回调函数。
   返回：self：为了链接下去。

异常：

- ValueError：steps 和 max_steps 都不是 None
- ValueError：steps 或 max_steps <= 0

#### evaluate

`evaluate(self, input_fn, steps=None, hooks=None, checkpoint_path=None, name=None)`

根据所给数据 input_fn， 对模型进行验证。

对于每一步，执行 input_fn（返回数据的一个 batch）。一直进行验证，直到：

- steps 个 batches 进行完毕，或者
- input_fn 抛出了越界异常（OutOfRangeError 或 StopIteration）

参数：

1. input_fn：一个函数，构造了验证所需的输入数据，必须返回以下之一：
   1. 一个 'tf.data.Dataset'对象：Dataset 的输出必须是一个元组 (features, labels)，元组要求如下。
   2. 一个元组 (features, labels)：features 是一个 Tensor 或者一个字典（特征名为 Tensor），labels 是一个 Tensor 或者一个字典（特征名为 Tensor）。features 和 labels 都被 model_fn 所使用，应该符合 model_fn 输入的要求。
1. steps：模型验证的步数。如果是 None， 则一直验证，直到 input_fn 抛出了超过界限的异常。
1. hooks：SessionRunHook 子类实例的列表。用于在验证内部执行。
1. checkpoint_path： 用于验证的检查点路径。如果是 None， 则使用 model_dir 中最新的检查点。
1. name：验证的名字。使用者可以针对不同的数据集运行多个验证操作，比如训练集 vs 测试集。不同验证的结果被保存在不同的文件夹中，且分别出现在 tensorboard 中。

返回：

- 返回一个字典，包括 model_fn 中指定的评价指标、global_step（包含验证进行的全局步数）

异常：

- ValueError：如果 step 小于等于 0
- ValueError：如果 model_dir 指定的模型没有被训练，或者指定的 checkpoint_path 为空。

#### predict

`predict(self, input_fn, predict_keys=None, hooks=None, checkpoint_path=None, yield_single_examples=True)`

对给出的特征进行预测。

参数：

1. input_fn：一个函数，构造特征。预测一直进行下去，直到 input_fn 抛出了越界异常（OutOfRangeError 或 StopIteration）。函数必须返回以下之一：
   1. 一个 'tf.data.Dataset'对象：Dataset 的输出和以下的限制相同。
   2. features：一个 Tensor 或者一个字典（特征名为 Tensor）。features 被 model_fn 所使用，应该符合 model_fn 输入的要求。
   3. 一个元组，其中第一项为 features。
1. predict_keys：字符串列表，要预测的键值。当 EstimatorSpec.predictions 是一个 dict 时使用。如果使用了 predict_keys， 那么剩下的预测值会从字典中过滤掉。如果是 None，则返回全部。
1. hooks：SessionRunHook 子类实例的列表。用于在预测内部回调。
1. checkpoint_path： 用于预测的检查点路径。如果是 None， 则使用 model_dir 中最新的检查点。
1. yield_single_examples：If False, yield the whole batch as returned by the model_fn instead of decomposing the batch into individual elements. This is useful if model_fn returns some tensors whose first dimension is not equal to the batch size.

返回：
- predictions tensors 的值

异常：
- ValueError：model_dir 中找不到训练好的模型。
- ValueError：预测值的 batch 长度不同，且 yield_single_examples 为 True。
- ValueError：predict_keys 和 predictions 之间有冲突。例如，predict_keys 不是 None，但是 EstimatorSpec.predictions 不是一个 dict。

### estimator.train/evaluate

了解了 Estimator 可以理解上面的`estimator.train`和`estimator.evaluate`了

搞定了上面的部分就可以完成[model_fn_builder](#model_fn_builder)部分的理解，然后返回[main](#main-of-pretrain)。这下只剩`estimator.train`和`estimator.evaluate`了。

```python
estimator = tf.contrib.tpu.TPUEstimator(
    use_tpu=FLAGS.use_tpu,
    model_fn=model_fn,
    config=run_config,
    train_batch_size=FLAGS.train_batch_size,
    eval_batch_size=FLAGS.eval_batch_size)

if FLAGS.do_train:
  tf.logging.info("***** Running training *****")
  tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
  # 从tfrecord解析出BERT的输入数据
  train_input_fn = input_fn_builder(
      input_files=input_files,
      max_seq_length=FLAGS.max_seq_length,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq,
      is_training=True)
  estimator.train(input_fn=train_input_fn, max_steps=FLAGS.num_train_steps)

if FLAGS.do_eval:
  tf.logging.info("***** Running evaluation *****")
  tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
  # 如果不训练只是eval，输入也是一样的tfrecord
  eval_input_fn = input_fn_builder(
      input_files=input_files,
      max_seq_length=FLAGS.max_seq_length,
      max_predictions_per_seq=FLAGS.max_predictions_per_seq,
      is_training=False)

  # evaluate操作得到结果
  result = estimator.evaluate(
      input_fn=eval_input_fn, steps=FLAGS.max_eval_steps)
````
