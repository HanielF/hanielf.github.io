---
title: ClassyVision框架和多模态视频分类流程
comments: true
mathjax: false
date: 2021-04-03 16:11:01
tags:
  [
    ClassyVision,
    MultiModal,
    VideoClassification,
    DeepLearning,
    Pytorch,
    DistributedTraining,
  ]
categories: MachineLearning
urlname: classy-vision-and-multimodal-video-classification
---

<meta name="referrer" content="no-referrer" />

{% note info %}
记录对 ClassyVision 框架的理解，以及数据处理 pipeline。中间有用到 pytorch 的分布式训练。
{% endnote %}

<!--more-->

## 拉数据和处理

1. 先处理数据得到`videos.txt`，每行一个 pid
2. 然后得到 pid-label 的类似`saishi_train_data_v3.txt`的文件。
3. 然后得到`视频ID,概率值,标签,描述`的数据回查文件。
4. 然后拉取数据，就是`run.sh`
5. 然后生成

### Info

- 登录拉数据 webserver
  - `ssh xudi06@relay.corp.kuaishou.com`
  - `ssh web_server@bjpg-rs8553.yz02 -p22`
- `df -h`找到 mmu 项目路径`/mnt/mmu_ssd/share/yangfan/yuanwei/xingshisenlin_saishi`
- 查看`run.sh`

### run.sh 脚本说明

- `run.sh`脚本文件
- 命令：webserver 下`bash run.sh`
- 功能：拉取视频相关计数、特征、文本、视频封面和指定帧、用户特征
- 生成：`users.txt，author_2tower_embedding_by_photo_id，text.txt，cover目录，users_list.txt，users_list_out.txt`
- users_list 需要自己手动通过`users.txt`生成

- 准备说明：
  - videos.txt: 每行一个 pid

```bash
nohup sh /data/project/export_photo_info_count.sh -f `pwd`/videos.txt -o `pwd`/users.txt &
nohup sh /data/project/export_features.sh -i `pwd`/videos.txt -o `pwd` -t 10 -d "9" &
nohup java -Dfile.encoding=UTF-8 -cp /data/project/chenxiaohui/kuaishou-mmu-data/*:kuaishou-mmu-data-1.0-SNAPSHOT.jar com.kuaishou.runner.BaseRunner -r  ExportTextSer    vice -i `pwd`/videos.txt -o `pwd`/text.txt -t 10 &
nohup sh  /data/project/export_frames.sh  `pwd`/videos.txt `pwd`/cover 1 100 30 &
```

### run.sh 导出视频相关计数

- bash 脚本：`export_photo_info_count.sh`
- 功能：导出视频相关计数
- 输入：videos.txt
- 输出：users.txt
- 格式：photoId \t viewCount \t likeCount \t unlikeCount \t commentCount \t forwardCount \t authorId
- 样例：36853120853 77026 496 0 9 0 1337316458
- wiki: <https://docs.corp.kuaishou.com/k/home/VEPkMOB4UCl4/fcACxteH0QO2VqgPQNA1h5NOx>
- 使用：`` nohup sh /data/project/export_photo_info_count.sh -f `pwd`/videos.txt -o `pwd`/users.txt & ``
- 程序运行参数说明：
  - -f 输入文件，其中每一行是一个 photo id
  - -o 指定的输出文件，结果：photoId \t viewCount \t likeCount \t unlikeCount \t commentCount \t forwardCount \t authorId

### run.sh 导出用户特征向量

- bash 脚本：`export_features.sh`
- 功能：导出特征，-d9 是导出用户特征向量
- 输入：videos.txt
- 输出：author_2tower_embedding_by_photo_id
- 格式：pid \t 64 维的 embedding 向量，向量也可能为空
- wiki：<https://docs.corp.kuaishou.com/k/home/VBbdL4rQHmz4/fcABg6qainvICo6XTlhYqNAQN>
- 使用：`` nohup sh /data/project/export_features.sh -i `pwd`/videos.txt -o `pwd` -t 10 -d "9" & ``
- 参数
  - -i 指定将要传入的 photo id 数据文件，每一行一个 photo id，需要传入绝对路径，导出文件会自动生成
  - -o 指定输出文件的存放目录，绝对路径, 不能以"/"结尾
  - -t 线程个数
  - -d 想要的算法结果，以逗号分隔开，比如 -d "1,2"
- 上述的`-d 9`表示输出`author_2tower_embedding_by_photo_id`，是推荐那边的双塔模型输出的作者信息编码，作用可以理解成这个作者偏向于发什么类型的视频。
- 这个后续会使用`generate_rec_for_raw_user_embeding_from_hive_file()`方法处理得到 rec
- 关于`-d`更多的信息看 wiki

### run.sh 导出文本

- 功能：导出视频相关文本，包括视频第一帧的 caption、标题 title、内容 text、OCR 结果、作者添加的视频字幕
- wiki: <https://docs.corp.kuaishou.com/k/home/VaBp46x_CVEw/fcABSVDkEsfNzNYd5emmlRsye>
- 输入：videos.txt
- 输出：text.txt
- 格式：pid + caption + "\t" + title + "\t" + text + "\t" + ocr + "\t" + speech
- 命令：`` nohup java -Dfile.encoding=UTF-8 -cp /data/project/chenxiaohui/kuaishou-mmu-data/*:kuaishou-mmu-data-1.0-SNAPSHOT.jar com.kuaishou.runner.BaseRunner -r ExportTextService -i `pwd`/videos.txt -o `pwd`/text.txt -t 10 & ``
- 输出：`caption + "\t" + title + "\t" + text + "\t" + ocr + "\t" + speech`
- 参数：
  - -t 线程个数
  - -o 指定输出文件的存放目录，绝对路径, 不能以"/"结尾
  - 输入方式一、输入指定 id 范围
    - -s 开始 id
    - -e 结束 id
  - 输入方式二、输入指定 id
    - -i 指定将要传入的 photo id 数据文件，每一行一个 photo id，需要传入绝对路径

### run.sh 导出视频封面和指定帧

- wiki: <https://docs.corp.kuaishou.com/k/home/VPLKPDK8Fv-k/fcACJiIC2EN2KOKRYwvDciV0g>
- 命令：`` nohup sh /data/project/export_frames.sh `pwd`/videos.txt `pwd`/cover 1 100 30 & ``
- 输入：videos.txt
- 输出：cover 目录下的视频封面和指定帧
- 参数：
  - 第一个参数，photo id 列表文件，每一行一个 photo id，需要传入绝对路径
  - 第二个参数，图片存储到的目录，绝对路径
  - 第三个参数，运行类型。如果导出指定帧(type=2)，photo id 列表文件输入应该是 photo_id\t1,2,3 类型，帧 id 用逗号隔开，photo_id 和帧 id 用\t 隔开. 注意：\t 是不可见的符号，也可以用单个空格进行分割
  - 第四个参数，线程数，最大 200
  - 第五个参数，文件夹数量，一般为 10 (数据是按照”photo_id%文件夹数量“的方式下载存放的，请注意)，每个目录最多存放 10w 个

### get_user_list

准备好 author id，和`cut -f7 -d $'\t' users.txt | sort | uniq > users_list.txt`效果一样

### run.sh 导出用户特征

- wiki: <https://docs.corp.kuaishou.com/k/home/VdeW2QONo4Qw/fcAB7R1aarXIFGzBDBwTvkI0D>
- 命令：`` sh /data/project/user_info_export.sh -i `pwd`/users_list.txt ``
- 输入：`users_list.txt`
- 输出：`users_list_out.txt`
- 格式：userId  用户名 用户简介
- 参数：
  - -i 输入文件，其中每一行是一个 author id
- 需要先准备好 author id：`cut -f7 -d $'\t' users.txt | sort | uniq > users_list.txt`
- 输出：运行成功会在输入文件的文件夹产生一个新的文件，文件名与输入文件名前缀相同，比如  author.txt 会生成  author_out.txt
        结果格式为 userId  用户名 用户简介

### mk_train_org

把拉取下来的，存放在 cover 目录中的数据，打上对应的 label。输出`train_org.txt`文件。

label 文件是人工标注的结果 或者自己清洗得到的

关键代码：

```python
# 读取label和对应的类别编号
# 0 n 其他
# 1 n 舞蹈
labelindex={}
with open('/share/yuanwei05/xingshisenlin/saishi/data/saishi_class.label','r') as fin:
    for l in fin:
      l=l.strip().split()
      labelindex[l[2]]=int(l[0])

# 给每个拉取下来存放在cover的pic，打上对应的label，结果如下
# /share/yangfan/yuanwei/xingshisenlin_saishi/cover/0/36872762760.jpg 0
# /share/yangfan/yuanwei/xingshisenlin_saishi/cover/0/36873258930.jpg 0
pidlabel={}
with open('./saishi_train_data_v3.txt','r') as fin:
    for l in fin:
      l=l.strip().split()
      pid=l[0]
      label=l[1]
      pidlabel[pid]=label
    #with open('./coarse_train_v3/train_org.txt','w') as fout:
    with open('./train_data_v3/train_org.txt','w') as fout:
      image_dir='./cover'
      for target in sorted(os.listdir(image_dir)):
        d = os.path.join(image_dir, target)
        if not os.path.isdir(d):
          continue
        for root, _, fnames in sorted(os.walk(d,followlinks=True)):
          for fname in sorted(fnames):
            if has_file_allowed_extension(fname, IMG_EXTENSIONS):
              path = os.path.join(root, fname)
              image_id = fname.split(".")[0]
              if '_' in image_id:
                image_id=image_id.split('_')[0]
              if image_id in pidlabel:
                #if pidlabel[image_id]=='婴儿否':
                #    continue
                fout.write('%s\t%d\n'%(os.path.abspath(path),labelindex[pidlabel[image_id]]))
```

### transfer_user_pid

- 输入`users_list_out.txt`文件
- 输出`users_profile.txt`
- 输出文件格式是`pid 用户名 用户简介`

```python
def transfer_user_pid():
    lines = open('users_list_out.txt').readlines()
    dic = {}
    for line in lines:
        s = line.strip('\n').split('\t')
        dic[s[0]] = '\t'.join(s[1:])

    lines = open('users.txt').readlines()
    output_f = open('users_profile.txt', 'w')
    for line in lines:
        s = line.strip('\n').split('\t')
        pid = s[0]
        info = dic[s[-1]]
        output_f.write(pid + '\t' + info + '\n')
```

### split_train_val

- 输入：`train_org.txt`
- 输出：`train_data.txt`和`val_data.txt`

```python
def split_train_val(data_path, save_path):
    f = open(os.path.join(data_path, "train_org.txt"), "r")
    lines = f.readlines()
    f.close()
    random.seed(123)
    random.shuffle(lines)
    pos = int(len(lines) * 0.95)
    train_f = open(os.path.join(save_path, "train_data.txt"), "w")
    for line in lines[:pos]:
        train_f.write(line)
    train_f.close()
    val_f = open(os.path.join(save_path,"val_data.txt"), "w")
    for line in lines[pos:]:
        val_f.write(line)
    val_f.close()
```

### generate_rec_for_raw_user_embeding

- 输入`text.txt, author_2tower_embedding_by_photo_id_path, users_profile.txt`
- 输出`train.txt, val.txt, train.idx, train.rec, val.idx, val.rec`
- 依赖`mxnet`和`encode`函数

```python
def encode(record, index, label, data):
    #idx_file = 'tmp.idx'
    #rec_file = 'tmp.rec'
    #record = mx.recordio.MXIndexedRecordIO(idx_file, rec_file, 'w')
    header = mx.recordio.IRHeader(0, label, 0, 0)
    #mes = json.dumps([1,2,3])
    s = mx.recordio.pack(header, data)
    record.write_idx(index, s)
```

## 模型训练

### bash 运行脚本

路径：`/share/yuanwei/multimodal/run_saishi.sh`

```bash
export PATH=/opt/conda/envs/rapids/bin:$PATH
export PYTHONPATH=/opt/conda/envs/rapids:$PYTHONPATH

# export CUDA_VISIBLE_DEVICES=0,1,2,3,5,6,7
# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=29600 --use_env \
#                classy_train.py --skip_tensorboard --device=gpu --config=/share/yuanwei/xingshisenlin/saishi/configs/config_attention_ywcode_0301_data_v3.json \
# --num_workers=4 --log_freq=50 --distributed_backend=ddp --checkpoint_folder /share/yuanwei05/xingshisenlin/saishi/models/models_20210301 > /share/yuanwei05/xingshisenlin/saishi/20210301.log 2>&1
#
# mkdir /share/yuanwei/xingshisenlin/saishi/clean_data_iter/models_20210225_focus_sub_val
# mv /share/yuanwei/model_test_tmp/*.txt /share/yuanwei/xingshisenlin/saishi/clean_data_iter/models_20210225_focus_sub_val
# wait
# python -m torch.distributed.launch --nnodes=1 --nproc_per_node=8 --master_addr=localhost --master_port=29600 --use_env \
#                classy_train.py --skip_tensorboard --device=gpu --config=/share/yuanwei/xingshisenlin/saishi/configs/config_attention_ywcode_testonly_data_v2.json \
# --num_workers=4 --log_freq=50 --distributed_backend=ddp --checkpoint_folder /share/yuanwei05/xingshisenlin/saishi/models/models_20210225_focus_sub > /share/yuanwei05/xingshisenlin/saishi/testonly.log 2>&1
#
# mkdir /share/yuanwei/xingshisenlin/saishi/clean_data_iter/models_20210225_focus_sub_train
# mv /share/yuanwei/model_test_tmp/*.txt /share/yuanwei/xingshisenlin/saishi/clean_data_iter/models_20210225_focus_sub_train

sh predict_saishi_eval.sh /share/yuanwei05/xingshisenlin/saishi/eval/usemodel_0301_eval_0121-0128_prob_tmp /share/yuanwei05/xingshisenlin/saishi/models/models_20210301/checkpoint.torch 8 1

wait

#sh run_saishi_clean_data.sh
```

#### 参数说明

- --nnodes=1: 当前 job 包含一个节点
- --nproc_per_node=8: 每个节点 8 个任务
- --master_addr: master 节点的 ip
- --master_port=29600: master 节点的 port
- --use_env: 如果为 TRUE，用 LOCAL_RANK 环境变量给子进程传递 local_rank，这时候程 y 序不会传递--local_rank

- --config: classy_train 的配置文件
- num_workers: classy_vision.trainer 多线程的线程数
- --distributed_backend: 用于设置 trainer_class 为 DistributedTrainer

### 文件说明

1. 运行`/share/yuanwei/multimodal/classy_train.py`训练
2. 配置文件为`/share/yuanwei/xingshisenlin/saishi/configs/config_attention_ywcode_0301_data_v3.json`
3. 保存 checkpoint 的 checkpoint_folder 为`/share/yuanwei05/xingshisenlin/saishi/models/models_20210301`
4. 预训练 ckp 为`/share/wuxiangyu/multimodal/models_ft/finetune.pt`
5. task 文件为`fine_tuning_task_pretrain.py`，继承了`ClassificationTask(ClassyTask)`，通过`classy_train.py`中的`build_task(config)`得到
6. loss 文件为`class_multimodal_rawuser_noisetxt_noiseauthor_outtarget.py`
7. 原始 dataset 说明
   1. 会直接`self.lines = open(meta_file).readlines()`读取 meta 文件
   2. MXNet 通过索引记录文件和相应的索引文件读取
      1. `self.record = mx.recordio.MXIndexedRecordIO(self.idx_file, self.rec_file, 'r')`
      2. `self.record.read_idx(index)`
   3. 从 rec 文件中获得一些特征，text_feature，user_profile_feature，user_feature
   4. `__getitem__`返回的是`image, int(target), text_feature, user_profile_feature, torch.tensor(user_feature),int(index)`
8. classy_dataset 文件为`class_multimodal_rawuser_noisetxt_noiseauthor_outtarget.py`，
   1. 其中训练文件为`/share/yangfan/yuanwei/xingshisenlin_saishi/train_data_v3/`
   2. 其中 train 数据样例为`/share/yangfan/yuanwei/xingshisenlin_saishi/cover/11/38989219271.jpg 38989219271 38989219271 38989219271 0`
   3. dataset 类文件为`multimodal_dataset_raw_user.py`
   4. dataset同时会读取`rec`文件和`idx`文件
   5. 还会在其中创建 bert-tokenizer 和 transform
   6. 还会生成 noise words 和 drop words 和 target map
   7. 会将 dataset 中的`target`通过`"target_file": "/share/yuanwei/xingshisenlin/saishi/data/saishi_class.target`转换成 one-hot 编码的类别
   8. 会给`user_feature`添加 noise
   9. 会给`text_feature`按照 drop file 中的词进行 drop，就是将 drop file 中的词都删掉，然后在`randint(0,min(len(s),self.token_max_length-10))`位置随机加入一个 noise file 中的词。后面还会做 padding
   10. `user_profile_feature`和`text_feature`操作相同，但是不会 drop word 和 noise
   11. 最后会将`sample[0:-1]`进行 transform，然后加上`index`返回
9. meters 文件为`accuracy_meter_allownegative_dict_3.py`，只看了基本的函数，其他还没具体看，后面遇到了再回来看
10. optimizer: SGD
11. models 文件为`classy_multimodal_yw_3.py`，其中
    1. image_model：resnet，内嵌
    2. image_fc_model：mlp
    3. text_model：nlp_new,`/home/xudi06/multimodal/classy_vision/models/nlp_new.py`
       1. bert: bert-base-chinese
       2. mlp
    4. user_profile_model：nlp_new
    5. user_model：mlp
    6. fuse_model: multi_head_attention_new
    7. heads: FullyConnectedHeadFeaDict


## 模型测试和评估

测试的过程大体如下：
1. 取一批新数据，没有标注，作为测试集（非验证集）
2. 获取这些样本的data、特征等，和训练的特征一样
3. 有两种方式喂给模型
   1. 和训练方式相同：gputest，流程是：
      1. 修改gputest-config，将drop和noise比例置0
      2. 和模型训练相同，只是epoch为1
      3. 在模型训练的代码中有一段代码会保存结果在一个临时目录
        > classification_task:812
        > self.write_test[local_rank]=open('/share/yuanwei05/model_test_tmp/%s.txt'%(local_rank),'w')
   2. 使用`run_.sh->predict_.sh->test_.sh`，加载checkpoint，然后一条一条读入，进行预测，得到结果。
4. 如果是gputest方式，手动将临时结果目录下的文件，copy到自己的目录下
5. 使用`/share/yuanwei05/wudao/eval/get_shuf_202104_strategy.py`随机采样test结果，然后进行数据回查
6. 在数据回查的过程中，设置一个阈值，高于这个阈值的就为真
7. 通过阈值，每个类采样一两百给人工评，可以算出precision，近似整个测试集的precision
8. 对于训练的模型，每个phase都会保存model，checkpoint会保存最后一个phase的model，如果模型没有过拟合，那么最后一个phase即checkpoint的结果和之前最好的model，效果差不会很大，一个点左右无所谓。

最后的生成文件如下：

```bash
> ll -t /share/yangfan/yuanwei/wudao_eval_2021_0401-0403
total 34G
-rw-r--r-- 1 1001 1001  19K Apr  7 12:16 data_format_gputest.py
-rw-r--r-- 1 root root  16M Apr  7 12:14 train.idx
-rw-r--r-- 1 root root 6.6G Apr  7 12:14 train.rec
-rw-r--r-- 1 root root  91M Apr  7 12:14 train.txt
-rw-r--r-- 1 root root  51M Apr  7 12:07 train_data.txt
-rw-r--r-- 1 root root  51M Apr  7 12:04 train_org.txt
-rw-r--r-- 1 root root 1.7G Apr  7 12:03 audio_128.txt
-rw-r--r-- 1 root root  780 Apr  7 11:55 needRunVideoEmbedding.txt
-rw-r--r-- 1 root root 4.1G Apr  7 11:55 video_embedding.txt
-rw-r--r-- 1 root root  866 Apr  7 11:51 get_videoembedding.py
-rw-rw-r-- 1 1001 1001  59M Apr  7 11:09 users_profile.txt
-rw-rw-r-- 1 1001 1001  21G Apr  7 11:08 audio
-rw-rw-r-- 1 1001 1001 395M Apr  7 11:08 author_2tower_embedding_by_photo_id
-rw------- 1 1001 1001 416K Apr  7 11:08 nohup.out
-rw-rw-r-- 1 1001 1001  33M Apr  7 11:08 users_list_out.txt
-rw-rw-r-- 1 1001 1001 411M Apr  7 11:06 text.txt
drwxrwxr-x 1 1001 1001   30 Apr  7 11:04 cover/
-rw-rw-r-- 1 1001 1001 4.2M Apr  7 11:01 users_list.txt
-rw-r--r-- 1 1001 1001  701 Apr  7 11:01 run.sh
-rw-rw-r-- 1 1001 1001  24M Apr  7 10:55 users.txt
-rw-r--r-- 1 root root 8.0M Apr  7 10:51 videos.txt
-rw-r--r-- 1 root root 1.2K Apr  7 10:30 mk_train_org.py
-rw-r--r-- 1 1001 1001 5.0K Apr  7 10:28 main.py
-rw-r--r-- 1 1001 1001  863 Apr  7 10:28 find_noframe0.py
-rw-r--r-- 1 1001 1001  748 Apr  7 10:28 convert_to_128.py
-rwxr-xr-x 1 1001 1001  350 Apr  7 10:28 get_user_by_photo.sh*
```

## 推荐类别项目

### 关键词收集

- 推荐
- 必备
- 分享
- 适合
- 好物
- 推广

### 数据量

| 类别 | 已标注 | 未标注 | 共计  |
| ---- | ------ | ------ | ----- |
| 穿搭 | 9993   | 5007   | 15000 |
| 旅游 | 2845   | 8622   | 11467 |
| 美食 | 8498   | 6520   | 15018 |
| 美妆 | 9027   | 5973   | 15000 |
| 汽车 | 3522   | 11478  | 15000 |
| 亲子 | 2898   | 10514  | 13412 |
| 游戏 | 3334   | 11666  | 15000 |

