## 比赛

[千言数据集：文本相似度](https://aistudio.baidu.com/aistudio/competition/detail/45)


```python
# 正式开始实验之前首先通过如下命令安装最新版本的 paddlenlp
!pip install --upgrade paddlenlp -i https://pypi.org/simples
```


```python
import time
import os
import numpy as np

import paddle
import paddle.nn.functional as F
import paddlenlp


```

## 1. <font color='red'>第一步</font>：数据加载（注意：下面三种数据实际运行是每次只选择一种数据集运行并进行试验）


```python
user_dir = '/home/aistudio/'
traindataset = 'bq_corpus'   #选择进行训练的数据集['bq_corpus','lcqmc','paws-x']
```

### 1.1. bq_corpus 数据加载


```python
!unzip ./data/data78992/bq_corpus.zip -d ~/data/
!rm -r ./data/__MACOSX/
```

    Archive:  ./data/data78992/bq_corpus.zip
       creating: /home/aistudio/data/bq_corpus/
      inflating: /home/aistudio/data/bq_corpus/train.tsv  
       creating: /home/aistudio/data/__MACOSX/
       creating: /home/aistudio/data/__MACOSX/bq_corpus/
      inflating: /home/aistudio/data/__MACOSX/bq_corpus/._train.tsv  
      inflating: /home/aistudio/data/bq_corpus/dev.tsv  
      inflating: /home/aistudio/data/__MACOSX/bq_corpus/._dev.tsv  
      inflating: /home/aistudio/data/bq_corpus/License.pdf  
      inflating: /home/aistudio/data/__MACOSX/bq_corpus/._License.pdf  
      inflating: /home/aistudio/data/bq_corpus/test.tsv  
      inflating: /home/aistudio/data/__MACOSX/bq_corpus/._test.tsv  
      inflating: /home/aistudio/data/bq_corpus/User_Agreement.pdf  
      inflating: /home/aistudio/data/__MACOSX/bq_corpus/._User_Agreement.pdf  
      inflating: /home/aistudio/data/__MACOSX/._bq_corpus  
    /bin/sh: 1: Syntax error: EOF in backquote substitution



```python
from paddlenlp.datasets import DatasetBuilder
class bq_corpusfile(DatasetBuilder):
    SPLITS = {
        'train': os.path.join(user_dir,'data','bq_corpus','train.tsv'),
        'dev': os.path.join(user_dir,'data','bq_corpus','dev.tsv'),
        'test': os.path.join(user_dir,'data','bq_corpus','test.tsv'),

    }

    def _get_data(self, mode, **kwargs):
        filename = self.SPLITS[mode]
        return filename

    def _read(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                data = line.strip().split("\t")
                if len(data) == 3:
                    query, title, label = data
                    yield {"query": query, "title": title, "label": label}
                elif len(data) == 2:
                    query, title = data
                    yield {"query": query, "title": title, "label": ''}
                else:
                    continue

    def get_labels(self):
        return ["0", "1"]
```


```python
if traindataset == 'bq_corpus':    
    print('construct bq_corpus')
    def load_dataset(name=None,
                    data_files=None,
                    splits=None,
                    lazy=None,
                    **kwargs):
    
        reader_cls = bq_corpusfile
        print(reader_cls)
        if not name:
            reader_instance = reader_cls(lazy=lazy, **kwargs)
        else:
            reader_instance = reader_cls(lazy=lazy, name=name, **kwargs)

        datasets = reader_instance.read_datasets(data_files=data_files, splits=splits)
        return datasets

    # 一键加载 bq_corpus 的训练集、验证集
    train_ds, dev_ds = load_dataset(splits=["train", "dev"])
    # 输出训练集的前 3 条样本
    for idx, example in enumerate(train_ds):
        if idx <= 5:
            print(example)
```

    construct bq_corpus
    <class '__main__.bq_corpusfile'>
    {'query': '用微信都6年，微信没有微粒贷功能', 'title': '4。号码来微粒贷', 'label': 0}
    {'query': '微信消费算吗', 'title': '还有多少钱没还', 'label': 0}
    {'query': '交易密码忘记了找回密码绑定的手机卡也掉了', 'title': '怎么最近安全老是要改密码呢好麻烦', 'label': 0}
    {'query': '你好我昨天晚上申请的没有打电话给我今天之内一定会打吗？', 'title': '什么时候可以到账', 'label': 0}
    {'query': '“微粒贷开通"', 'title': '你好，我的微粒贷怎么没有开通呢', 'label': 0}
    {'query': '为什么借款后一直没有给我回拨电话', 'title': '怎么申请借款后没有打电话过来呢！', 'label': 1}


### 1.2. lcqmc 数据加载


```python
if traindataset == 'lcqmc':
    from paddlenlp.datasets import load_dataset
    # 一键加载 Lcqmc 的训练集、验证集
    train_ds, dev_ds = load_dataset("lcqmc", splits=["train", "dev"])
    # 输出训练集的前 3 条样本
    for idx, example in enumerate(train_ds):
        if idx <= 3:
            print(example)
```

    100%|██████████| 6827/6827 [00:00<00:00, 55738.56it/s]


    {'query': '喜欢打篮球的男生喜欢什么样的女生', 'title': '爱打篮球的男生喜欢什么样的女生', 'label': 1}
    {'query': '我手机丢了，我想换个手机', 'title': '我想买个新手机，求推荐', 'label': 1}
    {'query': '大家觉得她好看吗', 'title': '大家觉得跑男好看吗？', 'label': 0}
    {'query': '求秋色之空漫画全集', 'title': '求秋色之空全集漫画', 'label': 1}


### 1.3. paws-x-zh 数据加载


```python
!unzip ./data/data78992/paws-x-zh.zip -d ~/data/
!rm -r ./data/__MACOSX/
```

    Archive:  ./data/data78992/paws-x-zh.zip
       creating: /home/aistudio/data/paws-x-zh/
      inflating: /home/aistudio/data/paws-x-zh/train.tsv  
       creating: /home/aistudio/data/__MACOSX/
       creating: /home/aistudio/data/__MACOSX/paws-x-zh/
      inflating: /home/aistudio/data/__MACOSX/paws-x-zh/._train.tsv  
      inflating: /home/aistudio/data/paws-x-zh/dev.tsv  
      inflating: /home/aistudio/data/__MACOSX/paws-x-zh/._dev.tsv  
      inflating: /home/aistudio/data/paws-x-zh/License.pdf  
      inflating: /home/aistudio/data/__MACOSX/paws-x-zh/._License.pdf  
      inflating: /home/aistudio/data/paws-x-zh/test.tsv  
      inflating: /home/aistudio/data/__MACOSX/paws-x-zh/._test.tsv  
      inflating: /home/aistudio/data/__MACOSX/._paws-x-zh  



```python
from paddlenlp.datasets import DatasetBuilder
class paws_x_zhfile(DatasetBuilder):
    SPLITS = {
        'train': os.path.join(user_dir,'data','paws-x-zh','train.tsv'),
        'dev': os.path.join(user_dir,'data','paws-x-zh','dev.tsv'),
        'test': os.path.join(user_dir,'data','paws-x-zh','test.tsv'),

    }

    def _get_data(self, mode, **kwargs):
        filename = self.SPLITS[mode]
        return filename

    def _read(self, filename):
        with open(filename, 'r', encoding='utf-8') as f:
            for line in f:
                data = line.strip().split("\t")
                if len(data) == 3:
                    query, title, label = data
                    yield {"query": query, "title": title, "label": label}
                elif len(data) == 2:
                    query, title = data
                    yield {"query": query, "title": title, "label": ''}
                else:
                    continue

    def get_labels(self):
        return ["0", "1"]
```


```python
if traindataset == 'paws-x':    
    print('construct paws-x-zh')
    def load_dataset(name=None,
                    data_files=None,
                    splits=None,
                    lazy=None,
                    **kwargs):
    
        reader_cls = paws_x_zhfile
        print(reader_cls)
        if not name:
            reader_instance = reader_cls(lazy=lazy, **kwargs)
        else:
            reader_instance = reader_cls(lazy=lazy, name=name, **kwargs)

        datasets = reader_instance.read_datasets(data_files=data_files, splits=splits)
        return datasets

    # 一键加载 bq_corpus 的训练集、验证集
    train_ds, dev_ds = load_dataset(splits=["train", "dev"])
    # 输出训练集的前 3 条样本
    for idx, example in enumerate(train_ds):
        if idx <= 5:
            print(example)
```

    construct paws-x-zh
    <class '__main__.paws_x_zhfile'>
    {'query': '1560年10月，他在巴黎秘密会见了英国大使Nicolas Throckmorton，要求他通过苏格兰返回英国。', 'title': '1560年10月，他在巴黎秘密会见了英国大使尼古拉斯·斯罗克莫顿，并要求他通过英格兰返回苏格兰的护照。', 'label': 0}
    {'query': '1975年的NBA赛季 -  76赛季是全美篮球协会的第30个赛季。', 'title': '1975-76赛季的全国篮球协会是NBA的第30个赛季。', 'label': 1}
    {'query': '还有具体的讨论，公众形象辩论和项目讨论。', 'title': '还有公开讨论，特定档案讨论和项目讨论。', 'label': 0}
    {'query': '当可以保持相当的流速时，结果很高。', 'title': '当可以保持可比较的流速时，结果很高。', 'label': 1}
    {'query': '它是Akmola地区Zerendi区的所在地。', 'title': '它是Akmola地区Zerendi区的所在地。', 'label': 1}
    {'query': '威廉亨利亨利哈曼于1828年2月17日出生在弗吉尼亚州的韦恩斯伯勒，他的父母是刘易斯和莎莉（加伯）哈曼。', 'title': '威廉亨利哈曼于1828年2月17日出生于弗吉尼亚州韦恩斯伯勒。他的父母是刘易斯和莎莉（加伯）哈曼。', 'label': 1}



通过 PaddleNLP 加载进来的 [LCQMC](http://icrc.hitsz.edu.cn/Article/show/171.html) 数据集是原始的明文数据集.

#### 定义样本转换函数


```python
 #查看可以使用的模型
#  dir(paddlenlp.transformers)
```


```python
# 因为是基于预训练模型 ERNIE-Gram 来进行，所以需要首先加载 ERNIE-Gram 的 tokenizer，
# 后续样本转换函数基于 tokenizer 对文本进行切分

# MODEL_NAME = "ernie-gram-zh"
MODEL_NAME = "ernie-1.0"
# MODEL_NAME = "roberta-wwm-ext-large"
# MODEL_NAME = "roberta-wwm-ext"

# tokenizer = paddlenlp.transformers.ErnieGramTokenizer.from_pretrained(MODEL_NAME)  #ernie-gram相关的字符编码
tokenizer = paddlenlp.transformers.ErnieTokenizer.from_pretrained(MODEL_NAME)    #ernie相关的字符编码
# tokenizer = paddlenlp.transformers.RobertaTokenizer.from_pretrained(MODEL_NAME)    #roberta相关的字符编码
```

    [2021-06-16 16:18:47,811] [    INFO] - Found /home/aistudio/.paddlenlp/models/ernie-1.0/vocab.txt



```python
print(tokenizer.pad_token_id)
print(tokenizer.pad_token_type_id)
```

    0
    0


### 2. <font color='red'>第二步</font>： 构造<font color='red'>convert_example函数</font>，用于将<font color='red'>单条原始数据</font>根据需求组装并<font color='red'>根据tokenizer进行编码</font>


```python
# 将 1 条明文数据的 query、title 拼接起来，根据预训练模型的 tokenizer 将明文转换为 ID 数据
# 返回 input_ids 和 token_type_ids

def convert_example(example, tokenizer, max_seq_length=512, is_test=False):

    query, title = example["query"], example["title"]

    encoded_inputs = tokenizer(
        text=query, text_pair=title, max_seq_len=max_seq_length)

    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]

    if not is_test:
        label = np.array([example["label"]], dtype="int64")
        return input_ids, token_type_ids, label
    # 在预测或者评估阶段，不返回 label 字段
    else:
        return input_ids, token_type_ids
```


```python
### 对训练集的第 1 条数据进行转换
# input_ids, token_type_ids, label = convert_example(train_ds[0], tokenizer)
# print(input_ids)
# print(token_type_ids)
# print(label)
```

    [1, 6992, 17, 530, 136, 4, 44, 11, 822, 1756, 890, 632, 32, 373, 15, 514, 20, 19, 175, 6065, 9625, 10300, 2052, 10508, 9844, 9724, 9691, 9516, 4, 41, 323, 44, 124, 93, 733, 318, 784, 1593, 381, 514, 20, 12043, 2, 6992, 17, 530, 136, 4, 44, 11, 822, 1756, 890, 632, 32, 373, 15, 514, 20, 19, 175, 919, 422, 630, 431, 17963, 431, 700, 492, 1240, 1410, 4, 145, 41, 323, 44, 124, 93, 514, 318, 784, 1593, 381, 733, 318, 784, 5, 455, 575, 12043, 2]
    [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
    [0]



```python
# 为了后续方便使用，我们使用python偏函数（partial）给 convert_example 赋予一些默认参数
from functools import partial

# 训练集和验证集的样本转换函数
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=512)
```

### 3. <font color='red'>第三步</font>：将多个单条数据打包组装成 Batch 数据 并顺便根据自定义最大的单条数据长度在末尾补齐数据，即进行 Padding

### 3.1 数据 pipeline相关的基本示例

上一小节，我们完成了对单条样本的转换，本节我们需要将样本组合成 Batch 数据，对于不等长的数据还需要进行 Padding 操作，便于 GPU 训练。

PaddleNLP 提供了许多关于 NLP 任务中构建有效的数据 pipeline 的常用 API

| API                             | 简介                                       |
| ------------------------------- | :----------------------------------------- |
| `paddlenlp.data.Stack`          | 堆叠N个具有相同shape的输入数据来构建一个batch |
| `paddlenlp.data.Pad`            | 将长度不同的多个句子padding到统一长度，取N个输入数据中的最大长度 |
| `paddlenlp.data.Tuple`          | 将多个batchify函数包装在一起 |

更多数据处理操作详见： [https://paddlenlp.readthedocs.io/zh/latest/data_prepare/data_preprocess.html](https://paddlenlp.readthedocs.io/zh/latest/data_prepare/data_preprocess.html)


```python
# from paddlenlp.data import Stack, Pad, Tuple
# a = [1, 2, 3, 4]
# b = [3, 4, 5, 6]
# c = [5, 6, 7, 8]
# result = Stack()([a, b, c])
# print("Stacked Data: \n", result)
# print()

# a = [1, 2, 3, 4]
# b = [5, 6, 7]
# c = [8, 9]
# result = Pad(pad_val=0)([a, b, c])
# print("Padded Data: \n", result)
# print()

# data = [
#         [[1, 2, 3, 4], [1]],
#         [[5, 6, 7], [0]],
#         [[8, 9], [1]],
#        ]
# batchify_fn = Tuple(Pad(pad_val=0), Stack())
# ids, labels = batchify_fn(data)
# print("ids: \n", ids)
# print()
# print("labels: \n", labels)
# print()
```

    Stacked Data: 
     [[1 2 3 4]
     [3 4 5 6]
     [5 6 7 8]]
    
    Padded Data: 
     [[1 2 3 4]
     [5 6 7 0]
     [8 9 0 0]]
    
    ids: 
     [[1 2 3 4]
     [5 6 7 0]
     [8 9 0 0]]
    
    labels: 
     [[1]
     [0]
     [1]]
    



```python
# 我们的训练数据会返回 input_ids, token_type_ids, labels 3 个字段
# 因此针对这 3 个字段需要分别定义 3 个组 batch 操作
from paddlenlp.data import Stack, Pad, Tuple
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # token_type_ids
    Stack(dtype="int64")  # label
): [data for data in fn(samples)]
```

#### 3.2 定义 Dataloader
下面我们基于组 batchify_fn 函数和样本转换函数 trans_func 来构造训练集的 DataLoader, 支持多卡训练



```python

# 定义分布式 Sampler: 自动对训练数据进行切分，支持多卡并行训练
batch_sampler = paddle.io.DistributedBatchSampler(train_ds, batch_size=32, shuffle=True)

# 基于 train_ds 定义 train_data_loader
# 因为我们使用了分布式的 DistributedBatchSampler, train_data_loader 会自动对训练数据进行切分
train_data_loader = paddle.io.DataLoader(
        dataset=train_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)

# 针对验证集数据加载，我们使用单卡进行评估，所以采用 paddle.io.BatchSampler 即可
# 定义 dev_data_loader
batch_sampler = paddle.io.BatchSampler(dev_ds, batch_size=32, shuffle=False)
dev_data_loader = paddle.io.DataLoader(
        dataset=dev_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)
```


```python
#打印一条数据数据查看
# for i in dev_data_loader:
#     print(i[0][1])
#     print(i[1][1])
#     print(i[2][1])
#     break
```

    Tensor(shape=[52], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
           [1   , 1022, 264 , 151 , 237 , 10  , 305 , 742 , 188 , 291 , 1114, 12045, 2   , 329 , 182 , 47  , 1513, 661 , 1022, 1877, 165 , 32  , 1917, 10  , 614 , 356 , 670 , 2   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   , 0   ])
    Tensor(shape=[52], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
           [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])
    Tensor(shape=[1], dtype=int64, place=CUDAPlace(0), stop_gradient=True,
           [0])



```python
# dir(paddlenlp.transformers)
```

### 2.3 模型搭建

自从 2018 年 10 月以来，NLP 个领域的任务都通过 Pretrain + Finetune 的模式相比传统 DNN 方法在效果上取得了显著的提升，本节我们以百度开源的预训练模型 ERNIE-Gram 为基础模型，在此之上构建 Point-wise 语义匹配网络。

首先我们来定义网络结构:

#### 2.3.1 方式一：使用基本预训练模型自定义下游任务搭建法 


```python
import paddle.nn as nn

# 我们基于 ERNIE-Gram 模型结构搭建 Point-wise 语义匹配网络
# 所以此处先定义 ERNIE-Gram 的 pretrained_model
# pretrained_model = paddlenlp.transformers.ErnieGramModel.from_pretrained(MODEL_NAME)
# pretrained_model = paddlenlp.transformers.ErnieModel.from_pretrained(MODEL_NAME)


class PointwiseMatching(nn.Layer):
   
    # 此处的 pretained_model 在本例中会被 ERNIE-Gram 预训练模型初始化
    def __init__(self, pretrained_model, dropout=None):
        super().__init__()
        self.ptm = pretrained_model
        self.dropout = nn.Dropout(dropout if dropout is not None else 0.1)

        # 语义匹配任务: 相似、不相似 2 分类任务
        self.classifier = nn.Linear(self.ptm.config["hidden_size"], 2)

    def forward(self,
                input_ids,
                token_type_ids=None,
                position_ids=None,
                attention_mask=None):

        # 此处的 Input_ids 由两条文本的 token ids 拼接而成
        # token_type_ids 表示两段文本的类型编码
        # 返回的 cls_embedding 就表示这两段文本经过模型的计算之后而得到的语义表示向量
        _, cls_embedding = self.ptm(input_ids, token_type_ids, position_ids,
                                    attention_mask)

        cls_embedding = self.dropout(cls_embedding)

        # 基于文本对的语义表示向量进行 2 分类任务
        logits = self.classifier(cls_embedding)
        # probs = F.softmax(logits)

        return logits

# 定义 Point-wise 语义匹配网络
model = PointwiseMatching(pretrained_model)
```

    [2021-06-13 19:07:32,196] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-1.0/ernie_v1_chn_base.pdparams
    [2021-06-13 19:07:38,468] [    INFO] - Weights from pretrained model not used in ErnieModel: ['cls.predictions.layer_norm.weight', 'cls.predictions.decoder_bias', 'cls.predictions.transform.bias', 'cls.predictions.transform.weight', 'cls.predictions.layer_norm.bias']


<font color='red' size=4>读[源码](https://gitee.com/paddlepaddle/PaddleNLP/blob/develop/paddlenlp/transformers/ernie/modeling.py)</font>

<font color='red' size=4>可见 ErnieForSequenceClassification就等价于PointwiseMatching（单塔模型）！！！！只不过PointwiseMatching输出是概率，反而不能和损失函数对应了！！！！</font>

<font color='red' size=4>之后调优时使用下面的模型代替上面的模型！！！</font>

#### 2.3.2 方式二：使用已有的序列分类预训练模型下游任务搭建法 


```python
# print(paddlenlp.transformers.RobertaForSequenceClassification.from_pretrained.__doc__)
```


```python
# 搭建模型
# model = paddlenlp.transformers.ErnieGramForSequenceClassification.from_pretrained(MODEL_NAME,num_classes=2)  #ernie-gram进行序列分类相关的模型
model = paddlenlp.transformers.ErnieForSequenceClassification.from_pretrained(MODEL_NAME,num_classes=2)  #ernie进行序列分类相关的模型
# model = paddlenlp.transformers.RobertaForSequenceClassification.from_pretrained(MODEL_NAME,num_classes=2)  #roberta进行序列分类的模型
```

    [2021-06-16 16:19:08,514] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-1.0/ernie_v1_chn_base.pdparams
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.weight. classifier.weight is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.bias. classifier.bias is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))


### 2.4 模型训练 & 评估


```python
from paddlenlp.transformers import LinearDecayWithWarmup

epochs = 5
num_training_steps = len(train_data_loader) * epochs

# 定义 learning_rate_scheduler，负责在训练过程中对 lr 进行调度
lr_scheduler = LinearDecayWithWarmup(5E-5, num_training_steps, 0.0)
# lr_scheduler = LinearDecayWithWarmup(1E-3, num_training_steps, 0.0)

# Generate parameter names needed to perform weight decay.
# All bias and LayerNorm parameters are excluded.
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]

# 定义 Optimizer
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    weight_decay=0.01,
    apply_decay_param_fun=lambda x: x in decay_params)

# 采用交叉熵 损失函数
criterion = paddle.nn.loss.CrossEntropyLoss()

# 评估的时候采用准确率指标
metric = paddle.metric.Accuracy()
```


```python
# 因为训练过程中同时要在验证集进行模型评估，因此我们先定义评估函数
import paddle.nn.functional as F

@paddle.no_grad()
def evaluate(model, criterion, metric, data_loader, phase="dev"):
    model.eval()
    metric.reset()
    losses = []
    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        logits = model(input_ids=input_ids, token_type_ids=token_type_ids)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        accu = metric.accumulate()
    print("eval {} loss: {:.5}, accu: {:.5}".format(phase,
                                                    np.mean(losses), accu))
    model.train()
    metric.reset()
    return accu
```


```python
# 接下来，开始正式训练模型，训练时间较长，可注释掉这部分

global_step = 0
global_accu = 0.0
tic_train = time.time()

for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):

        input_ids, token_type_ids, labels = batch
        logits = model(input_ids=input_ids, token_type_ids=token_type_ids)
        loss = criterion(logits, labels)
        probs = F.softmax(logits, axis=1)
        correct = metric.compute(probs, labels)
        metric.update(correct)
        acc = metric.accumulate()

        global_step += 1
        
        # 每间隔 10 step 输出训练指标
        if global_step % 50 == 0:
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, accu: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, loss, acc,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_grad()

        # 每间隔 100 step 在验证集和测试集上进行评估
        if global_step % 500 == 0:
            eva_accu = evaluate(model, criterion, metric, dev_data_loader, "dev")
            #训练过程中保存最大验证结果的模型
            if eva_accu > global_accu:   
                print(f'evaluate accu: {eva_accu}>history accu:{global_accu} ==> save the model!')
                global_accu = eva_accu
                save_dir = os.path.join(f'checkpoint_{traindataset}_{MODEL_NAME}', "model_%d_%.3f" % (global_step,global_accu))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_param_path = os.path.join(save_dir, 'model_state.pdparams')
                paddle.save(model.state_dict(), save_param_path)
                # tokenizer.save_pretrained(save_dir)

# 训练结束后，存储模型参数
save_dir = os.path.join(f'checkpoint_{traindataset}_{MODEL_NAME}', "model_%d" % global_step)
os.makedirs(save_dir)

save_param_path = os.path.join(save_dir, 'model_state.pdparams')
paddle.save(model.state_dict(), save_param_path)
# tokenizer.save_pretrained(save_dir)
```

模型训练过程中会输出如下日志:
```
global step 5310, epoch: 3, batch: 1578, loss: 0.31671, accu: 0.95000, speed: 0.63 step/s
global step 5320, epoch: 3, batch: 1588, loss: 0.36240, accu: 0.94063, speed: 6.98 step/s
global step 5330, epoch: 3, batch: 1598, loss: 0.41451, accu: 0.93854, speed: 7.40 step/s
global step 5340, epoch: 3, batch: 1608, loss: 0.31327, accu: 0.94063, speed: 7.01 step/s
global step 5350, epoch: 3, batch: 1618, loss: 0.40664, accu: 0.93563, speed: 7.83 step/s
global step 5360, epoch: 3, batch: 1628, loss: 0.33064, accu: 0.93958, speed: 7.34 step/s
global step 5370, epoch: 3, batch: 1638, loss: 0.38411, accu: 0.93795, speed: 7.72 step/s
global step 5380, epoch: 3, batch: 1648, loss: 0.35376, accu: 0.93906, speed: 7.92 step/s
global step 5390, epoch: 3, batch: 1658, loss: 0.39706, accu: 0.93924, speed: 7.47 step/s
global step 5400, epoch: 3, batch: 1668, loss: 0.41198, accu: 0.93781, speed: 7.41 step/s
eval dev loss: 0.4177, accu: 0.89082
global step 5410, epoch: 3, batch: 1678, loss: 0.34453, accu: 0.93125, speed: 0.63 step/s
global step 5420, epoch: 3, batch: 1688, loss: 0.34569, accu: 0.93906, speed: 7.75 step/s
global step 5430, epoch: 3, batch: 1698, loss: 0.39160, accu: 0.92917, speed: 7.54 step/s
global step 5440, epoch: 3, batch: 1708, loss: 0.46002, accu: 0.93125, speed: 7.05 step/s
global step 5450, epoch: 3, batch: 1718, loss: 0.32302, accu: 0.93188, speed: 7.14 step/s
global step 5460, epoch: 3, batch: 1728, loss: 0.40802, accu: 0.93281, speed: 7.22 step/s
global step 5470, epoch: 3, batch: 1738, loss: 0.34607, accu: 0.93348, speed: 7.44 step/s
global step 5480, epoch: 3, batch: 1748, loss: 0.34709, accu: 0.93398, speed: 7.38 step/s
global step 5490, epoch: 3, batch: 1758, loss: 0.31814, accu: 0.93437, speed: 7.39 step/s
global step 5500, epoch: 3, batch: 1768, loss: 0.42689, accu: 0.93125, speed: 7.74 step/s
eval dev loss: 0.41789, accu: 0.88968
```

基于默认参数配置进行单卡训练大概要持续 4 个小时左右，会训练完成 3 个 Epoch, 模型最终的收敛指标结果如下:


| 数据集 | Accuracy |
| -------- | -------- |
| dev.tsv     | 89.62  |

可以看到: 我们基于 PaddleNLP ，利用 ERNIE-Gram 预训练模型使用非常简洁的代码，就在权威语义匹配数据集上取得了很不错的效果.

### 2.5 模型预测



#### 2.5.2 预测lcqmc

接下来我们使用已经训练好的语义匹配模型对一些预测数据进行预测。待预测数据为每行都是文本对的 tsv 文件，我们使用 Lcqmc 数据集的测试集作为我们的预测数据，进行预测并提交预测结果到 [千言文本相似度竞赛](https://aistudio.baidu.com/aistudio/competition/detail/45)

下载我们已经训练好的语义匹配模型, 并解压


```python
# 下载我们基于 Lcqmc 事先训练好的语义匹配模型并解压
! wget https://paddlenlp.bj.bcebos.com/models/text_matching/ernie_gram_zh_pointwise_matching_model.tar
! tar -xvf ernie_gram_zh_pointwise_matching_model.tar
```

    --2021-06-09 22:39:51--  https://paddlenlp.bj.bcebos.com/models/text_matching/ernie_gram_zh_pointwise_matching_model.tar
    Resolving paddlenlp.bj.bcebos.com (paddlenlp.bj.bcebos.com)... 182.61.200.229, 182.61.200.195, 2409:8c00:6c21:10ad:0:ff:b00e:67d
    Connecting to paddlenlp.bj.bcebos.com (paddlenlp.bj.bcebos.com)|182.61.200.229|:443... connected.
    HTTP request sent, awaiting response... 200 OK
    Length: 597667840 (570M) [application/x-tar]
    Saving to: ‘ernie_gram_zh_pointwise_matching_model.tar.1’
    
    ernie_gram_zh_point 100%[===================>] 569.98M  23.7MB/s    in 17s     
    
    2021-06-09 22:40:10 (33.4 MB/s) - ‘ernie_gram_zh_pointwise_matching_model.tar.1’ saved [597667840/597667840]
    
    ernie_gram_zh_pointwise_matching_model/
    ernie_gram_zh_pointwise_matching_model/model_state.pdparams
    ernie_gram_zh_pointwise_matching_model/vocab.txt
    ernie_gram_zh_pointwise_matching_model/tokenizer_config.json



```python
# 测试数据由 2 列文本构成 tab 分隔
# Lcqmc 默认下载到如下路径
! head -n3 "${HOME}/.paddlenlp/datasets/LCQMC/lcqmc/lcqmc/test.tsv"
```

    谁有狂三这张高清的	这张高清图，谁有
    英雄联盟什么英雄最好	英雄联盟最好英雄是什么
    这是什么意思，被蹭网吗	我也是醉了，这是什么意思


#### 定义预测函数


```python
import paddle.nn.functional as F

def predict(model, data_loader):
    
    batch_probs = []

    # 预测阶段打开 eval 模式，模型中的 dropout 等操作会关掉
    model.eval()

    with paddle.no_grad():
        for batch_data in data_loader:
            input_ids, token_type_ids = batch_data
            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)
            
            # 获取每个样本的预测概率: [batch_size, 2] 的矩阵
            logits = model(
                input_ids=input_ids, token_type_ids=token_type_ids)
            # print(logits)
            batch_prob = F.softmax(logits, axis=1).numpy()

            batch_probs.append(batch_prob)
        batch_probs = np.concatenate(batch_probs, axis=0)

        return batch_probs
```

#### 定义预测数据的 data_loader


```python
# 预测数据的转换函数
# predict 数据没有 label, 因此 convert_exmaple 的 is_test 参数设为 True
trans_func = partial(
    convert_example,
    tokenizer=tokenizer,
    max_seq_length=512,
    is_test=True)

# 预测数据的组 batch 操作
# predict 数据只返回 input_ids 和 token_type_ids，因此只需要 2 个 Pad 对象作为 batchify_fn
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
    Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment_ids
): [data for data in fn(samples)]

# 加载预测数据
if traindataset == 'lcqmc':
    test_ds = load_dataset("lcqmc", splits=["test"])
    print('load lcqmc test dataset finished!')
else:
    test_ds = load_dataset(splits=['test']) #加载bq_corpus
    print('load not lcqmc test dataset finished!')

```

    <class '__main__.bq_corpusfile'>
    load not lcqmc test dataset finished!



```python
test_ds[:3]
```




    [{'query': '为什么我无法看到额度', 'title': '为什么开通了却没有额度', 'label': ''},
     {'query': '为啥换不了', 'title': '为两次还都提示失败呢', 'label': ''},
     {'query': '借了钱，但还没有通过，可以取消吗？', 'title': '可否取消', 'label': ''}]




```python
batch_sampler = paddle.io.BatchSampler(test_ds, batch_size=32, shuffle=False)

# 生成预测数据 data_loader
predict_data_loader =paddle.io.DataLoader(
        dataset=test_ds.map(trans_func),
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)
```

#### 定义预测模型


```python
# 搭建模型
# model = paddlenlp.transformers.ErnieGramForSequenceClassification.from_pretrained(MODEL_NAME,num_classes=2)  #ernie-gram进行序列分类相关的模型
model = paddlenlp.transformers.ErnieForSequenceClassification.from_pretrained(MODEL_NAME,num_classes=2)  #ernie进行序列分类相关的模型
# model = paddlenlp.transformers.RobertaForSequenceClassification.from_pretrained(MODEL_NAME,num_classes=2)  #roberta进行序列分类的模型

# model = PointwiseMatching(pretrained_model)

```

    [2021-06-16 16:19:35,507] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-1.0/ernie_v1_chn_base.pdparams


#### 加载已训练好的模型参数


```python
# 刚才下载的模型解压之后存储路径为 ./ernie_gram_zh_pointwise_matching_model/model_state.pdparams
state_dict = paddle.load(os.path.join(f'checkpoint_{traindataset}_{MODEL_NAME}', 'model_9500_0.858','model_state.pdparams'))


# 刚才下载的模型解压之后存储路径为 ./pointwise_matching_model/ernie1.0_base_pointwise_matching.pdparams
# state_dict = paddle.load("pointwise_matching_model/ernie1.0_base_pointwise_matching.pdparams")
model.set_dict(state_dict)
```

#### 开始预测


```python
for idx, batch in enumerate(predict_data_loader):
    if idx < 1:
        print(batch)
```

    [Tensor(shape=[32, 48], dtype=int64, place=CUDAPinnedPlace, stop_gradient=True,
           [[1   , 13  , 614 , ..., 0   , 0   , 0   ],
            [1   , 13  , 3221, ..., 0   , 0   , 0   ],
            [1   , 1051, 15  , ..., 0   , 0   , 0   ],
            ...,
            [1   , 13  , 614 , ..., 632 , 1086, 2   ],
            [1   , 16  , 1051, ..., 0   , 0   , 0   ],
            [1   , 413 , 41  , ..., 0   , 0   , 0   ]]), Tensor(shape=[32, 48], dtype=int64, place=CUDAPinnedPlace, stop_gradient=True,
           [[0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0],
            ...,
            [0, 0, 0, ..., 1, 1, 1],
            [0, 0, 0, ..., 0, 0, 0],
            [0, 0, 0, ..., 0, 0, 0]])]



```python
# 执行预测函数
y_probs = predict(model, predict_data_loader)

# 根据预测概率获取预测 label
y_preds = np.argmax(y_probs, axis=1)
```

#### 输出预测结果


```python
# 我们按照千言文本相似度竞赛的提交格式将预测结果存储在 lcqmc.tsv 中，用来后续提交
# 同时将预测结果输出到终端，便于大家直观感受模型预测效果
if traindataset == 'lcqmc':
    test_ds = load_dataset("lcqmc", splits=["test"]) #加载lcqmc
else:
    test_ds = load_dataset(splits=['test']) #加载bq_corpus

with open(f"{traindataset}.tsv", 'w', encoding="utf-8") as f:
    f.write("index\tprediction\n")    
    for idx, y_pred in enumerate(y_preds):
        f.write("{}\t{}\n".format(idx, y_pred))
        text_pair = test_ds[idx]
        text_pair["label"] = y_pred
        if idx % 100 ==0:  #每隔100条打印一次
            print(text_pair)
print(f"{traindataset}.tsv save finished!")
```

#### 提交 LCQMC 预测结果[千言文本相似度竞赛](https://aistudio.baidu.com/aistudio/competition/detail/45)

千言文本相似度竞赛一共有 3 个数据集: lcqmc、bq_corpus、paws-x, 我们刚才生成了 lcqmc 的预测结果 lcqmc.tsv, 同时我们在项目内提供了 bq_corpus、paw-x 数据集的空预测结果，我们将这 3 个文件打包提交到千言文本相似度竞赛，即可看到自己的模型在 Lcqmc 数据集上的竞赛成绩。




```python
# 打包预测结果
!zip submit.zip lcqmc.tsv paws-x.tsv bq_corpus.tsv
```

    updating: lcqmc.tsv (deflated 65%)
    updating: paws-x.tsv (deflated 64%)
    updating: bq_corpus.tsv (deflated 65%)


##### 提交预测结果 submit.zip 到 [千言文本相似度竞赛](https://aistudio.baidu.com/aistudio/competition/detail/45)


[返回](/PaddleNLP_Homework)
