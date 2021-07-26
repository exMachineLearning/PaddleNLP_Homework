# [千言数据集：信息抽取](https://aistudio.baidu.com/aistudio/competition/detail/46)之——DUIE<font color='color'>注释版本</font>

注意：本项目fork了项目[『NLP打卡营』实践课4 基于预训练模型完成实体关系抽取](https://aistudio.baidu.com/aistudio/projectdetail/1639963),不过在此基础上<font color="red">增加了阅读代码时的注释</font>

信息抽取旨在从非结构化自然语言文本中提取结构化知识，如实体、关系、事件等。对于给定的自然语言句子，根据预先定义的schema集合，抽取出所有满足schema约束的SPO三元组。

例如，「妻子」关系的schema定义为：      
{      
    S_TYPE: 人物,        
    P: 妻子,      
    O_TYPE: {      
        @value: 人物       
    }       
}        

该示例展示了如何使用PaddleNLP快速完成实体关系抽取，参与[千言信息抽取-关系抽取比赛](https://aistudio.baidu.com/aistudio/competition/detail/46)打榜。





```python
# 安装paddlenlp最新版本
!pip install --upgrade paddlenlp

%cd relation_extraction/
```

    Looking in indexes: https://mirror.baidu.com/pypi/simple/
    Collecting paddlenlp
    [?25l  Downloading https://mirror.baidu.com/pypi/packages/62/10/ccc761d3e3a994703f31a4d0f93db0d13789d1c624a0cbbe9fe6439ed601/paddlenlp-2.0.5-py3-none-any.whl (435kB)
    [K     |████████████████████████████████| 440kB 13.1MB/s eta 0:00:01
    [?25hRequirement already satisfied, skipping upgrade: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (4.1.0)
    Requirement already satisfied, skipping upgrade: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.4.4)
    Requirement already satisfied, skipping upgrade: multiprocess in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.70.11.1)
    Requirement already satisfied, skipping upgrade: visualdl in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.2.0)
    Requirement already satisfied, skipping upgrade: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.9.0)
    Requirement already satisfied, skipping upgrade: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.42.1)
    Requirement already satisfied, skipping upgrade: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (1.2.2)
    Requirement already satisfied, skipping upgrade: dill>=0.3.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from multiprocess->paddlenlp) (0.3.3)
    Requirement already satisfied, skipping upgrade: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.20.3)
    Requirement already satisfied, skipping upgrade: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (3.8.2)
    Requirement already satisfied, skipping upgrade: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (0.7.1.1)
    Requirement already satisfied, skipping upgrade: pandas in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.1.5)
    Requirement already satisfied, skipping upgrade: matplotlib in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (2.2.3)
    Requirement already satisfied, skipping upgrade: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.21.0)
    Requirement already satisfied, skipping upgrade: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (7.1.2)
    Requirement already satisfied, skipping upgrade: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (3.14.0)
    Requirement already satisfied, skipping upgrade: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.0.0)
    Requirement already satisfied, skipping upgrade: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.1.1)
    Requirement already satisfied, skipping upgrade: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (2.22.0)
    Requirement already satisfied, skipping upgrade: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (0.8.53)
    Requirement already satisfied, skipping upgrade: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.15.0)
    Requirement already satisfied, skipping upgrade: scikit-learn>=0.21.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from seqeval->paddlenlp) (0.24.2)
    Requirement already satisfied, skipping upgrade: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (0.6.1)
    Requirement already satisfied, skipping upgrade: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (2.2.0)
    Requirement already satisfied, skipping upgrade: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (0.23)
    Requirement already satisfied, skipping upgrade: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (2.6.0)
    Requirement already satisfied, skipping upgrade: pytz>=2017.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->visualdl->paddlenlp) (2019.3)
    Requirement already satisfied, skipping upgrade: python-dateutil>=2.7.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pandas->visualdl->paddlenlp) (2.8.0)
    Requirement already satisfied, skipping upgrade: kiwisolver>=1.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (1.1.0)
    Requirement already satisfied, skipping upgrade: cycler>=0.10 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (0.10.0)
    Requirement already satisfied, skipping upgrade: pyparsing!=2.0.4,!=2.1.2,!=2.1.6,>=2.0.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from matplotlib->visualdl->paddlenlp) (2.4.2)
    Requirement already satisfied, skipping upgrade: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (0.10.0)
    Requirement already satisfied, skipping upgrade: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (5.1.2)
    Requirement already satisfied, skipping upgrade: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (16.7.9)
    Requirement already satisfied, skipping upgrade: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.4.10)
    Requirement already satisfied, skipping upgrade: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.3.0)
    Requirement already satisfied, skipping upgrade: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (2.0.1)
    Requirement already satisfied, skipping upgrade: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.3.4)
    Requirement already satisfied, skipping upgrade: Jinja2>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp) (2.10.1)
    Requirement already satisfied, skipping upgrade: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp) (2.8.0)
    Requirement already satisfied, skipping upgrade: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (0.16.0)
    Requirement already satisfied, skipping upgrade: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (1.1.0)
    Requirement already satisfied, skipping upgrade: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (7.0)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (2019.9.11)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (3.0.4)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (2.8)
    Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (1.25.6)
    Requirement already satisfied, skipping upgrade: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp) (0.18.0)
    Requirement already satisfied, skipping upgrade: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp) (3.9.9)
    Requirement already satisfied, skipping upgrade: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (2.1.0)
    Requirement already satisfied, skipping upgrade: scipy>=0.19.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (1.6.3)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (0.14.1)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl->paddlenlp) (0.6.0)
    Requirement already satisfied, skipping upgrade: setuptools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from kiwisolver>=1.0.1->matplotlib->visualdl->paddlenlp) (56.2.0)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.5->Flask-Babel>=1.0.0->visualdl->paddlenlp) (1.1.1)
    Requirement already satisfied, skipping upgrade: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->flake8>=3.7.9->visualdl->paddlenlp) (7.2.0)
    Installing collected packages: paddlenlp
      Found existing installation: paddlenlp 2.0.1
        Uninstalling paddlenlp-2.0.1:
          Successfully uninstalled paddlenlp-2.0.1
    Successfully installed paddlenlp-2.0.5
    /home/aistudio/relation_extraction


## 关系抽取介绍

针对 DuIE2.0 任务中多条、交叠SPO这一抽取目标，比赛对标准的 'BIO' 标注进行了扩展。
对于每个 token，根据其在实体span中的位置（包括B、I、O三种），我们为其打上三类标签，并且根据其所参与构建的predicate种类，将 B 标签进一步区分。给定 schema 集合，对于 N 种不同 predicate，以及头实体/尾实体两种情况，我们设计对应的共 2*N 种 B 标签，再合并 I 和 O 标签，故每个 token 一共有 (2*N+2) 个标签，如下图所示。


<div align="center">
<img src="https://ai-studio-static-online.cdn.bcebos.com/f984664777b241a9b43ef843c9b752f33906c8916bc146a69f7270b5858bee63" width="500" height="400" alt="标注策略" align=center />
</div>

### 评价方法

对测试集上参评系统输出的SPO结果和人工标注的SPO结果进行精准匹配，采用F1值作为评价指标。注意，对于复杂O值类型的SPO，必须所有槽位都精确匹配才认为该SPO抽取正确。针对部分文本中存在实体别名的问题，使用百度知识图谱的别名词典来辅助评测。F1值的计算方式如下：

F1 = (2 * P * R) / (P + R)，其中

- P = 测试集所有句子中预测正确的SPO个数 / 测试集所有句子中预测出的SPO个数
- R = 测试集所有句子中预测正确的SPO个数 / 测试集所有句子中人工标注的SPO个数

### Step1：构建模型

该任务可以看作一个序列标注任务，所以基线模型采用的是ERNIE序列标注模型。

**PaddleNLP提供了ERNIE预训练模型常用序列标注模型，可以通过指定模型名字完成一键加载。PaddleNLP为了方便用户处理数据，内置了对于各个预训练模型对应的Tokenizer，可以完成文本token化，转token ID，文本长度截断等操作。**

文本数据处理直接调用tokenizer即可输出模型所需输入数据。




```python
import os
import json
from paddlenlp.transformers import ErnieForTokenClassification, ErnieTokenizer

label_map_path = os.path.join('data', "predicate2id.json")

if not (os.path.exists(label_map_path) and os.path.isfile(label_map_path)):
    sys.exit("{} dose not exists or is not a file.".format(label_map_path))
with open(label_map_path, 'r', encoding='utf8') as fp:
    label_map = json.load(fp)
    
num_classes = (len(label_map.keys()) - 2) * 2 + 2

model = ErnieForTokenClassification.from_pretrained("ernie-1.0", num_classes=(len(label_map) - 2) * 2 + 2)
tokenizer = ErnieTokenizer.from_pretrained("ernie-1.0")

inputs = tokenizer(text="请输入测试样例", max_seq_len=20)

print(label_map)  ##查看下label_map的具体内容 
```

    [2021-06-26 22:01:39,033] [    INFO] - Downloading https://paddlenlp.bj.bcebos.com/models/transformers/ernie/ernie_v1_chn_base.pdparams and saved to /home/aistudio/.paddlenlp/models/ernie-1.0
    [2021-06-26 22:01:39,037] [    INFO] - Downloading ernie_v1_chn_base.pdparams from https://paddlenlp.bj.bcebos.com/models/transformers/ernie/ernie_v1_chn_base.pdparams
    100%|██████████| 392507/392507 [00:06<00:00, 63714.84it/s]
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.weight. classifier.weight is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.bias. classifier.bias is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    [2021-06-26 22:01:52,192] [    INFO] - Downloading vocab.txt from https://paddlenlp.bj.bcebos.com/models/transformers/ernie/vocab.txt
    100%|██████████| 90/90 [00:00<00:00, 3445.89it/s]





    {'O': 0,
     'I': 1,
     '注册资本': 2,
     '作者': 3,
     '所属专辑': 4,
     '歌手': 5,
     '邮政编码': 6,
     '主演': 7,
     '上映时间_@value': 8,
     '上映时间_inArea': 9,
     '饰演_@value': 10,
     '饰演_inWork': 11,
     '国籍': 12,
     '成立日期': 13,
     '毕业院校': 14,
     '作曲': 15,
     '作词': 16,
     '编剧': 17,
     '导演': 18,
     '面积': 19,
     '占地面积': 20,
     '总部地点': 21,
     '制片人': 22,
     '嘉宾': 23,
     '简称': 24,
     '主持人': 25,
     '获奖_@value': 26,
     '获奖_inWork': 27,
     '获奖_onDate': 28,
     '获奖_period': 29,
     '海拔': 30,
     '出品公司': 31,
     '配音_@value': 32,
     '配音_inWork': 33,
     '所在城市': 34,
     '号': 35,
     '主角': 36,
     '创始人': 37,
     '父亲': 38,
     '祖籍': 39,
     '母亲': 40,
     '朝代': 41,
     '董事长': 42,
     '人口数量': 43,
     '妻子': 44,
     '丈夫': 45,
     '票房_@value': 46,
     '票房_inArea': 47,
     '专业代码': 48,
     '气候': 49,
     '修业年限': 50,
     '改编自': 51,
     '官方语言': 52,
     '首都': 53,
     '主题曲': 54,
     '校长': 55,
     '代言人': 56}



### Step2：加载并处理数据


从比赛官网下载数据集，解压存放于data/目录下并重命名为train_data.json, dev_data.json, test_data.json.(注意：train_data.json, dev_data.json, test_data.json是样本数据集，比官网数据小，全数据集是不包含`_data`的数据)

我们可以加载自定义数据集。通过继承[`paddle.io.Dataset`](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/io/Dataset_cn.html#dataset)，自定义实现`__getitem__` 和 `__len__`两个方法。



```python
from typing import Optional, List, Union, Dict

import numpy as np
import paddle
from tqdm import tqdm
from paddlenlp.utils.log import logger

from data_loader import parse_label, DataCollator, convert_example_to_feature
from extract_chinese_and_punct import ChineseAndPunctuationExtractor


class DuIEDataset(paddle.io.Dataset):
    """
    Dataset of DuIE.
    """

    def __init__(
            self,
            input_ids: List[Union[List[int], np.ndarray]],
            seq_lens: List[Union[List[int], np.ndarray]],
            tok_to_orig_start_index: List[Union[List[int], np.ndarray]],
            tok_to_orig_end_index: List[Union[List[int], np.ndarray]],
            labels: List[Union[List[int], np.ndarray, List[str], List[Dict]]]):
        super(DuIEDataset, self).__init__()

        self.input_ids = input_ids
        self.seq_lens = seq_lens
        self.tok_to_orig_start_index = tok_to_orig_start_index
        self.tok_to_orig_end_index = tok_to_orig_end_index
        self.labels = labels

    def __len__(self):
        if isinstance(self.input_ids, np.ndarray):
            return self.input_ids.shape[0]
        else:
            return len(self.input_ids)

    def __getitem__(self, item):
        return {
            "input_ids": np.array(self.input_ids[item]),
            "seq_lens": np.array(self.seq_lens[item]),
            "tok_to_orig_start_index":
            np.array(self.tok_to_orig_start_index[item]),
            "tok_to_orig_end_index": np.array(self.tok_to_orig_end_index[item]),
            # If model inputs is generated in `collate_fn`, delete the data type casting.
            "labels": np.array(
                self.labels[item], dtype=np.float32),
        }

    @classmethod
    def from_file(cls,
                  file_path: Union[str, os.PathLike],
                  tokenizer: ErnieTokenizer,
                  max_length: Optional[int]=512,
                  pad_to_max_length: Optional[bool]=None):
        assert os.path.exists(file_path) and os.path.isfile(
            file_path), f"{file_path} dose not exists or is not a file."
        label_map_path = os.path.join(
            os.path.dirname(file_path), "predicate2id.json")
        assert os.path.exists(label_map_path) and os.path.isfile(
            label_map_path
        ), f"{label_map_path} dose not exists or is not a file."
        with open(label_map_path, 'r', encoding='utf8') as fp:
            label_map = json.load(fp)
        chineseandpunctuationextractor = ChineseAndPunctuationExtractor()

        input_ids, seq_lens, tok_to_orig_start_index, tok_to_orig_end_index, labels = (
            [] for _ in range(5))
        dataset_scale = sum(1 for line in open(file_path, 'r'))
        logger.info("Preprocessing data, loaded from %s" % file_path)
        with open(file_path, "r", encoding="utf-8") as fp:
            lines = fp.readlines()
            for line in tqdm(lines):
                example = json.loads(line)
                input_feature = convert_example_to_feature(
                    example, tokenizer, chineseandpunctuationextractor,
                    label_map, max_length, pad_to_max_length)
                input_ids.append(input_feature.input_ids)
                seq_lens.append(input_feature.seq_len)
                tok_to_orig_start_index.append(
                    input_feature.tok_to_orig_start_index)
                tok_to_orig_end_index.append(
                    input_feature.tok_to_orig_end_index)
                labels.append(input_feature.labels)

        return cls(input_ids, seq_lens, tok_to_orig_start_index,
                   tok_to_orig_end_index, labels)

```


```python
data_path = 'data'
batch_size = 32
max_seq_length = 128

train_file_path = os.path.join(data_path, 'train_data.json')
train_dataset = DuIEDataset.from_file(
    train_file_path, tokenizer, max_seq_length, True)
train_batch_sampler = paddle.io.BatchSampler(
    train_dataset, batch_size=batch_size, shuffle=True, drop_last=True)
collator = DataCollator()
train_data_loader = paddle.io.DataLoader(
    dataset=train_dataset,
    batch_sampler=train_batch_sampler,
    collate_fn=collator)

eval_file_path = os.path.join(data_path, 'dev_data.json')
test_dataset = DuIEDataset.from_file(
    eval_file_path, tokenizer, max_seq_length, True)
test_batch_sampler = paddle.io.BatchSampler(
    test_dataset, batch_size=batch_size, shuffle=False, drop_last=True)
test_data_loader = paddle.io.DataLoader(
    dataset=test_dataset,
    batch_sampler=test_batch_sampler,
    collate_fn=collator)
```

    [2021-06-11 17:41:38,692] [    INFO] - Preprocessing data, loaded from data/train_data.json
    100%|██████████| 10010/10010 [00:18<00:00, 553.08it/s]
    [2021-06-11 17:41:56,818] [    INFO] - Preprocessing data, loaded from data/dev_data.json
    100%|██████████| 1000/1000 [00:01<00:00, 571.08it/s]


### Step3：定义损失函数和优化器，开始训练

我们选择均方误差作为损失函数，使用[`paddle.optimizer.AdamW`](https://www.paddlepaddle.org.cn/documentation/docs/zh/api/paddle/optimizer/adamw/AdamW_cn.html#adamw)作为优化器。



在训练过程中，模型保存在当前目录checkpoints文件夹下。同时在训练的同时使用官方评测脚本进行评估，输出P/R/F1指标。
在验证集上F1可以达到69.42。



```python
import paddle.nn as nn

class BCELossForDuIE(nn.Layer):
    def __init__(self, ):
        super(BCELossForDuIE, self).__init__()
        self.criterion = nn.BCEWithLogitsLoss(reduction='none')

    def forward(self, logits, labels, mask):
        loss = self.criterion(logits, labels)
        mask = paddle.cast(mask, 'float32')
        loss = loss * mask.unsqueeze(-1)
        loss = paddle.sum(loss.mean(axis=2), axis=1) / paddle.sum(mask, axis=1)
        loss = loss.mean()
        return loss
```


```python
from utils import write_prediction_results, get_precision_recall_f1, decoding

@paddle.no_grad()
def evaluate(model, criterion, data_loader, file_path, mode):
    """
    mode eval:
    eval on development set and compute P/R/F1, called between training.
    mode predict:
    eval on development / test set, then write predictions to \
        predict_test.json and predict_test.json.zip \
        under /home/aistudio/relation_extraction/data dir for later submission or evaluation.
    """
    example_all = []
    with open(file_path, "r", encoding="utf-8") as fp:
        for line in fp:
            example_all.append(json.loads(line))
    id2spo_path = os.path.join(os.path.dirname(file_path), "id2spo.json")
    with open(id2spo_path, 'r', encoding='utf8') as fp:
        id2spo = json.load(fp)

    model.eval()
    loss_all = 0
    eval_steps = 0
    formatted_outputs = []
    current_idx = 0
    for batch in tqdm(data_loader, total=len(data_loader)):
        eval_steps += 1
        input_ids, seq_len, tok_to_orig_start_index, tok_to_orig_end_index, labels = batch
        logits = model(input_ids=input_ids)
        mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and((input_ids != 2))
        loss = criterion(logits, labels, mask)
        loss_all += loss.numpy().item()
        probs = F.sigmoid(logits)
        logits_batch = probs.numpy()
        seq_len_batch = seq_len.numpy()
        tok_to_orig_start_index_batch = tok_to_orig_start_index.numpy()
        tok_to_orig_end_index_batch = tok_to_orig_end_index.numpy()
        formatted_outputs.extend(decoding(example_all[current_idx: current_idx+len(logits)],
                                          id2spo,
                                          logits_batch,
                                          seq_len_batch,
                                          tok_to_orig_start_index_batch,
                                          tok_to_orig_end_index_batch))
        current_idx = current_idx+len(logits)
    loss_avg = loss_all / eval_steps
    print("eval loss: %f" % (loss_avg))

    if mode == "predict":
        predict_file_path = os.path.join("/home/aistudio/relation_extraction/data", 'predictions.json')
    else:
        predict_file_path = os.path.join("/home/aistudio/relation_extraction/data", 'predict_eval.json')

    predict_zipfile_path = write_prediction_results(formatted_outputs,
                                                    predict_file_path)

    if mode == "eval":
        precision, recall, f1 = get_precision_recall_f1(file_path,
                                                        predict_zipfile_path)
        os.system('rm {} {}'.format(predict_file_path, predict_zipfile_path))
        return precision, recall, f1
    elif mode != "predict":
        raise Exception("wrong mode for eval func")
```


```python
from paddlenlp.transformers import LinearDecayWithWarmup

learning_rate = 2e-5
num_train_epochs = 5
warmup_ratio = 0.06

criterion = BCELossForDuIE()
# Defines learning rate strategy.
steps_by_epoch = len(train_data_loader)
num_training_steps = steps_by_epoch * num_train_epochs
lr_scheduler = LinearDecayWithWarmup(learning_rate, num_training_steps, warmup_ratio)
optimizer = paddle.optimizer.AdamW(
    learning_rate=lr_scheduler,
    parameters=model.parameters(),
    apply_decay_param_fun=lambda x: x in [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])])
```


```python
# 模型参数保存路径
!mkdir checkpoints
```

    mkdir: cannot create directory ‘checkpoints’: File exists


### Step4：提交预测结果

加载训练保存的模型加载后进行预测。

**NOTE:** 注意设置用于预测的模型参数路径。


```python
import time
import paddle.nn.functional as F

# Starts training.
global_step = 0
logging_steps = 50
save_steps = 10000
num_train_epochs = 2
output_dir = 'checkpoints'
tic_train = time.time()
model.train()
for epoch in range(num_train_epochs):
    print("\n=====start training of %d epochs=====" % epoch)
    tic_epoch = time.time()
    for step, batch in enumerate(train_data_loader):
        input_ids, seq_lens, tok_to_orig_start_index, tok_to_orig_end_index, labels = batch
        logits = model(input_ids=input_ids)
        mask = (input_ids != 0).logical_and((input_ids != 1)).logical_and(
            (input_ids != 2))
        loss = criterion(logits, labels, mask)
        loss.backward()
        optimizer.step()
        lr_scheduler.step()
        optimizer.clear_gradients()
        loss_item = loss.numpy().item()

        if global_step % logging_steps == 0:
            print(
                "epoch: %d / %d, steps: %d / %d, loss: %f, speed: %.2f step/s"
                % (epoch, num_train_epochs, step, steps_by_epoch,
                    loss_item, logging_steps / (time.time() - tic_train)))
            tic_train = time.time()

        if global_step % save_steps == 0 and global_step != 0:
            print("\n=====start evaluating ckpt of %d steps=====" %
                    global_step)
            precision, recall, f1 = evaluate(
                model, criterion, test_data_loader, eval_file_path, "eval")
            print("precision: %.2f\t recall: %.2f\t f1: %.2f\t" %
                    (100 * precision, 100 * recall, 100 * f1))
            print("saving checkpoing model_%d.pdparams to %s " %
                    (global_step, output_dir))
            paddle.save(model.state_dict(),
                        os.path.join(output_dir, 
                                        "model_%d.pdparams" % global_step))
            model.train()

        global_step += 1
    tic_epoch = time.time() - tic_epoch
    print("epoch time footprint: %d hour %d min %d sec" %
            (tic_epoch // 3600, (tic_epoch % 3600) // 60, tic_epoch % 60))

# Does final evaluation.
print("\n=====start evaluating last ckpt of %d steps=====" %
        global_step)
precision, recall, f1 = evaluate(model, criterion, test_data_loader,
                                    eval_file_path, "eval")
print("precision: %.2f\t recall: %.2f\t f1: %.2f\t" %
        (100 * precision, 100 * recall, 100 * f1))
paddle.save(model.state_dict(),
            os.path.join(output_dir,
                            "model_%d.pdparams" % global_step))
print("\n=====training complete=====")
```

    
    =====start training of 0 epochs=====
    epoch: 0 / 2, steps: 0 / 312, loss: 0.724156, speed: 110.16 step/s
    epoch: 0 / 2, steps: 50 / 312, loss: 0.487328, speed: 4.28 step/s
    epoch: 0 / 2, steps: 100 / 312, loss: 0.198309, speed: 4.27 step/s
    epoch: 0 / 2, steps: 150 / 312, loss: 0.128729, speed: 4.30 step/s
    epoch: 0 / 2, steps: 200 / 312, loss: 0.093066, speed: 4.28 step/s
    epoch: 0 / 2, steps: 250 / 312, loss: 0.073819, speed: 4.28 step/s
    epoch: 0 / 2, steps: 300 / 312, loss: 0.060449, speed: 4.27 step/s
    epoch time footprint: 0 hour 1 min 13 sec
    
    =====start training of 1 epochs=====
    epoch: 1 / 2, steps: 38 / 312, loss: 0.049595, speed: 4.27 step/s
    epoch: 1 / 2, steps: 88 / 312, loss: 0.043262, speed: 4.26 step/s
    epoch: 1 / 2, steps: 138 / 312, loss: 0.038916, speed: 4.28 step/s
    epoch: 1 / 2, steps: 188 / 312, loss: 0.035242, speed: 4.29 step/s
    epoch: 1 / 2, steps: 238 / 312, loss: 0.031852, speed: 4.27 step/s
    epoch: 1 / 2, steps: 288 / 312, loss: 0.031410, speed: 4.28 step/s
    epoch time footprint: 0 hour 1 min 12 sec
    
    =====start evaluating last ckpt of 624 steps=====


    100%|██████████| 31/31 [00:02<00:00, 11.27it/s]


    eval loss: 0.027972
    precision: 0.00	 recall: 0.00	 f1: 0.00	
    
    =====training complete=====


## 下面直接使用命令行训练，修改：
- run_duie.py中数据集加载：`train_data.json-->train.json/dev_data.json-->dev.json`
- predict.sh中数据文件路径：`DATASET_FILE=./data/test_data.json-->DATASET_FILE=./data/test.json`


```python
!sh train.sh  #原始代码运行时长: 4小时8分钟23秒843毫秒-->precision: 64.49	 recall: 73.16	 f1: 68.55
```

    + export BATCH_SIZE=32
    + export LR=2e-5
    + export EPOCH=12
    + unset CUDA_VISIBLE_DEVICES
    + python -m paddle.distributed.launch --gpus 0 run_duie.py --device gpu --seed 42 --do_train --data_path ./data --max_seq_length 128 --batch_size 32 --num_train_epochs 12 --learning_rate 2e-5 --warmup_ratio 0.06 --output_dir ./checkpoints
    -----------  Configuration Arguments -----------
    gpus: 0
    heter_worker_num: None
    heter_workers: 
    http_port: None
    ips: 127.0.0.1
    log_dir: log
    nproc_per_node: None
    run_mode: None
    server_num: None
    servers: 
    training_script: run_duie.py
    training_script_args: ['--device', 'gpu', '--seed', '42', '--do_train', '--data_path', './data', '--max_seq_length', '128', '--batch_size', '32', '--num_train_epochs', '12', '--learning_rate', '2e-5', '--warmup_ratio', '0.06', '--output_dir', './checkpoints']
    worker_num: None
    workers: 
    ------------------------------------------------
    WARNING 2021-06-27 20:09:41,666 launch.py:357] Not found distinct arguments and compiled with cuda or xpu. Default use collective mode
    launch train in GPU mode!
    INFO 2021-06-27 20:09:41,668 launch_utils.py:510] Local start 1 processes. First process distributed environment info (Only For Debug): 
        +=======================================================================================+
        |                        Distributed Envs                      Value                    |
        +---------------------------------------------------------------------------------------+
        |                       PADDLE_TRAINER_ID                        0                      |
        |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:60385               |
        |                     PADDLE_TRAINERS_NUM                        1                      |
        |                PADDLE_TRAINER_ENDPOINTS                 127.0.0.1:60385               |
        |                     PADDLE_RANK_IN_NODE                        0                      |
        |                 PADDLE_LOCAL_DEVICE_IDS                        0                      |
        |                 PADDLE_WORLD_DEVICE_IDS                        0                      |
        |                     FLAGS_selected_gpus                        0                      |
        |             FLAGS_selected_accelerators                        0                      |
        +=======================================================================================+
    
    INFO 2021-06-27 20:09:41,668 launch_utils.py:514] details abouts PADDLE_TRAINER_ENDPOINTS can be found in log/endpoints.log, and detail running logs maybe found in log/workerlog.0
    launch proc_id:279 idx:0



```python
%cd ~/relation_extraction/
!bash predict.sh  #直接运行内存会溢出！因为全数据集暂用内存太多，建议本地服务器跑这行代码！
```

    /home/aistudio/relation_extraction
    + export CUDA_VISIBLE_DEVICES=0
    + CUDA_VISIBLE_DEVICES=0
    + export BATCH_SIZE=32
    + BATCH_SIZE=32
    + export CKPT=./checkpoints/model_64224.pdparams
    + CKPT=./checkpoints/model_64224.pdparams
    + export DATASET_FILE=./data/test.json
    + DATASET_FILE=./data/test.json
    + python run_duie.py --do_predict --init_checkpoint ./checkpoints/model_64224.pdparams --predict_data_file ./data/test.json --max_seq_length 512 --batch_size 32
    [32m[2021-07-14 12:02:01,969] [    INFO][0m - Downloading https://paddlenlp.bj.bcebos.com/models/transformers/ernie/ernie_v1_chn_base.pdparams and saved to /home/aistudio/.paddlenlp/models/ernie-1.0[0m
    [32m[2021-07-14 12:02:01,970] [    INFO][0m - Downloading ernie_v1_chn_base.pdparams from https://paddlenlp.bj.bcebos.com/models/transformers/ernie/ernie_v1_chn_base.pdparams[0m
    100%|████████████████████████████████| 392507/392507 [00:06<00:00, 60516.91it/s]
    W0714 12:02:08.557142   871 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0714 12:02:08.562187   871 device_context.cc:422] device: 0, cuDNN Version: 7.6.
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.weight. classifier.weight is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.bias. classifier.bias is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    [32m[2021-07-14 12:02:15,867] [    INFO][0m - Downloading vocab.txt from https://paddlenlp.bj.bcebos.com/models/transformers/ernie/vocab.txt[0m
    100%|█████████████████████████████████████████| 90/90 [00:00<00:00, 2057.26it/s]
    [32m[2021-07-14 12:02:16,102] [    INFO][0m - Preprocessing data, loaded from ./data/test.json[0m
      2%|▋                                   | 1947/101311 [00:08<07:12, 229.48it/s]

预测结果会被保存在data/predictions.json，data/predictions.json.zip，其格式与原数据集文件一致。

之后可以使用官方评估脚本评估训练模型在dev_data.json上的效果。如：

```shell
python re_official_evaluation.py --golden_file=dev_data.json  --predict_file=predicitons.json.zip [--alias_file alias_dict]
```
输出指标为Precision, Recall 和 F1，Alias file包含了合法的实体别名，最终评测的时候会使用，这里不予提供。

之后在test_data.json上预测，然后预测结果（submission.zip文件）至[千言评测页面](https://aistudio.baidu.com/aistudio/competition/detail/46)。



## Tricks

### 尝试更多的预训练模型

基线采用的预训练模型为ERNIE，PaddleNLP提供了丰富的预训练模型，如BERT，RoBERTa，Electra，XLNet等
参考[预训练模型文档](https://paddlenlp.readthedocs.io/zh/latest/model_zoo/transformers.html)

如可以选择RoBERTa large中文模型优化模型效果，只需更换模型和tokenizer即可无缝衔接。


```python
from paddlenlp.transformers import RobertaForTokenClassification, RobertaTokenizer

model = RobertaForTokenClassification.from_pretrained(
    "roberta-wwm-ext-large",
    num_classes=(len(label_map) - 2) * 2 + 2)
tokenizer = RobertaTokenizer.from_pretrained("roberta-wwm-ext-large")
```

    [2021-06-11 17:45:02,058] [    INFO] - Downloading https://paddlenlp.bj.bcebos.com/models/transformers/roberta_large/roberta_chn_large.pdparams and saved to /home/aistudio/.paddlenlp/models/roberta-wwm-ext-large
    [2021-06-11 17:45:02,061] [    INFO] - Downloading roberta_chn_large.pdparams from https://paddlenlp.bj.bcebos.com/models/transformers/roberta_large/roberta_chn_large.pdparams
    100%|██████████| 1271615/1271615 [00:27<00:00, 46861.17it/s]
    [2021-06-11 17:45:34,542] [    INFO] - Downloading vocab.txt from https://paddlenlp.bj.bcebos.com/models/transformers/roberta_large/vocab.txt
    100%|██████████| 107/107 [00:00<00:00, 3004.92it/s]


### 模型集成

使用多个模型进行训练预测，将各个模型预测结果进行融合。

以上基线实现基于PaddleNLP，开源不易，希望大家多多支持~ 
**记得给[PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)点个小小的Star⭐，及时跟踪最新消息和功能哦**

GitHub地址：[https://github.com/PaddlePaddle/PaddleNLP](https://github.com/PaddlePaddle/PaddleNLP)

