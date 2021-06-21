## 比赛

### [千言数据集：情感分析](https://aistudio.baidu.com/aistudio/competition/detail/50)




```python

# 正式开始实验之前首先通过如下命令安装最新版本的 paddlenlp
!pip install --upgrade paddlenlp -i https://pypi.org/simple
```

    Collecting paddlenlp
    [?25l  Downloading https://files.pythonhosted.org/packages/b1/e9/128dfc1371db3fc2fa883d8ef27ab6b21e3876e76750a43f58cf3c24e707/paddlenlp-2.0.2-py3-none-any.whl (426kB)
    [K     |████████████████████████████████| 430kB 31kB/s eta 0:00:012
    [?25hRequirement already satisfied, skipping upgrade: visualdl in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.1.1)
    Requirement already satisfied, skipping upgrade: seqeval in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (1.2.2)
    Requirement already satisfied, skipping upgrade: jieba in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.42.1)
    Requirement already satisfied, skipping upgrade: h5py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (2.9.0)
    Requirement already satisfied, skipping upgrade: multiprocess in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.70.11.1)
    Requirement already satisfied, skipping upgrade: colorlog in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (4.1.0)
    Requirement already satisfied, skipping upgrade: colorama in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from paddlenlp) (0.4.4)
    Requirement already satisfied, skipping upgrade: bce-python-sdk in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (0.8.53)
    Requirement already satisfied, skipping upgrade: requests in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (2.22.0)
    Requirement already satisfied, skipping upgrade: pre-commit in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.21.0)
    Requirement already satisfied, skipping upgrade: flask>=1.1.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.1.1)
    Requirement already satisfied, skipping upgrade: flake8>=3.7.9 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (3.8.2)
    Requirement already satisfied, skipping upgrade: Pillow>=7.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (7.1.2)
    Requirement already satisfied, skipping upgrade: numpy in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.20.3)
    Requirement already satisfied, skipping upgrade: Flask-Babel>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.0.0)
    Requirement already satisfied, skipping upgrade: protobuf>=3.11.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (3.14.0)
    Requirement already satisfied, skipping upgrade: six>=1.14.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (1.15.0)
    Requirement already satisfied, skipping upgrade: shellcheck-py in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from visualdl->paddlenlp) (0.7.1.1)
    Requirement already satisfied, skipping upgrade: scikit-learn>=0.21.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from seqeval->paddlenlp) (0.24.2)
    Requirement already satisfied, skipping upgrade: dill>=0.3.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from multiprocess->paddlenlp) (0.3.3)
    Requirement already satisfied, skipping upgrade: pycryptodome>=3.8.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp) (3.9.9)
    Requirement already satisfied, skipping upgrade: future>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from bce-python-sdk->visualdl->paddlenlp) (0.18.0)
    Requirement already satisfied, skipping upgrade: idna<2.9,>=2.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (2.8)
    Requirement already satisfied, skipping upgrade: urllib3!=1.25.0,!=1.25.1,<1.26,>=1.21.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (1.25.6)
    Requirement already satisfied, skipping upgrade: chardet<3.1.0,>=3.0.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (3.0.4)
    Requirement already satisfied, skipping upgrade: certifi>=2017.4.17 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from requests->visualdl->paddlenlp) (2019.9.11)
    Requirement already satisfied, skipping upgrade: virtualenv>=15.2 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (16.7.9)
    Requirement already satisfied, skipping upgrade: toml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (0.10.0)
    Requirement already satisfied, skipping upgrade: importlib-metadata; python_version < "3.8" in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (0.23)
    Requirement already satisfied, skipping upgrade: identify>=1.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.4.10)
    Requirement already satisfied, skipping upgrade: cfgv>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (2.0.1)
    Requirement already satisfied, skipping upgrade: pyyaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (5.1.2)
    Requirement already satisfied, skipping upgrade: aspy.yaml in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.3.0)
    Requirement already satisfied, skipping upgrade: nodeenv>=0.11.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from pre-commit->visualdl->paddlenlp) (1.3.4)
    Requirement already satisfied, skipping upgrade: itsdangerous>=0.24 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (1.1.0)
    Requirement already satisfied, skipping upgrade: click>=5.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (7.0)
    Requirement already satisfied, skipping upgrade: Werkzeug>=0.15 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (0.16.0)
    Requirement already satisfied, skipping upgrade: Jinja2>=2.10.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flask>=1.1.1->visualdl->paddlenlp) (2.10.1)
    Requirement already satisfied, skipping upgrade: pycodestyle<2.7.0,>=2.6.0a1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (2.6.0)
    Requirement already satisfied, skipping upgrade: mccabe<0.7.0,>=0.6.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (0.6.1)
    Requirement already satisfied, skipping upgrade: pyflakes<2.3.0,>=2.2.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from flake8>=3.7.9->visualdl->paddlenlp) (2.2.0)
    Requirement already satisfied, skipping upgrade: pytz in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp) (2019.3)
    Requirement already satisfied, skipping upgrade: Babel>=2.3 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Flask-Babel>=1.0.0->visualdl->paddlenlp) (2.8.0)
    Requirement already satisfied, skipping upgrade: scipy>=0.19.1 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (1.6.3)
    Requirement already satisfied, skipping upgrade: threadpoolctl>=2.0.0 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (2.1.0)
    Requirement already satisfied, skipping upgrade: joblib>=0.11 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from scikit-learn>=0.21.3->seqeval->paddlenlp) (0.14.1)
    Requirement already satisfied, skipping upgrade: zipp>=0.5 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from importlib-metadata; python_version < "3.8"->pre-commit->visualdl->paddlenlp) (0.6.0)
    Requirement already satisfied, skipping upgrade: MarkupSafe>=0.23 in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from Jinja2>=2.10.1->flask>=1.1.1->visualdl->paddlenlp) (1.1.1)
    Requirement already satisfied, skipping upgrade: more-itertools in /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages (from zipp>=0.5->importlib-metadata; python_version < "3.8"->pre-commit->visualdl->paddlenlp) (7.2.0)
    Installing collected packages: paddlenlp
      Found existing installation: paddlenlp 2.0.1
        Uninstalling paddlenlp-2.0.1:
          Successfully uninstalled paddlenlp-2.0.1
    Successfully installed paddlenlp-2.0.2


### 1 加载数据集


```python
user_dir = '/home/aistudio/'
traindataset = 'COTE-MFW'   #选择进行训练的数据集['COTE-BD', 'COTE-MFW', 'COTE-DP']
```


```python
import os
import random
# 读取tsv
def read_tsv(tsv):
    datas = []
    headline = None
    with open(tsv, 'r', encoding='UTF-8') as f:
        for i, line in enumerate(f):
            if i==0:
                print(line)
                headline = line
                continue
            else:
                data = line.strip('\n\t').split('\t')
                datas.append('\t'.join(data)+'\n')
    return datas,headline

# 写入tsv
def write_tsv(tsv, datas):
    with open(tsv, 'w', encoding='UTF-8') as f:
        for line in datas:
            f.write(line)

# 切分并转换
def split_dataset_change(tsv, split_pro):
    random.seed(10)
    datas,headline = read_tsv(tsv)
    random.shuffle(datas)
    # print(datas[:5])
    split_num = int(split_pro*len(datas))
    trainlist = datas[:split_num]
    trainlist.insert(0,headline)
    write_tsv(tsv, trainlist)  #save train.tsv
    devlist = datas[split_num:]
    devlist.insert(0,headline)
    write_tsv('/'.join(tsv.split('/')[:-1])+'/dev.tsv', devlist)  #save dev.tsv
```

#### 1.1 COTE_BD 数据集


```python
!unzip ./data/data53469/COTE-BD.zip -d ~/data/
!rm -r ./data/__MACOSX/
```

    Archive:  ./data/data53469/COTE-BD.zip
      inflating: /home/aistudio/data/COTE-BD/License.pdf  
      inflating: /home/aistudio/data/COTE-BD/test.tsv  
      inflating: /home/aistudio/data/COTE-BD/train.tsv  
    rm: cannot remove './data/__MACOSX/': No such file or directory



```python
split_dataset_change(os.path.join(user_dir,'data',traindataset,'train.tsv'), 0.8)  #该数据集只有train和test，需要手动将train拆分为train和dev
```

    label	text_a
    



```python
from paddlenlp.datasets import DatasetBuilder
import os
class COTE_BD(DatasetBuilder):
    SPLITS = {
        'train': [os.path.join(user_dir,'data',traindataset,'train.tsv'),(0, 1), 1],
        'dev': [os.path.join(user_dir,'data',traindataset,'dev.tsv'),(0, 1), 1],
        'test': [os.path.join(user_dir,'data',traindataset,'test.tsv'), (1, ), 1],

    }

    def _get_data(self, mode, **kwargs):
        filename,_,_ = self.SPLITS[mode]
        return filename

    def _read(self, filename, split):
        _,  field_indices, num_discard_samples = self.SPLITS[split]
        with open(filename, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx < num_discard_samples:
                    continue
                line_stripped = line.strip().split('\t')
                if not line_stripped:
                    continue
                example = [line_stripped[indice] for indice in field_indices]
                if split == "test":
                    yield {"tokens": list(example[0])}
                else:
                    try:
                        entity, text = example[0], example[1]
                        start_idx = text.index(entity)
                    except:
                        # drop the dirty data
                        continue

                    labels = ['O'] * len(text)
                    labels[start_idx] = "B"
                    for idx in range(start_idx + 1, start_idx + len(entity)):
                        labels[idx] = "I"
                    yield {
                        "tokens": list(text),
                        "labels": labels,
                        "entity": entity
                    }

    def get_labels(self):
        return ["B", "I", "O"]
```


```python
if traindataset == 'COTE-BD':    
    print('construct COTE-BD')
    def load_dataset(name=None,
                    data_files=None,
                    splits=None,
                    lazy=None,
                    **kwargs):
    
        reader_cls = COTE_BD
        print(reader_cls)
        if not name:
            reader_instance = reader_cls(lazy=lazy, **kwargs)
        else:
            reader_instance = reader_cls(lazy=lazy, name=name, **kwargs)

        datasets = reader_instance.read_datasets(data_files=data_files, splits=splits)
        return datasets

    # 一键加载 bq_corpus 的训练集、验证集
    train_ds, dev_ds, test_ds = load_dataset(traindataset, splits=["train", "dev", "test"])
    # 输出测试样本
    print('-'*5+'train sample data'+'-'*5)
    print(train_ds[0])
    print(train_ds[1])
    print(train_ds[2])
    print('-'*5+'dev sample data'+'-'*5)
    print(dev_ds[0])
    print(dev_ds[1])
    print(dev_ds[2])
    print('-'*5+'test sample data'+'-'*5)
    print(test_ds[0])
    print(test_ds[1])
    print(test_ds[2])
```

    construct COTE-BD
    <class '__main__.COTE_BD'>
    -----train sample data-----
    {'tokens': ['王', '浩', '，', '1', '9', '6', '1', '年', '0', '2', '月', '出', '生', '，', '南', '京', '理', '工', '大', '学', '动', '力', '学', '院', '研', '究', '员', '，', '博', '士', '，', '现', '任', '南', '京', '理', '工', '大', '学', '动', '力', '学', '院', '院', '长', '。'], 'labels': [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '王浩'}
    {'tokens': ['龙', '门', '飞', '甲', '也', '是', '今', '年', '非', '常', '不', '错', '的', '电', '影', '之', '一', '，', '优', '点', '如', '下', '，', '1', '看', '完', '整', '部', '电', '影', '不', '会', '头', '晕', '，', '2', '打', '斗', '场', '面', '过', '瘾', '3', 'D', '效', '果', '非', '常', '好', '、', '化', '妆', '服', '装', '场', '景', '极', '好', '，', '3', '，', '情', '节', '有', '趣', '紧', '凑', '。'], 'labels': [0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '龙门飞甲'}
    {'tokens': ['《', '金', '刚', '狼', '2', '》', '是', '部', '失', '望', '之', '作', '。'], 'labels': [2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '金刚狼2'}
    -----dev sample data-----
    {'tokens': ['大', '海', '是', '让', '人', '心', '胸', '开', '阔', '的', '地', '方', '，', '看', '海', '其', '实', '最', '佳', '的', '时', '间', '是', '下', '午', '五', '点', '以', '后', '，', '游', '客', '很', '少', '，', '亚', '龙', '湾', '海', '滩', '边', '有', '好', '多', '吊', '床', '和', '摇', '椅', '.', '夕', '阳', '西', '下', '，', '波', '光', '粼', '粼', '，', '景', '能', '动', '人', '，', '人', '亦', '动', '心', '。'], 'labels': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '亚龙湾'}
    {'tokens': ['白', '天', '还', '可', '以', '在', '那', '花', '2', '块', '钱', '坐', '船', '到', '对', '面', '的', '麻', '斜', '，', '从', '另', '一', '角', '度', '欣', '赏', '观', '海', '长', '廊', '，', '湛', '江', '建', '筑', '。'], 'labels': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2], 'entity': '观海长廊'}
    {'tokens': ['爱', '情', '，', '陈', '慧', '恬', '演', '唱', '歌', '手', '。'], 'labels': [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '爱情'}
    -----test sample data-----
    {'tokens': ['毕', '棚', '沟', '的', '风', '景', '早', '有', '所', '闻', '，', '尤', '其', '以', '秋', '季', '的', '风', '景', '最', '美', '，', '但', '是', '这', '次', '去', '晚', '了', '，', '红', '叶', '全', '掉', '完', '了', '，', '黄', '叶', '也', '看', '不', '到', '了', '，', '下', '了', '雪', '只', '能', '看', '看', '雪', '山', '了', '，', '还', '好', '雪', '山', '的', '雄', '伟', '确', '实', '值', '得', '一', '看', '。']}
    {'tokens': ['虽', '然', '剧', '情', '老', '套', ' ', '但', '韩', '孝', '珠', '那', '天', '真', '无', '邪', '的', '笑', ' ', '苏', '志', '燮', '苦', '逼', '的', '表', '情', '一', '摆', ' ', '流', '着', '泪', '着', '背', '对', '女', '主', '走', '远', ' ', '这', '画', '面', '完', '全', '可', '以', '忽', '略', '剧', '情', '啊', ' ', '又', '把', '我', '虐', '的', '哭', '成', '狗', '啊', ' ', '看', '过', '创', '可', '贴', ' ', '和', '只', '有', '你', '之', '后', '太', '喜', '欢', '韩', '孝', '珠', '了', '😂', '😂', '😂', '还', '有', '就', '是', '整', '部', '电', '影', '配', '乐', '好', '棒', '啊', ' ', '整', '体', '太', '美', '。']}
    {'tokens': ['每', '次', '有', '朋', '友', '来', ' ', '深', '圳', ' ', '基', '本', '都', '要', '去', ' ', '世', '界', '之', '窗', ' ', '，', '里', '面', '就', '是', '世', '界', '各', '国', '的', '特', '色', '景', '点', '的', '缩', '小', '版', '，', '个', '人', '感', '觉', '没', '有', '太', '大', '意', '思', '，', '但', '是', '世', '界', '之', '窗', '美', '逢', '节', '假', '日', '都', '会', '有', '很', '多', '活', '动', '和', '节', '目', '比', '较', '有', '意', '思', '～', '就', '单', '单', '进', '去', '逛', '逛', '1', '8', '0', '的', '门', '票', '感', '觉', '有', '点', '儿', '小', '贵', '。']}


#### 1.2 COTE-MFW 数据集


```python
!unzip ./data/data53469/COTE-MFW.zip -d ~/data/
!rm -r ./data/__MACOSX/
```

    Archive:  ./data/data53469/COTE-MFW.zip
      inflating: /home/aistudio/data/COTE-MFW/License.pdf  
      inflating: /home/aistudio/data/COTE-MFW/test.tsv  
      inflating: /home/aistudio/data/COTE-MFW/train.tsv  
    rm: cannot remove './data/__MACOSX/': No such file or directory



```python
split_dataset_change(os.path.join(user_dir,'data',traindataset,'train.tsv'), 0.8)  #该数据集只有train和test，需要手动将train拆分为train和dev`
```

    label	text_a
    



```python
from paddlenlp.datasets import DatasetBuilder
import os
class COTE_MFW(DatasetBuilder):
    SPLITS = {
        'train': [os.path.join(user_dir,'data',traindataset,'train.tsv'),(0, 1), 1],
        'dev': [os.path.join(user_dir,'data',traindataset,'dev.tsv'),(0, 1), 1],
        'test': [os.path.join(user_dir,'data',traindataset,'test.tsv'), (1, ), 1],

    }

    def _get_data(self, mode, **kwargs):
        filename,_,_ = self.SPLITS[mode]
        return filename

    def _read(self, filename, split):
        _,  field_indices, num_discard_samples = self.SPLITS[split]
        with open(filename, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx < num_discard_samples:
                    continue
                line_stripped = line.strip().split('\t')
                if not line_stripped:
                    continue
                example = [line_stripped[indice] for indice in field_indices]
                if split == "test":
                    yield {"tokens": list(example[0])}
                else:
                    try:
                        entity, text = example[0], example[1]
                        start_idx = text.index(entity)
                    except:
                        # drop the dirty data
                        continue

                    labels = ['O'] * len(text)
                    labels[start_idx] = "B"
                    for idx in range(start_idx + 1, start_idx + len(entity)):
                        labels[idx] = "I"
                    yield {
                        "tokens": list(text),
                        "labels": labels,
                        "entity": entity
                    }

    def get_labels(self):
        return ["B", "I", "O"]
```


```python
if traindataset == 'COTE-MFW':    
    print('construct COTE-MFW')
    def load_dataset(name=None,
                    data_files=None,
                    splits=None,
                    lazy=None,
                    **kwargs):
    
        reader_cls = COTE_MFW
        print(reader_cls)
        if not name:
            reader_instance = reader_cls(lazy=lazy, **kwargs)
        else:
            reader_instance = reader_cls(lazy=lazy, name=name, **kwargs)

        datasets = reader_instance.read_datasets(data_files=data_files, splits=splits)
        return datasets

    # 一键加载 bq_corpus 的训练集、验证集
    train_ds, dev_ds, test_ds = load_dataset(traindataset, splits=["train", "dev", "test"])
    # 输出测试样本
    print('-'*5+'train sample data'+'-'*5)
    print(train_ds[0])
    print(train_ds[1])
    print(train_ds[2])
    print('-'*5+'dev sample data'+'-'*5)
    print(dev_ds[0])
    print(dev_ds[1])
    print(dev_ds[2])
    print('-'*5+'test sample data'+'-'*5)
    print(test_ds[0])
    print(test_ds[1])
    print(test_ds[2])
```

    construct COTE-MFW
    <class '__main__.COTE_MFW'>
    -----train sample data-----
    {'tokens': ['鄯', '善', '县', '城', '确', '实', '是', '一', '座', '小', '县', '城', '，', '除', '了', '一', '条', '主', '要', '大', '街', '其', '余', '的', '小', '街', '道', '都', '很', '陈', '旧', '，', '离', '库', '姆', '塔', '格', '沙', '漠', '近', '，', '沙', '漠', '边', '缘', '的', '绿', '洲'], 'labels': [0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '鄯善县城'}
    {'tokens': ['修', '真', '观', '在', '乌', '镇', '东', '栅', '西', '头', '的', '印', '家', '巷', '里', '(', '现', '名', '观', '前', '街', ')', '。'], 'labels': [0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '修真观'}
    {'tokens': [' ', '不', '过', '需', '要', '注', '意', '的', '就', '是', '，', '如', '果', '你', '想', '去', '香', '山', '寺', '的', '话', '，', '就', '不', '要', '坐', '电', '瓶', '车', '了', '，', '因', '为', '香', '山', '寺', '在', '东', '山', '石', '窟', '和', '白', '园', '中', '间', '的', '位', '置', '，', '你', '坐', '电', '瓶', '车', '就', '会', '直', '接', '错', '过', '了', '。'], 'labels': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '白园'}
    -----dev sample data-----
    {'tokens': ['（', '门', '票', '8', '0', '元', '，', '只', '游', '玩', '古', '镇', '不', '进', '景', '点', '是', '不', '用', '买', '票', '的', '）', '我', '是', '吃', '“', '毛', '毛', '鱼', '”', '而', '知', '道', '的', '靖', '港', '古', '镇', '，', '今', '天', '终', '得', '一', '游', '。'], 'labels': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '靖港古镇'}
    {'tokens': ['梅', '庵', '挺', '小', '，', '好', '像', '也', '没', '什', '么', '游', '客', '参', '观', '。', '不', '过', '个', '人', '挺', '喜', '欢'], 'labels': [0, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '梅庵'}
    {'tokens': ['明', '远', '楼', '是', '古', '代', '科', '举', '考', '试', '监', '考', '的', '地', '方', '，', '站', '得', '高', '看', '得', '远', '，', '明', '远', '楼', '的', '下', '面', '就', '是', '一', '档', '一', '档', '的', '考', '室', '。'], 'labels': [0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '明远楼'}
    -----test sample data-----
    {'tokens': ['神', '女', '溪', '据', '说', '在', '山', '峡', '蓄', '水', '前', '就', '是', '条', '很', '清', '澈', '的', '小', '溪', '，', '蓄', '水', '后', '很', '多', '遗', '迹', '都', '淹', '没', '在', '水', '底', '了', '，', '这', '里', '的', '水', '确', '实', '和', '外', '面', '黄', '黄', '的', '水', '不', '一', '样', '。']}
    {'tokens': ['哈', '尔', '盖', '是', '从', '刚', '察', '到', '西', '海', '镇', '之', '间', '的', '一', '个', '小', '镇', '，', '也', '是', '环', '湖', '的', '一', '个', '必', '经', '之', '处', '。']}
    {'tokens': ['1', '9', '4', '9', '年', '全', '国', '解', '放', '后', '，', '对', '大', '桥', '进', '行', '了', '改', '建', '，', '并', '更', '名', '为', '八', '一', '大', '桥', '。']}


#### 1.3 COTE-DP 数据集


```python
!unzip ./data/data53469/COTE-DP.zip -d ~/data/
!rm -r ./data/__MACOSX/
```

    Archive:  ./data/data53469/COTE-DP.zip
      inflating: /home/aistudio/data/COTE-DP/License.pdf  
      inflating: /home/aistudio/data/COTE-DP/test.tsv  
      inflating: /home/aistudio/data/COTE-DP/train.tsv  
    rm: cannot remove './data/__MACOSX/': No such file or directory



```python
split_dataset_change(os.path.join(user_dir,'data',traindataset,'train.tsv'), 0.8)  #该数据集只有train和test，需要手动将train拆分为train和dev
```

    label	text_a
    



```python
from paddlenlp.datasets import DatasetBuilder
import os
class COTE_DP(DatasetBuilder):
    SPLITS = {
        'train': [os.path.join(user_dir,'data',traindataset,'train.tsv'),(0, 1), 1],
        'dev': [os.path.join(user_dir,'data',traindataset,'dev.tsv'),(0, 1), 1],
        'test': [os.path.join(user_dir,'data',traindataset,'test.tsv'), (1, ), 1],

    }

    def _get_data(self, mode, **kwargs):
        filename,_,_ = self.SPLITS[mode]
        return filename

    def _read(self, filename, split):
        _,  field_indices, num_discard_samples = self.SPLITS[split]
        with open(filename, 'r', encoding='utf-8') as f:
            for idx, line in enumerate(f):
                if idx < num_discard_samples:
                    continue
                line_stripped = line.strip().split('\t')
                if not line_stripped:
                    continue
                example = [line_stripped[indice] for indice in field_indices]
                if split == "test":
                    yield {"tokens": list(example[0])}
                else:
                    try:
                        entity, text = example[0], example[1]
                        start_idx = text.index(entity)
                    except:
                        # drop the dirty data
                        continue

                    labels = ['O'] * len(text)
                    labels[start_idx] = "B"
                    for idx in range(start_idx + 1, start_idx + len(entity)):
                        labels[idx] = "I"
                    yield {
                        "tokens": list(text),
                        "labels": labels,
                        "entity": entity
                    }

    def get_labels(self):
        return ["B", "I", "O"]
```


```python
if traindataset == 'COTE-DP':    
    print('construct COTE-DP')
    def load_dataset(name=None,
                    data_files=None,
                    splits=None,
                    lazy=None,
                    **kwargs):
    
        reader_cls = COTE_DP
        print(reader_cls)
        if not name:
            reader_instance = reader_cls(lazy=lazy, **kwargs)
        else:
            reader_instance = reader_cls(lazy=lazy, name=name, **kwargs)

        datasets = reader_instance.read_datasets(data_files=data_files, splits=splits)
        return datasets

    # 一键加载 bq_corpus 的训练集、验证集
    train_ds, dev_ds, test_ds = load_dataset(traindataset, splits=["train", "dev", "test"])
    # 输出测试样本
    print('-'*5+'train sample data'+'-'*5)
    print(train_ds[0])
    print(train_ds[1])
    print(train_ds[2])
    print('-'*5+'dev sample data'+'-'*5)
    print(dev_ds[0])
    print(dev_ds[1])
    print(dev_ds[2])
    print('-'*5+'test sample data'+'-'*5)
    print(test_ds[0])
    print(test_ds[1])
    print(test_ds[2])
```

    construct COTE-DP
    <class '__main__.COTE_DP'>
    -----train sample data-----
    {'tokens': ['生', '态', '冷', '盘', '盐', '水', '鸭', '半', '只', '：', '不', '愧', '是', '狮', '王', '府', '的', '传', '统', '特', '色', '菜', '，', '鸭', '肉', '咸', '鲜', '，', '肉', '质', '很', '嫩', '，', '没', '有', '异', '味', '，', '我', '吃', '了', '好', '几', '块', '。'], 'labels': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '狮王府'}
    {'tokens': ['难', '道', '这', '就', '是', '传', '说', '中', '的', '人', '间', '自', '有', '真', '情', '在', '？', '°', '°', '・', '(', '＞', '_', '＜', ')', '・', '°', '°', '给', '小', '万', '食', '堂', '点', '一', '万', '个', '赞', '！', '！', '感', '谢', '！', '！'], 'labels': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '小万食堂'}
    {'tokens': ['在', '进', '站', '以', '后', '的', '左', '手', '边', '的', '2', '楼', '和', '麦', '当', '劳', '算', '是', '遥', '遥', '相', '望', '的', '在', '火', '车', '站', '吃', '肯', '德', '基', '，', '还', '是', '套', '餐', '相', '对', '比', '价', '划', '算', '的', '因', '为', '不', '能', '使', '用', '优', '惠', '券', '的', '，', '所', '以', '还', '是', '比', '较', '贵', '的', '买', '了', '一', '个', '圣', '代', '，', '6', '.', '5', '，', '如', '果', '是', '套', '餐', '的', '话', '基', '本', '上', '一', '个', '人', '3', '9', '的', '这', '样', '的', '价', '格', '大', '概', '是', '一', '个', '汉', '堡', '+', '薯', '条', '+', '饮', '料', '后', '面', '有', '个', '老', '外', '想', '点', '炸', '鸡', '累', '的', '东', '西', '都', '是', '说', '没', '有', '，', '也', '许', '是', '真', '的', '还', '没', '做', '好', '吧', '！'], 'labels': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '肯德基'}
    -----dev sample data-----
    {'tokens': ['逛', '街', '随', '处', '可', '见', '雅', '克', '雅', '思', '，', '高', '中', '时', '对', '面', '就', '是', '一', '家', '，', '上', '学', '期', '间', '每', '个', '星', '期', '都', '在', '那', '里', '拉', '动', '经', '济', '发', '展', '。'], 'labels': [2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '雅克雅思'}
    {'tokens': ['『', '位', '置', '』', '一', '直', '听', '说', '这', '家', '焖', '子', '在', '所', '城', '里', '，', '但', '是', '具', '体', '在', '哪', '不', '知', '道', '，', '因', '为', '这', '次', '来', '吃', '小', '腰', '子', '，', '一', '坐', '下', '就', '看', '见', '一', '大', '姨', '在', '卖', '焖', '子', '，', '凭', '直', '觉', '点', '开', '大', '众', '点', '评', '，', '搜', '索', '所', '城', '里', '大', '姨', '焖', '子', '，', '看', '到', '了', '图', '片', '上', '一', '模', '一', '样', '的', '高', '师', '傅', '开', '锁', '4', '9', '9', '9', '9', '9', '9', '.', '.', '.', '就', '是', '她', '了', '。'], 'labels': [2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 0, 1, 1, 1, 1, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '所城里大姨焖子'}
    {'tokens': ['好', '利', '来', '真', '是', '充', '满', '了', '童', '年', '的', '回', '忆', '啊', '这', '家', '店', '并', '不', '大', '，', '蛋', '糕', '面', '包', '的', '种', '类', '还', '行', '，', '最', '丰', '富', '的', '就', '是', '新', '百', '附', '近', '那', '家', '，', '听', '说', '是', '总', '店', '很', '喜', '欢', '好', '利', '来', '的', '切', '块', '蛋', '糕', '，', '提', '拉', '米', '苏', '杯', '味', '道', '还', '可', '以', '，', '香', '甜', '不', '腻', '，', '上', '面', '一', '层', '微', '苦', '的', '可', '可', '粉', '很', '搭', '下', '面', '甜', '味', '的', '奶', '油', '慕', '斯', '酸', '奶', '挺', '好', '喝', '的', '，', '但', '是', '感', '觉', '不', '是', '自', '制', '的', '那', '种', '，', '还', '是', '希', '望', '推', '出', '自', '己', '现', '做', '的', '没', '有', '防', '腐', '剂', '的', '酸', '奶', '肉', '松', '面', '包', '很', '好', '吃', '，', '对', '肉', '松', '的', '东', '西', '毫', '无', '抵', '抗', '力', '，', '哈', '哈', '环', '境', '还', '行', '，', '看', '着', '挺', '干', '净', '，', '服', '务', '态', '度', '也', '很', '好', '，', '还', '会', '再', '去', '的', '^', '_', '^'], 'labels': [0, 1, 1, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2, 2], 'entity': '好利来'}
    -----test sample data-----
    {'tokens': ['还', '是', '第', '一', '次', '进', '星', '巴', '克', '店', '里', '吃', '东', '西', '\xa0', '那', '会', '儿', '第', '一', '次', '喝', '咖', '啡', '还', '是', '外', '带', '的']}
    {'tokens': ['阿', '春', '粤', '菜', '馆', '普', '君', '新', '城', '店', '在', '普', '君', '新', '城', '的', '二', '楼', '，', '从', '进', '入', '地', '下', '停', '车', '场', '开', '始', '就', '一', '直', '有', '明', '确', '的', '指', '示', '牌', '指', '引', '停', '车', '方', '向', '，', '汽', '车', '直', '达', '负', '二', '层', 'E', '区', '停', '车', '后', '再', '搭', '乘', '手', '扶', '电', '梯', '沿', '路', '跟', '着', '指', '示', '牌', '步', '行', '就', '可', '以', '找', '到', '阿', '春', '粤', '菜', '馆', '了', '。']}
    {'tokens': ['去', '三', '亚', '的', '时', '候', '去', '吃', '了', '大', '东', '海', '的', '拾', '味', '馆', '.', '得', '到', '了', '全', '家', '的', '一', '致', '好', '评', '.', '没', '想', '到', '学', '校', '附', '近', '也', '有', '一', '家', '.', '果', '断', '和', '室', '友', '约', '着', '看', '电', '影', '的', '时', '候', '我', '去', '吃', '.', '由', '于', '对', '椰', '香', '骨', '汤', '印', '象', '很', '深', '刻', '.', '浓', '浓', '的', '骨', '汤', '头', '里', '还', '有', '着', '椰', '子', '的', '清', '香', '味', '，', '喝', '完', '口', '也', '不', '会', '有', '很', '干', '的', '感', '觉', '，', '推', '荐', '.', '凉', '粉', '中', '规', '中', '矩', '，', '有', '点', '偏', '咸', '，', '总', '体', '还', '是', '不', '错', '的', '.', '香', '糯', '的', '椰', '子', '饭', '值', '得', '一', '试', '.', '在', '三', '亚', '时', '海', '南', '四', '大', '名', '菜', '就', '东', '山', '羊', '没', '能', '吃', '到', '，', '在', '这', '里', '终', '于', '凑', '齐', '了', '，', '东', '山', '羊', '刚', '入', '口', '时', '完', '全', '吃', '不', '出', '有', '羊', '的', '膻', '味', '，', '搭', '配', '蘸', '酱', '吃', '更', '好', '吃', '了', '，', '不', '过', '吃', '到', '后', '来', '膻', '味', '就', '出', '来', '了', '.', '整', '体', '来', '说', '还', '是', '不', '错', '的', '.', '不', '过', '觉', '得', '没', '三', '亚', '的', '那', '家', '氛', '围', '好', '.']}


### 2 开始进行训练数据预处理


```python
from functools import partial
import argparse
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.metrics import ChunkEvaluator
from paddlenlp.transformers import SkepCrfForTokenClassification, SkepModel, SkepTokenizer
```


```python
def convert_example_to_feature(example,
                               tokenizer,
                               max_seq_len=512,
                               no_entity_label="O",
                               is_test=False):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed 
    to be used in a sequence-pair classification task.
        
    A skep_ernie_1.0_large_ch/skep_ernie_2.0_large_en sequence has the following format:
    ::
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

    A skep_ernie_1.0_large_ch/skep_ernie_2.0_large_en sequence pair mask has the following format:
    ::

        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |

    If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization.
            Sequences longer than this will be truncated, sequences shorter will be padded.
        no_entity_label(obj:`str`, defaults to "O"): The label represents that the token isn't an entity. 
        is_test(obj:`False`, defaults to `False`): Whether the example contains label or not.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask.
        label(obj:`list[int]`, optional): The input label if not test data.
    """
    tokens = example['tokens']
    labels = example['labels']
    tokenized_input = tokenizer(
        tokens,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_len)

    input_ids = tokenized_input['input_ids']
    token_type_ids = tokenized_input['token_type_ids']
    seq_len = tokenized_input['seq_len']

    if is_test:
        return input_ids, token_type_ids, seq_len
    else:
        labels = labels[:(max_seq_len - 2)]
        encoded_label = np.array(
            [no_entity_label] + labels + [no_entity_label], dtype="int64")

        return input_ids, token_type_ids, seq_len, encoded_label

```


```python
def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)
```


```python
# import paddlenlp
# dir(paddlenlp.transformers)
```


```python
from paddlenlp.transformers import SkepForSequenceClassification, SkepTokenizer

MODEL_NAME = "skep_ernie_1.0_large_ch"

# 指定模型名称，一键加载模型
skep = SkepModel.from_pretrained(MODEL_NAME)
model = SkepCrfForTokenClassification(skep, num_classes=len(train_ds.label_list))
tokenizer = SkepTokenizer.from_pretrained(MODEL_NAME)
```

    [2021-06-17 22:14:39,652] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch/skep_ernie_1.0_large_ch.pdparams
    [2021-06-17 22:14:43,770] [    INFO] - Found /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch/skep_ernie_1.0_large_ch.vocab.txt



```python

# 批量数据大小
batch_size = 32
# 文本序列最大长度
max_seq_length = 256

label_map = {label: idx for idx, label in enumerate(train_ds.label_list)}
no_entity_label_idx = label_map.get("O", 2)

trans_func = partial(
        convert_example_to_feature,
        tokenizer=tokenizer,
        max_seq_len=max_seq_length,
        no_entity_label=no_entity_label_idx,
        is_test=False)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input ids
    Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # token type ids
    Stack(dtype='int64'),  # sequence lens
    Pad(axis=0, pad_val=no_entity_label_idx)  # labels
): [data for data in fn(samples)]

train_data_loader = create_dataloader(
    train_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)

dev_data_loader = create_dataloader(
    dev_ds,
    mode='train',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)

label_map  #产看下labelmap
```




    {'B': 0, 'I': 1, 'O': 2}




```python
epochs = 10
learning_rate=2e-5

num_training_steps = len(train_data_loader) * epochs
# Generate parameter names needed to perform weight decay.
# All bias and LayerNorm parameters are excluded.
decay_params = [
    p.name for n, p in model.named_parameters()
    if not any(nd in n for nd in ["bias", "norm"])
]
optimizer = paddle.optimizer.AdamW(
    learning_rate=learning_rate,
    parameters=model.parameters(),
    weight_decay=0.01,
    apply_decay_param_fun=lambda x: x in decay_params)
metric = ChunkEvaluator(label_list=train_ds.label_list, suffix=True)

```

### 3 训练
<font size=3>下面是模型训练，有两个注意点：</font>
- model传入参数时，需要给定labels，因为训练crf是需要通过损失函数进行的，评价时则不需要给出labels，此时返回的是预测标签结果
- metric参数位置兼容老版本，因此两种方式都行，当然以后还是按照说明文档来比较好！


```python
from utils import evaluate
import sys
# 开启训练
global_step = 0
global_accu = 0.0
tic_train = time.time()

for epoch in range(1, epochs + 1):
    for step, batch in enumerate(train_data_loader, start=1):
        input_ids, token_type_ids, seq_lens, labels = batch
        loss = model(input_ids, token_type_ids, seq_lens=seq_lens, labels=labels)
        # loss = model(input_ids, token_type_ids, seq_lens=seq_lens)
        # print(loss)
        # print(labels)
        # sys.exit()
        avg_loss = paddle.mean(loss)
        global_step += 1
        if global_step % 50 == 0 :
            print(
                "global step %d, epoch: %d, batch: %d, loss: %.5f, speed: %.2f step/s"
                % (global_step, epoch, step, avg_loss,
                    10 / (time.time() - tic_train)))
            tic_train = time.time()
        loss.backward()
        optimizer.step()
        optimizer.clear_grad()
        if global_step % 500 == 0 :
            eva_accu = evaluate(model, metric, dev_data_loader)
            #训练过程中保存最大验证结果的模型
            if eva_accu > global_accu:   
                print(f'evaluate accu: {eva_accu}>history accu:{global_accu} ==> save the model!')
                global_accu = eva_accu
                save_dir = os.path.join(f'checkpoint_{traindataset}_{MODEL_NAME}', "model_%d_%.3f" % (global_step,global_accu))
                if not os.path.exists(save_dir):
                    os.makedirs(save_dir)
                save_param_path = os.path.join(save_dir, 'model_state.pdparams')
                paddle.save(model.state_dict(), save_param_path)
```

### 4 预测


```python
#进行预测
def convert_example(example, tokenizer, max_seq_length=512, is_test=False):
    """
    Builds model inputs from a sequence or a pair of sequence for sequence classification tasks
    by concatenating and adding special tokens. And creates a mask from the two sequences passed 
    to be used in a sequence-pair classification task.
        
    A skep_ernie_1.0_large_ch/skep_ernie_2.0_large_en sequence has the following format:
    ::
        - single sequence: ``[CLS] X [SEP]``
        - pair of sequences: ``[CLS] A [SEP] B [SEP]``

    A skep_ernie_1.0_large_ch/skep_ernie_2.0_large_en sequence pair mask has the following format:
    ::

        0 0 0 0 0 0 0 0 0 0 0 1 1 1 1 1 1 1 1 1
        | first sequence    | second sequence |

    If `token_ids_1` is `None`, this method only returns the first portion of the mask (0s).

    Args:
        example(obj:`list[str]`): List of input data, containing text and label if it have label.
        tokenizer(obj:`PretrainedTokenizer`): This tokenizer inherits from :class:`~paddlenlp.transformers.PretrainedTokenizer` 
            which contains most of the methods. Users should refer to the superclass for more information regarding methods.
        max_seq_len(obj:`int`): The maximum total input sequence length after tokenization. 
            Sequences longer than this will be truncated, sequences shorter will be padded.

    Returns:
        input_ids(obj:`list[int]`): The list of token ids.
        token_type_ids(obj: `list[int]`): List of sequence pair mask. 
    """
    tokens = example["tokens"]
    encoded_inputs = tokenizer(
        tokens,
        return_length=True,
        is_split_into_words=True,
        max_seq_len=max_seq_length)
    input_ids = encoded_inputs["input_ids"]
    token_type_ids = encoded_inputs["token_type_ids"]
    seq_len = encoded_inputs["seq_len"]

    return input_ids, token_type_ids, seq_len


@paddle.no_grad()
def predict(model, data_loader, label_map):
    """
    Given a prediction dataset, it gives the prediction results.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        label_map(obj:`dict`): The label id (key) to label str (value) map.
    """
    model.eval()
    results = []
    for input_ids, token_type_ids, seq_lens in data_loader:
        preds = model(input_ids, token_type_ids, seq_lens=seq_lens)
        tags = parse_predict_result(preds.numpy(), seq_lens.numpy(), label_map)
        results.extend(tags)
    return results


def parse_predict_result(predictions, seq_lens, label_map):
    """
    Parses the prediction results to the label tag.
    """
    pred_tag = []
    for idx, pred in enumerate(predictions):
        seq_len = seq_lens[idx]
        # drop the "[CLS]" and "[SEP]" token
        tag = [label_map[i] for i in pred[1:seq_len - 1]]
        pred_tag.append(tag)
    return pred_tag


def create_dataloader(dataset,
                      mode='train',
                      batch_size=1,
                      batchify_fn=None,
                      trans_fn=None):
    if trans_fn:
        dataset = dataset.map(trans_fn)

    shuffle = True if mode == 'train' else False
    if mode == 'train':
        batch_sampler = paddle.io.DistributedBatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)
    else:
        batch_sampler = paddle.io.BatchSampler(
            dataset, batch_size=batch_size, shuffle=shuffle)

    return paddle.io.DataLoader(
        dataset=dataset,
        batch_sampler=batch_sampler,
        collate_fn=batchify_fn,
        return_list=True)
```


```python
#载入模型

label_map = {0: "B", 1: "I", 2: "O"}
no_entity_label_idx = 2

skep = SkepModel.from_pretrained(MODEL_NAME)
model = SkepCrfForTokenClassification(skep, num_classes=len(test_ds.label_list))
tokenizer = SkepTokenizer.from_pretrained(MODEL_NAME)

params_path = os.path.join(f'checkpoint_{traindataset}_{MODEL_NAME}', 'model_2400_0.878','model_state.pdparams')
if params_path and os.path.isfile(params_path):
    # 加载模型参数
    state_dict = paddle.load(params_path)
    model.set_dict(state_dict)
    print("Loaded parameters from %s" % params_path)
else:
    print("Can not load parameters from %s, please check your path!" % params_path)

```

    [2021-06-17 16:53:48,406] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch/skep_ernie_1.0_large_ch.pdparams
    [2021-06-17 16:53:52,012] [    INFO] - Found /home/aistudio/.paddlenlp/models/skep_ernie_1.0_large_ch/skep_ernie_1.0_large_ch.vocab.txt


    Loaded parameters from checkpoint_COTE-DP_skep_ernie_1.0_large_ch/model_2400_0.878/model_state.pdparams



```python
trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=max_seq_length)
batchify_fn = lambda samples, fn=Tuple(
    Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # input ids
    Pad(axis=0, pad_val=tokenizer.vocab[tokenizer.pad_token]),  # token type ids
    Stack(dtype='int64'),  # sequence lens
): [data for data in fn(samples)]

test_data_loader = create_dataloader(
    test_ds,
    mode='test',
    batch_size=batch_size,
    batchify_fn=batchify_fn,
    trans_fn=trans_func)

results = predict(model, test_data_loader, label_map)

```


```python
# data = test_ds.data[0]['tokens']
# label = results[0]
def get_words(data,label):
    word = ''
    words = []
    addFlag = False
    for i,token in enumerate(label):
        if token == 'B':
            addFlag = True
            word += data[i]
        elif token == 'I':
            addFlag = True
            word += data[i]
        elif token == 'O' and addFlag:
            words += [word]
            word = ''
            addFlag = False
    return words
# return '\t'.join(words)
# print(get_words(data,label))
```


```python
with open(os.path.join("results", f"{traindataset}.tsv"), 'w', encoding="utf8") as f:
    f.write("index\tprediction\n")
    for idx, example in enumerate(test_ds.data):
        try:
            words = get_words(example['tokens'],results[idx])
        except Exception as e:
            print(example['tokens'], results[idx],'\x01'.join(words))
            raise e
        f.write(str(idx)+"\t"+'\x01'.join(words)+"\n")
        
        if len(words) != 1:
            print(len(example['tokens']), len(results[idx]))
            print(example['tokens'], results[idx],'\x01'.join(words))
```

    39 39
    ['【', '品', '牌', '延', '伸', '】', '8', '5', '°', 'C', '是', '一', '家', '以', '咖', '啡', '蛋', '糕', '、', '蛋', '糕', '烘', '培', '为', '主', '的', '专', '卖', '店', '，', '8', '5', '度', 'C', '是', '其', '招', '牌', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'O', 'O', 'O', 'O', 'O'] 85°C度C
    43 43
    ['跟', '小', '伙', '伴', '们', '逛', '街', '走', '累', '了', '就', '到', '8', '5', '度', 'C', '休', '息', '，', '正', '好', '霸', '王', '葡', '萄', '柚', '做', '活', '动', '：', '第', '二', '杯', '半', '价', '，', '就', '点', '了', '杯', '试', '试', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 85度C葡萄柚
    36 36
    ['金', '陵', '饭', '店', '樱', '花', '苑', '烤', '青', '花', '鱼', '套', '餐', '在', '大', '众', '美', '团', '搞', '秒', '杀', '活', '动', '只', '要', '9', '.', '9', '优', '惠', '力', '度', '很', '大', '喔', '。'] ['B', 'I', 'I', 'I', 'O', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 金陵饭店花苑
    78 78
    ['这', '里', '的', '海', '蛎', '饼', '也', '是', '相', '当', '得', '不', '错', '，', '嚼', '一', '口', '海', '蛎', '饼', '，', '配', '上', '一', '口', '花', '生', '汤', '，', '真', '的', '是', '可', '以', '回', '味', '上', '一', '下', '午', '了', '总', '之', '这', '是', '家', '令', '人', '流', '连', '忘', '返', '、', '百', '吃', '不', '厌', '的', '小', '吃', '店', '，', '如', '果', '可', '以', '，', '我', '愿', '意', '每', '天', '来', '光', '顾', '一', '次', '！'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 
    42 42
    ['长', '沙', '唯', '一', '的', '一', '家', '正', '宗', '苦', '瓜', '烧', '鱼', '，', '味', '道', '棒', '棒', '哒', '!', '地', '址', '在', '金', '源', '大', '酒', '店', '旁', '边', '的', '巷', '内', '，', '很', '小', '的', '一', '家', '门', '面', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'B', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 正苦瓜烧鱼
    44 44
    ['巴', '贝', '拉', '意', '式', '休', '闲', '餐', '厅', '位', '于', '万', '达', '三', '楼', '，', '占', '地', '面', '积', '不', '小', '。', '同', '行', '几', '人', '点', '的', '是', '牛', '排', '自', '助', '，', '平', '均', '一', '位', '6', '8', '左', '右', '。'] ['B', 'I', 'I', 'O', 'O', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 巴贝拉休闲餐厅
    42 42
    ['还', '有', '主', '打', '的', '披', '萨', '，', '个', '人', '觉', '得', '好', '伦', '哥', '终', '于', '胜', '也', '披', '萨', '，', '败', '也', '披', '萨', '了', '。', '不', '，', '是', '胜', '也', '自', '助', '，', '败', '也', '自', '助', '了', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 好伦哥助
    60 60
    ['【', '评', '价', '】', '全', '聚', '德', '烤', '鸭', '是', '北', '京', '的', '特', '色', '，', '这', '家', '店', '在', '银', '川', '也', '开', '了', '很', '多', '年', '，', '装', '修', '和', '北', '京', '店', '差', '不', '多', '，', '古', '香', '古', '色', '的', '，', '但', '装', '修', '了', '有', '年', '头', '，', '显', '得', '有', '些', '陈', '旧', '。'] ['O', 'O', 'O', 'O', 'B', 'I', 'I', 'O', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 全聚德鸭
    66 66
    ['昨', '天', '趁', '着', '有', '时', '间', '就', '陪', '着', '夫', '人', '去', '美', '哒', '哒', '了', '，', '早', '上', '九', '点', '半', '出', '发', '坐', '公', '交', '换', '二', '次', '地', '铁', '到', '北', '仑', '的', 'P', 'E', 'E', 'K', 'A', 'B', 'O', 'O', '（', '皮', '卡', '博', '）', '美', '发', '沙', '龙', '店', '里', '已', '经', '是', '中', '午', '十', '二', '点', '了', '；'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'I', 'I', 'O', 'B', 'I', 'I', 'I', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] ABOO皮卡博）美发沙龙
    97 97
    ['所', '以', '好', '久', '没', '来', '了', '，', '最', '近', '农', '行', '搞', '活', '动', '，', '买', '一', '送', '一', '，', '晚', '餐', '平', '均', '下', '来', '1', '3', '8', '元', '一', '人', '，', '加', '上', '貌', '似', '是', '停', '业', '过', '重', '开', '，', '因', '为', '星', '城', '大', '酒', '店', '大', '堂', '上', '维', '景', '餐', '厅', '的', '楼', '梯', '口', '上', '放', '了', '个', '牌', '子', '，', '写', '着', '“', '维', '景', '餐', '厅', '隆', '重', '回', '归', '“', '，', '所', '以', '与', '朋', '友', '四', '人', '来', '吃', '了', '个', '晚', '餐', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 星城大酒店维景餐厅
    75 75
    ['4', '.', '再', '来', '就', '是', '妈', '妈', '红', '烧', '肉', '啦', '，', '现', '在', '家', '里', '没', '有', '那', '么', '多', '时', '间', '慢', '炖', '这', '类', '菜', '了', '，', '所', '以', '这', '种', '妈', '妈', '菜', '的', '口', '感', '不', '容', '错', '过', '滴', '，', '估', '计', '放', '了', '玫', '瑰', '腐', '乳', '，', '虽', '然', '不', '太', '喜', '欢', '烧', '红', '烧', '肉', '放', '大', '料', '，', '可', '是', '.', '.', '.'] ['O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'O', 'O', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 妈妈肉
    284 254
    ['周', '末', '去', '万', '达', '溜', '达', '溜', '达', '董', '小', '姐', '不', '等', '位', '的', '生', '意', '是', '真', '的', '好', '，', '哪', '里', '不', '用', '等', '，', '去', '的', '晚', '了', '，', '没', '有', '位', '置', '啊', '等', '着', '吧', '谁', '让', '我', '想', '吃', '这', '家', '的', '沸', '腾', '鱼', '呢', '水', '街', '的', '餐', '饮', '没', '有', '里', '面', '多', '，', '现', '在', '有', '几', '家', '也', '关', '门', '了', '董', '小', '姐', '的', '生', '意', '棒', '棒', '的', '，', '中', '午', '去', '的', '，', '居', '然', '爆', '满', '了', '里', '面', '有', '点', '黑', '，', '蓝', '蓝', '红', '红', '的', '其', '实', '可', '以', '稍', '微', '明', '亮', '一', '点', '的', '基', '本', '每', '一', '桌', '都', '点', '了', '沸', '腾', '鱼', '，', '超', '级', '大', '的', '一', '份', '，', '豆', '芽', '菜', '铺', '底', '的', '，', '可', '以', '忽', '略', '啦', '，', '我', '就', '吃', '鱼', '鱼', '片', '很', '嫩', '，', '没', '有', '鱼', '刺', '，', '好', '大', '好', '大', '的', '一', '片', '，', '鱼', '肉', '特', '别', '的', '多', '，', '2', '个', '人', '吃', '的', '很', '满', '意', '点', '的', '一', '份', '茄', '子', '煲', '稍', '微', '有', '点', '咸', '了', '，', '不', '过', '配', '米', '饭', '还', '可', '以', '量', '也', '还', '行', '董', '小', '姐', '的', '菜', '价', '个', '人', '感', '觉', '有', '点', '小', '贵', '，', '尤', '其', '蔬', '菜', '可', '以', '推', '几', '个', '特', '价', '菜', '这', '样', '吸', '引', '人', '不', '过', '团', '购', '的', '套', '餐', '很', '划', '算', '的', '沸', '腾', '鱼', '，', '土', '豆', '丝', '，', '色', '拉', '2', '个', '人', '吃', '可', '以', '的', '，', '小', '资', '下', '啦', '董', '小', '姐', '还', '有', '.', '.', '.'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 小姐沸腾鱼
    131 131
    ['今', '天', '位', '于', '南', '长', '街', '的', '蜀', '九', '香', '火', '锅', '1', '.', '9', '折', '，', '我', '和', '麻', '麻', '9', ':', '3', '0', '左', '右', '就', '到', '达', '目', '的', '地', '了', '，', '谁', '知', '早', '已', '有', '十', '几', '人', '已', '经', '在', '等', '了', '，', '商', '家', '支', '了', '个', '小', '棚', '子', '，', '还', '配', '备', '了', '一', '个', '大', '风', '扇', '，', '要', '1', '0', ':', '3', '0', '开', '始', '叫', '号', '，', '后', '来', '不', '少', '人', '向', '商', '家', '反', '映', '，', '1', '0', '点', '不', '到', '按', '照', '先', '来', '后', '到', '的', '顺', '序', '就', '取', '号', '了', '，', '1', '1', '点', '开', '餐', '，', '这', '样', '大', '家', '就', '不', '用', '一', '直', '耗', '在', '这', '里', '了', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'O', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 蜀九香锅
    128 128
    ['”', '“', '是', '的', '，', '是', '的', '，', '是', '的', '，', '简', '直', '难', '以', '置', '信', '这', '是', '真', '的', '，', '双', '1', '2', '在', '也', '不', '用', '“', '剁', '手', '”', '了', '小', '伙', '伴', '们', '还', '等', '啥', '，', '别', '给', '自', '己', '的', '嘴', '和', '胃', '留', '在', '2', '0', '1', '5', '的', '遗', '憾', '，', '来', '多', '少', '喝', '多', '少', '，', '来', '多', '少', '送', '多', '少', '，', '来', '多', '少', '再', '也', '不', '用', '“', '剁', '手', '”', '么', '么', '哒', '地', '址', '：', '昆', '明', '市', '人', '民', '西', '路', '保', '利', '六', '合', '中', '心', '美', '食', '生', '活', '街', '区', 'B', '6', '来', '自', 'H', 'O', 'N', 'G', '\xa0', 'K', 'O', 'N', 'G', '的', '大', '通', '冰', '室'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I'] 
    31 31
    ['竹', '筒', '酸', '笋', '牛', '肉', '也', '是', '一', '道', '下', '饭', '菜', '，', '酸', '汤', '泡', '碗', '饭', '完', '全', '可', '以', '爽', '翻', '你', '。', '老', '滇', '山', '寨'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I'] 
    159 159
    ['2', '0', '1', '6', '.', '0', '8', '.', '0', '3', '\xa0', '口', '留', '香', '煎', '饼', '果', '子', '（', 'C', 'B', 'D', '万', '达', '店', '）', '在', '青', '岛', '的', '那', '些', '天', '住', '在', '万', '达', '附', '近', '，', '前', '几', '天', '都', '累', '得', '半', '死', '，', '早', '餐', '都', '是', '妈', '妈', '直', '接', '到', '酒', '店', '附', '近', '随', '便', '买', '的', '，', '吃', '不', '大', '惯', '，', '于', '是', '最', '后', '一', '天', '我', '起', '了', '个', '大', '早', '，', '在', '点', '评', '搜', '了', '搜', '附', '近', '比', '较', '有', '名', '的', '小', '吃', '类', '，', '刚', '好', '搜', '到', '了', '这', '家', '和', '味', '为', '先', '豆', '腐', '脑', '，', '貌', '似', '在', '青', '岛', '早', '餐', '界', '都', '挺', '有', '名', '，', '想', '和', '妈', '妈', '把', '两', '家', '店', '都', '拔', '草', '了', '，', '于', '是', '就', '先', '去', '了', '这', '家', '买', '了', '个', '煎', '饼', '果', '子', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'I', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 口香煎饼果子和味为先豆腐脑
    49 49
    ['<', '蚝', '火', '记', '•', '渔', '乐', '场', '>', '长', '春', '海', '鲜', '榜', '单', '上', '的', '第', '一', '家', '曾', '和', '小', '伙', '伴', '一', '起', '种', '草', '，', '怎', '料', '我', '中', '了', '同', '城', '聚', '会', '哈', '哈', '哈', '我', '先', '去', '探', '探', '路', '。'] ['O', 'B', 'I', 'I', 'O', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 蚝火记渔乐场
    243 243
    ['以', '前', '的', '狗', '不', '理', '快', '餐', '店', '改', '的', '三', '不', '馆', '，', '以', '前', '只', '知', '道', '东', '站', '后', '广', '场', '那', '家', '，', '在', '大', '众', '点', '评', '搜', '索', '发', '现', '同', '安', '道', '有', '一', '家', '就', '过', '来', '了', '，', '在', '大', '众', '点', '评', '订', '的', '位', '子', '，', '到', '了', '之', '后', '发', '现', '环', '境', '还', '比', '较', '温', '馨', '，', '因', '为', '是', '地', '道', '的', '天', '津', '菜', '馆', '就', '点', '了', '噌', '蹦', '鲤', '鱼', '，', '醋', '椒', '豆', '腐', '，', '大', '拌', '菜', '，', '还', '有', '三', '不', '馆', '必', '点', '的', '肉', '龙', '，', '先', '说', '说', '鲤', '鱼', '，', '传', '统', '的', '做', '法', '应', '该', '是', '手', '拿', '鱼', '嘴', '过', '油', '炸', '，', '鱼', '头', '一', '半', '应', '该', '是', '没', '有', '炸', '过', '，', '显', '然', '不', '是', '传', '统', '的', '做', '法', '，', '所', '以', '味', '道', '也', '就', '那', '么', '回', '事', '，', '必', '点', '的', '肉', '龙', '相', '当', '的', '失', '望', '，', '还', '没', '有', '家', '里', '爸', '爸', '做', '的', '好', '吃', '，', '其', '余', '两', '个', '菜', '也', '都', '很', '一', '般', '，', '服', '务', '员', '业', '务', '也', '很', '不', '熟', '练', '，', '电', '子', '点', '餐', '，', '都', '是', '男', '服', '务', '生', '，', '一', '脸', '的', '不', '高', '兴', '，', '就', '跟', '去', '那', '吃', '饭', '的', '都', '欠', '他', '钱', '似', '的', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 
    23 23
    ['多', '乐', '之', '日', '，', '看', '《', '欢', '乐', '颂', '》', '时', '候', '种', '下', '的', '草', '一', '直', '未', '能', '拔', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 
    169 169
    ['偶', '遇', '这', '一', '清', '新', '之', '所', '，', '置', '身', '其', '中', '仿', '佛', '时', '间', '都', '变', '静', '了', '变', '慢', '了', '…', '…', '位', '置', '位', '于', '京', '华', '城', '三', '楼', '东', '南', '角', '，', '环', '境', '非', '常', '有', '文', '化', '气', '息', '，', '店', '内', '装', '饰', '古', '色', '古', '韵', '又', '结', '合', '现', '代', '时', '尚', '元', '素', '，', '经', '营', '着', '茶', '杯', '茶', '具', '茶', '叶', '以', '及', '男', '女', '服', '饰', '等', '，', '也', '提', '供', '茶', '类', '饮', '品', '供', '大', '家', '选', '择', '，', '有', '蜜', '宗', '茶', '、', '冰', '茶', '、', '红', '茶', '、', '绿', '茶', '、', '普', '洱', '铁', '观', '音', '等', '等', '“', '茶', '之', '然', '茶', '中', '星', '巴', '克', '”', '是', '他', '家', '的', '广', '告', '标', '语', '“', '创', '意', '茶', '文', '化', '，', '时', '尚', '茶', '生', '活', '”', '是', '他', '的', '态', '度', '难', '得', '的', '清', '静', '之', '地', '，', '下', '次', '还', '会', '再', '来', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'I', 'O', 'O', 'B', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 茶然星巴克
    166 166
    ['\xa0', '在', '几', '次', '和', '家', '里', '的', '几', '个', '主', '子', '商', '量', '后', '，', '他', '们', '一', '致', '下', '令', '，', '要', '我', '这', '个', '专', '业', '跑', '腿', '的', '去', '买', '蛋', '糕', '，', '但', '是', '太', '后', '怕', '我', '买', '着', '不', '合', '她', '的', '心', '意', '，', '于', '是', '决', '定', '亲', '自', '选', '购', '，', '一', '一', '小', '公', '举', '也', '要', '跟', '着', '，', '于', '是', '三', '个', '人', '一', '起', '来', '到', '了', '位', '于', '井', '冈', '山', '大', '道', '3', '0', '0', '号', '卡', '拉', '多', '(', '家', '乐', '福', '店', ')', '，', '然', '而', '去', '的', '太', '晚', '，', '面', '点', '师', '傅', '已', '下', '班', '，', '正', '好', '碰', '到', '一', '款', '蛋', '糕', '打', '折', '，', '用', '微', '信', '支', '付', '立', '减', '2', '0', '，', '于', '是', '太', '后', '下', '令', '就', '买', '这', '款', '了', '，', '还', '特', '别', '给', '一', '一', '的', '小', '闺', '蜜', '也', '买', '了', '一', '个', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'O', 'O', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 卡拉多乐福
    214 214
    ['种', '草', '已', '久', '的', '一', '家', '餐', '厅', '2', '0', '1', '5', '年', '中', '国', '5', '0', '佳', '候', '选', '餐', '厅', '之', '一', '设', '有', '包', '厢', '和', '大', '厅', '\xa0', '包', '厢', '都', '是', '独', '栋', '小', '楼', '\xa0', '私', '密', '性', '很', '好', '但', '数', '量', '有', '限', '大', '厅', '位', '置', '也', '不', '多', '装', '修', '并', '不', '金', '碧', '辉', '煌', '简', '单', '但', '有', '韵', '味', '的', '木', '质', '中', '国', '风', '推', '荐', '钱', '湖', '朋', '鱼', '\xa0', '朋', '鱼', '和', '青', '鱼', '是', '东', '钱', '湖', '两', '大', '名', '鱼', '之', '前', '在', '水', '上', '餐', '厅', '和', '农', '家', '乐', '也', '吃', '过', '很', '多', '次', '但', '和', '钱', '湖', '渔', '港', '的', '这', '道', '菜', '相', '比', '\xa0', '完', '全', '不', '是', '一', '个', '档', '次', '看', '似', '简', '单', '的', '清', '蒸', '\xa0', '却', '能', '把', '鱼', '肉', '烹', '制', '的', '细', '腻', '鲜', '美', '如', '斯', '回', '味', '还', '带', '着', '鱼', '肉', '的', '鲜', '甜', '\xa0', '感', '觉', '一', '百', '多', '的', '价', '格', '也', '并', '不', '贵', '相', '比', '而', '言', '\xa0', '茄', '子', '炒', '年', '糕', '和', '脆', '皮', '鸡', '就', '比', '较', '中', '规', '中', '矩', '了', '\xa0', '并', '没', '有', '朋', '鱼', '那', '么', '惊', '艳'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 鱼渔港
    21 21
    ['白', '沙', '大', '道', '的', '真', '好', '吃', '牛', '杂', '，', '东', '记', '湛', '江', '鸡', '旁', '边', '一', '点', '。'] ['O', 'O', 'O', 'O', 'O', 'B', 'O', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 真吃牛杂
    115 115
    ['☄', 'F', 'a', 'r', 'e', 'w', 'e', 'l', 'l', ',', 'T', 's', 'i', 'n', 'g', 'T', 'a', 'o', '☄', '\xa0', '♛', '\xa0', '♪', '♫', '⚐', '\xa0', '\xa0', 'θ', '\xa0', '♛', '再', '见', '青', '岛', '拾', '肆', '章', '，', '说', '说', '披', '萨', '烤', '盘', '版', '本', '的', '大', '集', '桥', '底', '烧', '烤', '留', '意', '酒', '肉', '屋', '已', '久', '，', '没', '想', '到', '初', '探', '竟', '然', '是', '在', '极', '地', '海', '洋', '世', '界', '周', '边', '正', '餐', '选', '择', '少', '，', '游', '客', '那', '么', '多', '为', '何', '就', '餐', '人', '数', '寥', '寥', '无', '几', '\xa0', '症', '结', '在', '于', '服', '务', '差', '，', '不', '是', '一', '般', '的', '差', '！'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 集桥底烧烤肉屋
    200 200
    ['银', '泰', '城', '太', '大', '了', '，', '我', '每', '次', '去', '都', '会', '迷', '路', '，', '银', '泰', '城', '作', '为', '淄', '博', '比', '较', '大', '的', '城', '市', '综', '合', '体', '位', '于', '北', '边', '，', '对', '于', '我', '在', '西', '边', '住', '的', '人', '来', '说', '有', '点', '远', '，', '但', '是', '交', '通', '比', '较', '方', '便', '，', '而', '且', '包', '含', '的', '比', '较', '广', '阔', '，', '购', '物', '超', '市', '，', '电', '影', '娱', '乐', '都', '可', '以', '不', '出', '门', '完', '成', '，', '还', '是', '比', '较', '愿', '意', '带', '孩', '子', '去', '玩', '的', '，', '孩', '子', '很', '喜', '欢', '的', '蚂', '蚁', '王', '国', '，', '每', '次', '进', '去', '都', '带', '不', '出', '来', '的', '，', '经', '常', '去', '那', '里', '吃', '饭', '，', '而', '且', '中', '影', '的', '影', '院', '也', '比', '较', '好', '，', '带', '孩', '子', '去', '看', '过', '超', '能', '陆', '战', '队', '，', '效', '果', '太', '棒', '了', '，', '经', '常', '会', '有', '活', '动', '在', '那', '里', '进', '行', '，', '还', '曾', '经', '去', '参', '加', '过', '时', '尚', '秀', '，', '在', '淄', '博', '来', '说', '的', '话', '，', '场', '地', '算', '是', '很', '大', '了'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 
    106 106
    ['远', '远', '就', '看', '到', '益', '聚', '园', '的', '招', '牌', '，', '招', '牌', '下', '面', '还', '写', '着', '"', '北', '菜', '南', '厨', '"', '四', '个', '大', '字', '，', '刚', '开', '始', '还', '不', '太', '理', '解', '，', '但', '是', '饭', '店', '老', '板', '一', '解', '释', '就', '明', '白', '了', '，', '原', '来', '他', '们', '家', '做', '的', '是', '上', '海', '菜', '，', '厨', '师', '是', '正', '宗', '上', '海', '人', '，', '但', '是', '菜', '的', '口', '味', '是', '经', '过', '改', '良', '，', '融', '合', '了', '当', '地', '人', '的', '口', '味', '，', '吃', '完', '后', '感', '觉', '名', '副', '其', '实', '。'] ['O', 'O', 'O', 'O', 'O', 'B', 'O', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 益园南厨
    303 254
    ['位', '置', '很', '好', '找', '，', '就', '在', '台', '湾', '城', '最', '东', '侧', '，', '环', '境', '也', '不', '错', '，', '临', '河', '而', '建', '，', '晚', '上', '吃', '着', '龙', '虾', '吹', '着', '凉', '风', '欣', '赏', '河', '景', '，', '有', '种', '在', '江', '南', '的', '韵', '味', '，', '还', '有', '在', '海', '边', '吃', '大', '排', '档', '的', '感', '觉', '，', '室', '内', '环', '境', '非', '常', '卫', '生', '，', '吃', '龙', '虾', '算', '是', '高', '档', '的', '，', '请', '客', '聚', '会', '拿', '得', '出', '面', '，', '比', '路', '边', '地', '摊', '强', '多', '了', '，', '这', '些', '是', '次', '要', '的', '，', '味', '道', '非', '常', '不', '错', '，', '虾', '是', '外', '地', '的', '，', '仔', '细', '观', '察', '虾', '腿', '清', '洗', '比', '较', '干', '净', '，', '肉', '比', '烧', '烤', '摊', '的', '大', '很', '多', '，', '这', '里', '龙', '虾', '非', '常', '新', '鲜', '，', '肉', '是', '白', '嫩', '，', '活', '虾', '加', '工', '的', '，', '很', '多', '外', '面', '烧', '烤', '摊', '卖', '的', '是', '菜', '市', '场', '送', '的', '死', '虾', '，', '肉', '是', '黑', '的', '，', '常', '吃', '的', '一', '对', '比', '就', '清', '楚', '了', '，', '这', '里', '龙', '虾', '五', '种', '口', '味', '，', '各', '有', '特', '色', '，', '推', '荐', '十', '三', '香', '，', '蒜', '泥', '，', '干', '煸', '的', '，', '别', '的', '地', '方', '不', '一', '定', '能', '吃', '到', '，', '味', '道', '很', '爽', '，', '把', '龙', '虾', '做', '成', '这', '样', '也', '是', '绝', '了', '，', '对', '得', '起', '招', '牌', '，', '绝', '味', '龙', '虾', '，', '本', '人', '一', '般', '不', '吃', '龙', '虾', '，', '因', '为', '不', '放', '心', '卫', '生', '，', '在', '这', '里', '可', '以', '大', '胆', '品', '尝', '，', '加', '工', '比', '较', '到', '位', '，', '别', '的', '地', '方', '.', '.', '.'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 吃虾
    37 37
    ['广', '东', '湛', '江', '安', '铺', '鸡', '休', '闲', '餐', '厅', '位', '为', '葛', '村', '路', '上', '。', '左', '边', '是', '粉', '之', '都', '，', '右', '边', '是', '中', '国', '兰', '州', '牛', '肉', '拉', '面', '。'] ['O', 'I', 'I', 'I', 'B', 'I', 'I', 'O', 'O', 'O', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 东湛江安铺鸡厅
    73 73
    ['这', '次', '看', '着', '时', '间', '还', '够', '就', '过', '来', '试', '试', '看', '，', '结', '果', '真', '的', '在', '，', '突', '然', '就', '觉', '得', '有', '点', '小', '感', '动', 'π', '_', 'π', '\xa0', '原', '味', '的', '马', '迭', '尔', '冰', '淇', '淋', '我', '还', '没', '有', '吃', '过', '，', '虽', '然', '天', '气', '还', '比', '较', '冷', '，', '但', '是', '吃', '着', '有', '一', '种', '不', '一', '样', '的', '感', '觉'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'O', 'O', 'O', 'B', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] π 马迭尔
    108 108
    ['该', '门', '店', '位', '于', '福', '州', '市', '金', '融', '家', '万', '达', '室', '内', '广', '场', '3', 'F', '营', '业', '时', '间', '：', '1', '0', ':', '0', '0', '2', '2', ':', '0', '0', '停', '车', '较', '为', '方', '便', '，', '万', '达', '底', '下', '1', '层', '，', '2', '层', '多', '可', '以', '停', '车', '买', '的', '是', '大', '众', '美', '团', '团', '购', '券', '，', '之', '前', '在', 'S', 'M', '一', '期', '哪', '家', 'D', 'Q', '不', '让', '用', '，', '很', '是', '生', '气', '，', '今', '日', '来', '福', '州', '出', '差', '，', '刚', '好', '想', '起', '还', '有', '1', '张', '券', '就', '过', '来', '了', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 
    78 78
    ['来', '说', '说', '菜', '品', '：', '朱', '小', '乐', '的', '龙', '虾', '生', '活', '：', '上', '的', '第', '一', '份', '龙', '虾', '，', '两', '层', '喔', '，', '中', '间', '是', '锅', '巴', '，', '锅', '巴', '蘸', '着', '底', '下', '的', '汁', '儿', '吃', '，', '很', '好', '吃', '，', '龙', '虾', '很', '大', '，', '虾', '腮', '也', '很', '干', '净', '，', '虾', '黄', '很', '多', '哦', '最', '喜', '欢', '蛋', '黄', '虾', '：', '锅', '巴', '超', '好', '吃', '！'] ['O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'O', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 朱小乐龙虾生活
    57 57
    ['对', '比', '探', '鱼', '，', '鱼', '太', '和', '探', '炉', '三', '家', '的', '烤', '鱼', '，', '价', '钱', '基', '本', '一', '样', '都', '是', '1', '3', '8', '左', '右', '，', '鱼', '的', '分', '量', '以', '及', '配', '菜', '的', '分', '量', '，', '探', '鱼', '最', '优', '、', '探', '炉', '次', '之', '，', '鱼', '太', '最', '少', '。'] ['O', 'O', 'B', 'I', 'O', 'O', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 探鱼太
    45 45
    ['我', '想', '说', '，', '这', '家', '店', '真', '的', '对', '不', '起', '它', '的', '名', '字', '！', '回', '家', '吃', '饭', '，', '吃', '了', '一', '次', '绝', '对', '不', '会', '想', '吃', '第', '二', '次', '了', '！', '因', '为', '菜', '太', '难', '吃', '了', '！'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 
    103 103
    ['从', '8', '月', '1', '4', '起', '一', '直', '到', '9', '月', '1', '0', '日', '，', '萌', '哒', '哒', '的', '小', '黄', '人', '在', '8', '5', '度', 'C', '上', '线', '了', '，', '每', '家', '店', '都', '贴', '着', '大', '大', '的', '小', '黄', '人', '宣', '传', '海', '报', '，', '单', '笔', '消', '费', '满', '2', '5', '元', '就', '可', '以', '以', '优', '惠', '任', '务', '价', '2', '5', '元', '购', '买', '大', '眼', '萌', '小', '黄', '人', '造', '型', '凉', '爽', '瓶', '，', '总', '共', '三', '款', '造', '型', '，', '直', '接', '购', '买', '的', '价', '格', '是', '6', '9', '元', '一', '个', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'O', 'B', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 人85度C
    96 96
    ['第', '一', '次', '来', '万', '达', '店', '喝', '，', '环', '境', '比', '较', '好', '，', '服', '务', '也', '不', '错', '选', '的', '黑', '龙', '茶', '，', '据', '说', '可', '以', '消', '脂', '，', '最', '近', '真', '的', '有', '胖', '很', '多', '味', '道', '还', '不', '错', '，', '糖', '量', '要', '的', '正', '常', '，', '也', '不', '是', '很', '甜', '，', '感', '觉', '还', '蛮', '对', '口', '味', '的', '缤', '纷', '果', '绿', '茶', '也', '好', '喝', '，', '入', '口', '酸', '酸', '甜', '甜', '的', '，', '还', '有', '绿', '茶', '的', '清', '香', '气', '，', '推', '荐'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 
    43 43
    ['同', '事', '的', '朋', '友', '请', '客', '在', '这', '家', '吃', '饭', '，', '订', '餐', '的', '时', '候', '还', '以', '为', '叫', '醉', '排', '骨', '呢', '，', '后', '来', '到', '店', '了', '才', '发', '现', '其', '实', '是', '叫', '醉', '得', '意', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 
    193 193
    ['这', '家', '火', '锅', '开', '业', '时', '候', '就', '知', '道', '一', '直', '没', '有', '去', '\xa0', '开', '始', '听', '名', '字', '还', '以', '为', '是', '澡', '堂', '后', '来', '得', '知', '是', '火', '锅', '同', '时', '也', '有', '典', '故', '的', '6', '6', '6', '\xa0', '牌', '楼', '巷', '下', '车', '走', '两', '步', '就', '到', '了', '\xa0', '门', '头', '漕', '运', '码', '头', '四', '个', '大', '字', '牌', '匾', '霸', '气', '十', '足', '蹬', '蹬', '蹬', '赶', '紧', '找', '了', '位', '置', '坐', '下', '点', '菜', '开', '吃', '有', '个', '喝', '的', '大', '力', '推', '荐', '\xa0', '柠', '檬', '还', '是', '蜂', '蜜', '什', '么', '的', '\xa0', '酸', '酸', '甜', '甜', '很', '浓', '郁', '好', '喝', '不', '得', '了', '鸳', '鸯', '锅', '辣', '的', '好', '给', '力', '啊', '\xa0', '感', '觉', '越', '来', '越', '不', '能', '吃', '辣', '了', '\xa0', '黄', '喉', '毛', '肚', '百', '叶', '之', '类', '的', '放', '进', '去', '入', '味', '的', '很', '\xa0', '鱼', '片', '摆', '盘', '视', '觉', '效', '果', '\xa0', '除', '了', '拍', '还', '是', '拍', '\xa0', '也', '很', '嫩', '黑', '白', '棋', '盘', '是', '豆', '腐', '和', '鸭', '血', '\xa0', '创', '意', '呀', '嘿', '！'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 
    41 41
    ['۰', '•', '❀', '･', '*', ':', '｡', '.', '✿', '【', '综', '合', '：', '★', '★', '★', '★', '★', '】', '✿', '.', '｡', ':', '*', '･', '❀', '•', '۰', '\xa0', '老', '板', '厚', '不', '厚', '道', '不', '知', '道', '。', '。', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 
    37 37
    ['居', '然', '还', '有', '抢', '购', '的', '玉', '米', '粥', '，', '买', '了', '碗', '\xa0', '找', '了', '过', '来', '！', '在', '汉', '姆', '连', '锁', '酒', '店', '对', '面', '！', '大', '大', '一', '个', '粥', '字', '！'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O'] 汉姆连锁粥
    231 231
    ['来', '合', '肥', '办', '事', '的', '，', '两', '个', '人', '跑', '到', '这', '边', '逛', '逛', '，', '觉', '得', '这', '名', '字', '有', '意', '思', '，', '就', '进', '来', '了', '，', '环', '境', '还', '可', '以', '，', '就', '是', '灯', '光', '暗', '了', '点', '，', '服', '务', '员', '拿', '了', '菜', '单', '，', '象', '征', '性', '点', '了', '四', '个', '，', '可', '是', '菜', '来', '了', '，', '发', '现', '量', '太', '大', '了', '，', '根', '本', '吃', '不', '掉', '，', '服', '务', '员', '也', '不', '提', '醒', '，', '问', '她', '说', '能', '不', '能', '退', '，', '回', '答', '是', '厨', '房', '在', '做', '，', '不', '能', '，', '无', '语', '呢', '，', '才', '上', '一', '个', '菜', '，', '厨', '房', '就', '一', '起', '做', '四', '个', '吗', '，', '还', '有', '就', '是', '，', '单', '刚', '下', '几', '分', '钟', '，', '鱼', '就', '上', '了', '，', '太', '快', '了', '吧', '，', '真', '怀', '疑', '是', '不', '是', '热', '好', '的', '，', '八', '大', '碗', '小', '炒', '，', '满', '心', '以', '为', '是', '特', '色', '呢', '，', '结', '果', '里', '面', '满', '满', '的', '茶', '干', '啊', '，', '晕', '，', '要', '知', '道', '我', '老', '家', '就', '是', '盛', '产', '这', '个', '的', '，', '八', '大', '碗', '三', '丝', '，', '是', '满', '满', '的', '干', '丝', '，', '真', '会', '忽', '悠', '呢', '，', '凉', '拌', '海', '带', '不', '错', '，', '给', '个', '赞', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 八碗
    120 120
    ['以', '前', '在', '台', '东', '潇', '洒', '名', '派', '剪', '过', '，', '这', '次', '发', '现', '公', '司', '附', '近', '的', '店', '有', '团', '购', '就', '果', '断', '团', '了', '一', '张', '，', '当', '然', '还', '有', '一', '个', '原', '因', '就', '是', '好', '奇', '啥', '叫', '水', '疗', 's', 'p', 'a', '，', '到', '店', '验', '证', '了', '团', '购', '券', '存', '了', '包', '开', '始', '等', '待', '，', '一', '个', '帅', '哥', '帮', '我', '洗', '了', '头', '问', '起', '水', '疗', '的', '事', '情', '，', '说', '水', '疗', '服', '务', '就', '是', '一', '直', '头', '皮', '护', '理', '方', '式', '，', '我', '说', '在', '哪', '做', '，', '帅', '哥', '说', '我', '这', '就', '是', '在', '给', '你', '做', '！'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 名派水疗spa
    55 55
    ['味', '道', '上', '一', '如', '既', '往', '的', '好', '吃', '，', '我', '去', '尝', '试', '过', '很', '多', '家', '的', '泰', '式', '火', '锅', '，', '但', '是', '和', '城', '市', '花', '园', '的', '都', '没', '法', '比', '，', '论', '泰', '式', '火', '锅', '哪', '家', '强', '，', '城', '市', '花', '园', '滨', '江', '店', '！'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I', 'O', 'O', 'O', 'O'] 城市花园城市花园
    88 88
    ['最', '近', '据', '说', '装', '修', '一', '新', '，', '每', '天', '在', '出', '租', '车', '顶', '灯', '上', '看', '见', '它', '家', '的', '广', '告', '轮', '番', '滚', '动', '啊', '…', '…', '终', '于', '找', '了', '个', '夜', '晚', '拉', '上', '家', '人', '，', '首', '次', '探', '访', '王', '庄', '阿', '咪', '大', '排', '档', '，', '也', '接', '一', '回', '地', '气', '哈', '哈', '…', '…', '\xa0', '【', '地', '理', '位', '置', '】', '王', '庄', '阿', '咪', '，', '顾', '名', '思', '义', '位', '于', '王', '庄', '啦', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'B', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 王阿咪大排档
    29 29
    ['吃', '韩', '国', '烤', '肉', '还', '是', '非', '常', '推', '荐', '这', '家', '的', '，', '泊', '富', '广', '场', '最', '火', '爆', '的', '店', '，', '焱', '石', '烤', '肉'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'I'] 
    39 39
    ['之', '前', '在', '大', '众', '免', '单', '抽', '中', '了', '猪', '吉', '面', '缘', '家', '的', '幸', '福', '拉', '面', '，', '趁', '着', '今', '天', '晚', '上', '有', '空', '就', '一', '个', '人', '过', '来', '品', '尝', '下', '！'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 
    38 38
    ['南', '宁', '饭', '店', '的', '小', '嘟', '来', '食', '街', '是', '儿', '时', '的', '记', '忆', '，', '已', '经', '挺', '多', '年', '没', '去', '了', '，', '昨', '晚', '突', '然', '想', '起', '就', '去', '了', '一', '次', '。'] ['B', 'I', 'I', 'I', 'O', 'O', 'I', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 南宁饭店嘟来食街
    152 152
    ['环', '境', '真', '的', '很', '隐', '蔽', '耶', '差', '点', '就', '找', '不', '到', '地', '方', '了', '环', '境', '还', '真', '的', '很', '不', '错', '哦', '中', '午', '，', '也', '没', '啥', '人', '，', '很', '清', '净', '点', '了', '蛋', '糕', '和', '咖', '啡', '，', '这', '里', '主', '推', '精', '品', '咖', '啡', '但', '是', '对', '咖', '啡', '就', '不', '怎', '么', '了', '解', '，', '所', '以', '店', '长', '还', '问', '我', '的', '口', '味', '如', '何', '，', '再', '推', '荐', '咖', '啡', '给', '我', '厕', '所', '也', '设', '计', '的', '特', '别', '，', '就', '是', '隐', '藏', '在', '屋', '子', '，', '很', '方', '面', '也', '很', '干', '净', '呀', '不', '过', '店', '内', '的', '位', '置', '不', '多', '，', '外', '面', '的', '话', '不', '会', '很', '晒', '，', '因', '为', '外', '面', '有', '好', '多', '树', '之', '类', '，', '但', '是', '蚊', '子', '就', '不', '少', '了', '6', '6', '6', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 
    188 188
    ['第', '一', '次', '来', '吃', '越', '南', '菜', '感', '觉', '万', '象', '城', '这', '家', '店', '气', '氛', '还', '不', '错', '环', '境', '优', '雅', '由', '于', '自', '己', '一', '个', '人', '吃', '就', '点', '了', '两', '个', '菜', '式', '，', '晚', '上', '去', '吃', '并', '不', '是', '太', '饿', '\xa0', '点', '了', '一', '个', '招', '牌', '菜', '安', '南', '春', '卷', '\xa0', '春', '卷', '里', '面', '夹', '杂', '着', '猪', '肉', '和', '粉', '条', '\xa0', '其', '实', '并', '没', '有', '想', '象', '中', '的', '那', '么', '好', '吃', '一', '共', '5', '块', '春', '卷', '价', '格', '倒', '是', '不', '菲', '没', '尝', '过', '的', '亲', '可', '以', '一', '试', '椰', '青', '果', '冻', '\xa0', '比', '较', '爽', '口', '\xa0', '果', '冻', '切', '割', '的', '很', '有', '艺', '术', '感', '上', '面', '还', '用', '薄', '荷', '叶', '点', '缀', '\xa0', '让', '人', '很', '有', '品', '尝', '的', '欲', '望', '椰', '冻', '里', '面', '有', '香', '浓', '的', '椰', '奶', '\xa0', '冻', '里', '还', '夹', '杂', '着', '椰', '子', '肉', '\xa0', '个', '人', '感', '觉', '作', '为', '饭', '后', '甜', '点', '品', '尝', '下', '还', '是', '不', '错', '的'] ['O', 'O', 'O', 'O', 'O', 'O', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 南菜安
    225 225
    ['安', '薇', '塔', '英', '国', '茶', '屋', '，', '位', '于', '和', '义', '大', '道', '购', '物', '中', '心', 'B', '区', '2', '楼', '，', '扶', '梯', '上', '去', '就', '能', '看', '见', '，', '整', '个', '店', '以', '蓝', '色', '调', '为', '主', '，', '一', '进', '去', '就', '是', '各', '种', '茶', '罐', '，', '看', '起', '来', '相', '当', '有', '感', '觉', '，', '点', '了', '一', '个', '单', '人', '套', '餐', '，', '包', '含', '一', '壶', '茶', '和', '一', '个', '两', '层', '的', '点', '心', '价', '，', '茶', '要', '了', '阿', '萨', '姆', '奶', '茶', '，', '听', '说', '他', '们', '家', '的', '茶', '有', '点', '甜', '，', '所', '以', '就', '要', '了', '半', '糖', '，', '其', '实', '可', '以', '让', '服', '务', '员', '直', '接', '把', '糖', '拿', '来', '自', '己', '按', '口', '味', '加', '更', '佳', '，', '茶', '的', '味', '道', '还', '可', '以', '，', '点', '心', '有', '两', '层', '，', '一', '层', '是', '三', '明', '治', '司', '康', '饼', '黄', '油', '等', '，', '还', '有', '一', '层', '是', '甜', '点', '，', '有', '曲', '奇', '饼', '，', '水', '果', '，', '布', '朗', '尼', '和', '一', '个', '布', '丁', '，', '也', '算', '是', '英', '国', '下', '午', '茶', '的', '标', '配', '了', '，', '点', '心', '的', '味', '道', '比', '较', '一', '般', '，', '闺', '蜜', '聚', '会', '还', '是', '可', '以', '小', '坐', '一', '下', '的', '。'] ['B', 'I', 'I', 'O', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 安薇塔国茶屋
    82 82
    ['这', '家', '旋', '疯', '年', '糕', '时', '尚', '韩', '料', '理', '的', '装', '修', '确', '实', '很', '新', '颖', '，', '以', '浪', '漫', '的', '蓝', '色', '为', '主', '色', '调', '，', '画', '满', '了', '可', '爱', '的', 'Q', 'Q', '年', '糕', '漫', '画', '，', '很', '有', '韩', '国', '风', '味', '，', '点', '了', '几', '个', '特', '色', '菜', '，', '推', '荐', '虾', '仁', '炒', '饭', '和', '五', '花', '肉', '炖', '泡', '菜', '锅', '，', '芝', '士', '味', '道', '也', '不', '错', '！'] ['O', 'O', 'B', 'I', 'I', 'I', 'O', 'O', 'I', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 旋疯年糕韩料理
    40 40
    ['蛋', '糕', '味', '道', '也', '是', '极', '好', '的', '，', '欧', '美', '香', '奶', '油', '细', '腻', '软', '滑', '，', '甜', '度', '刚', '好', '，', '不', '肥', '不', '腻', '，', '真', '心', '可', '以', '给', '一', '个', '好', '评', '！'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 
    39 39
    ['今', '天', '在', '嘉', '兴', '学', '院', '逛', '，', '之', '前', '这', '里', '买', '过', '玉', '米', '烙', '，', '味', '道', '还', '可', '以', '，', '又', '出', '来', '了', '几', '个', '新', '的', '品', '种', '的', '小', '吃', '。'] ['O', 'O', 'O', 'B', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O'] 嘉玉米烙
    25 25
    ['想', '多', '活', '几', '年', '的', '同', '志', '们', '千', '万', '不', '要', '吃', '百', '味', '鲜', '自', '助', '火', '锅', '昆', '明', '店', '。'] ['O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'O', 'B', 'I', 'I', 'O', 'O', 'I', 'I', 'O', 'O', 'O', 'O'] 百味鲜火锅
    42 42


[返回](/PaddleNLP_Homework)