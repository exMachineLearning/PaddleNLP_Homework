# [千言数据集：阅读理解](https://aistudio.baidu.com/aistudio/competition/detail/49)之DuReader-yesno

## 1 介绍
本项目参考自[官方的example](https://gitee.com/paddlepaddle/PaddleNLP/tree/develop/examples/machine_reading_comprehension/DuReader-yesno)，在此基础上增加了注释和实验。
DuReader-yesno数据集是一个以观点极性判断为目标任务的数据集，通过引入该数据集，可以弥补抽取类数据集的不足，从而更好地评价模型的自然语言理解能力，该数据集的任务定义如下：
对于一个给定的问题q、一系列相关文档D=d1, d2, …, dn，以及人工抽取答案段落摘要a，要求参评系统自动对问题q、候选文档D以及答案段落摘要a进行分析，输出每个答案段落摘要所表述的是非观点极性。其中，极性分为三类 {Yes, No, Depends}。其中：
- Yes：肯定观点，肯定观点指的是答案给出了较为明确的肯定态度。有客观事实的从客观事实的角度出发，主观态度类的从答案的整体态度来判断。
- No：否定观点，否定观点通常指的是答案较为明确的给出了与问题相反的态度。
- Depends：无法确定/分情况，主要指的是事情本身存在多种情况，不同情况下对应的观点不一致；或者答案本身对问题表示不确定，要具体具体情况才能判断。

## 2 准备和修改相关代码

### 2.1 准备相关代码


```python
# 更新paddlenlp
!pip install --upgrade paddlenlp > /dev/null
# 新建文件夹code
!mkdir ~/code/ > /dev/null
```


```python
# 下载example文件并将相关示例代码copy到code目录下
!git clone https://gitee.com/paddlepaddle/PaddleNLP.git ~/data 
!cp -r -n ~/data/examples/machine_reading_comprehension/DuReader-yesno/. ~/code/
# 进入code目录
%cd ~/code
```

### 2.2 修改相关代码进行训练

- 修改代码~/code/run_du.py，使其支持保存最佳结果，详细修改内容见源码(修改处有注释)。
- 新建~/code/train.sh，里面设置参数，然后进行训练


```python
#训练开始
%cd ~/code
!bash train.sh
```

### 2.3 增加代码进行测试

- 修改代码~/code/args.py，增加参数`test_model_path`用来保存训练好的参数路径
- 新建~/code/test.sh，里面设置参数，然后进行测试


```python
# 进入code目录
%cd ~/code
#测试开始
!bash test.sh
```
