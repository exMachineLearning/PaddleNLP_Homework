# [千言数据集：阅读理解](https://aistudio.baidu.com/aistudio/competition/detail/49)之dureader_robust

## 1 介绍
本项目参考自[官方的example](https://gitee.com/paddlepaddle/PaddleNLP/tree/develop/examples/machine_reading_comprehension/DuReader-robust)，在此基础上增加了注释和实验。
DuReaderrobust数据集是单篇章、抽取式阅读理解数据集，具体的任务定义为：
对于一个给定的问题q和一个篇章p，参赛系统需要根据篇章内容，给出该问题的答案a。数据集中的每个样本，是一个三元组<q, p, a>，例如：
- 问题 q: 乔丹打了多少个赛季
- 篇章 p: 迈克尔.乔丹在NBA打了15个赛季。他在84年进入nba，期间在1993年10月6日第一次退役改打棒球，95年3月18日重新回归，在99年1月13日第二次退役，后于2001年10月31日复出，在03年最终退役…
- 参考答案 (a): [‘15个’,‘15个赛季’]

## 2 准备和修改相关代码

### 2.1 准备相关代码


```python
# 更新paddlenlp
!pip install --upgrade paddlenlp > /dev/null
# 新建文件夹code
!mkdir ~/code/ > /dev/null
```

    mkdir: cannot create directory ‘/home/aistudio/code/’: File exists



```python
# 下载example文件并将相关示例代码copy到code目录下
!git clone https://gitee.com/paddlepaddle/PaddleNLP.git ~/data 
!cp -r -n ~/data/examples/machine_reading_comprehension/DuReader-robust/. ~/code/
# 进入code目录
%cd ~/code
```

    Cloning into '/home/aistudio/data'...
    remote: Enumerating objects: 7736, done.[K
    remote: Total 7736 (delta 0), reused 0 (delta 0), pack-reused 7736[K
    Receiving objects: 100% (7736/7736), 54.66 MiB | 7.87 MiB/s, done.
    Resolving deltas: 100% (4760/4760), done.
    Checking connectivity... done.
    /home/aistudio/code


### 2.2 修改相关代码进行训练

- 修改代码~/code/run_du.py，使其支持保存最佳结果，详细修改内容见源码(修改处有注释)。
- 新建~/code/train.sh，里面设置参数，然后进行训练


```python
# 进入code目录
%cd ~/code
#训练开始
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

    /home/aistudio/code
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
    training_script: run_du.py
    training_script_args: ['--task_name', 'dureader_robust', '--model_type', 'ernie_gram', '--model_name_or_path', 'ernie-gram-zh', '--test_model_path', './ckp/dureader-robust/best_0.889560.pdparams', '--max_seq_length', '384', '--batch_size', '12', '--learning_rate', '3e-5', '--num_train_epochs', '5', '--logging_steps', '10', '--valid_steps', '200', '--warmup_proportion', '0.1', '--weight_decay', '0.01', '--output_dir', './ckp/dureader-robust/', '--do_predict', '--device', 'gpu']
    worker_num: None
    workers: 
    ------------------------------------------------
    WARNING 2021-07-14 18:28:51,347 launch.py:357] Not found distinct arguments and compiled with cuda or xpu. Default use collective mode
    launch train in GPU mode!
    INFO 2021-07-14 18:28:51,349 launch_utils.py:510] Local start 1 processes. First process distributed environment info (Only For Debug): 
        +=======================================================================================+
        |                        Distributed Envs                      Value                    |
        +---------------------------------------------------------------------------------------+
        |                       PADDLE_TRAINER_ID                        0                      |
        |                 PADDLE_CURRENT_ENDPOINT                 127.0.0.1:39019               |
        |                     PADDLE_TRAINERS_NUM                        1                      |
        |                PADDLE_TRAINER_ENDPOINTS                 127.0.0.1:39019               |
        |                     PADDLE_RANK_IN_NODE                        0                      |
        |                 PADDLE_LOCAL_DEVICE_IDS                        0                      |
        |                 PADDLE_WORLD_DEVICE_IDS                        0                      |
        |                     FLAGS_selected_gpus                        0                      |
        |             FLAGS_selected_accelerators                        0                      |
        +=======================================================================================+
    
    INFO 2021-07-14 18:28:51,349 launch_utils.py:514] details abouts PADDLE_TRAINER_ENDPOINTS can be found in log/endpoints.log, and detail running logs maybe found in log/workerlog.0
    launch proc_id:12491 idx:0
    [2021-07-14 18:28:52,788] [    INFO] - Found /home/aistudio/.paddlenlp/models/ernie-gram-zh/vocab.txt
    [2021-07-14 18:28:52,800] [    INFO] - Already cached /home/aistudio/.paddlenlp/models/ernie-gram-zh/ernie_gram_zh.pdparams
    W0714 18:28:52.801362 12491 device_context.cc:404] Please NOTE: device: 0, GPU Compute Capability: 7.0, Driver API Version: 10.1, Runtime API Version: 10.1
    W0714 18:28:52.804977 12491 device_context.cc:422] device: 0, cuDNN Version: 7.6.
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.weight. classifier.weight is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    /opt/conda/envs/python35-paddle120-env/lib/python3.7/site-packages/paddle/fluid/dygraph/layers.py:1297: UserWarning: Skip loading for classifier.bias. classifier.bias is not found in the provided dict.
      warnings.warn(("Skip loading for {}. ".format(key) + str(err)))
    Loaded parameters from ./ckp/dureader-robust/best_0.889560.pdparams
    
      0%|          | 0/20038 [00:00<?, ?it/s]
     32%|███▏      | 6426/20038 [00:00<00:00, 64256.54it/s]
     58%|█████▊    | 11568/20038 [00:00<00:00, 53066.41it/s]
     88%|████████▊ | 17568/20038 [00:00<00:00, 54971.04it/s]
    100%|██████████| 20038/20038 [00:00<00:00, 53731.53it/s]
    Processing example: 1000
    time per 1000: 19.861122369766235
    Processing example: 2000
    time per 1000: 18.7887282371521
    Processing example: 3000
    time per 1000: 18.533133268356323
    Processing example: 4000
    time per 1000: 18.655021905899048
    Processing example: 5000
    time per 1000: 19.41499638557434
    Processing example: 6000
    time per 1000: 18.56358242034912
    Processing example: 7000
    time per 1000: 18.46160054206848
    Processing example: 8000
    time per 1000: 18.02328634262085
    Processing example: 9000
    time per 1000: 18.121554136276245
    Processing example: 10000
    time per 1000: 18.77463173866272
    Processing example: 11000
    time per 1000: 18.854174852371216
    Processing example: 12000
    time per 1000: 18.3427631855011
    Processing example: 13000
    time per 1000: 18.139728546142578
    Processing example: 14000
    time per 1000: 18.13940954208374
    Processing example: 15000
    time per 1000: 18.333805322647095
    Processing example: 16000
    time per 1000: 18.27450466156006
    Processing example: 17000
    time per 1000: 18.2143452167511
    Processing example: 18000
    time per 1000: 18.3362033367157
    Processing example: 19000
    time per 1000: 17.52031445503235
    Processing example: 20000
    time per 1000: 18.308919668197632
    Processing example: 21000
    time per 1000: 17.93433904647827
    Processing example: 22000
    time per 1000: 18.61131501197815
    Processing example: 23000
    time per 1000: 17.81761932373047
    Processing example: 24000
    time per 1000: 18.169885396957397
    Processing example: 25000
    time per 1000: 18.06153106689453
    Processing example: 26000
    time per 1000: 18.65974760055542
    Processing example: 27000
    time per 1000: 18.549249172210693
    Processing example: 28000
    time per 1000: 18.370241165161133
    Processing example: 29000
    time per 1000: 17.514158964157104
    Processing example: 30000
    time per 1000: 18.186832666397095
    Processing example: 31000
    time per 1000: 18.02071237564087
    Processing example: 32000
    time per 1000: 18.077165365219116
    Processing example: 33000
    time per 1000: 17.92798161506653
    Processing example: 34000
    time per 1000: 18.24454164505005
    Processing example: 35000
    time per 1000: 17.971704721450806
    Processing example: 36000
    time per 1000: 18.037253618240356
    Processing example: 37000
    time per 1000: 17.7172589302063
    Processing example: 38000
    time per 1000: 17.67001986503601
    Processing example: 39000
    time per 1000: 18.9000563621521
    Processing example: 40000
    time per 1000: 18.156458139419556
    Processing example: 41000
    time per 1000: 17.52651047706604
    Processing example: 42000
    time per 1000: 18.486640453338623
    Processing example: 43000
    time per 1000: 18.064385890960693
    Processing example: 44000
    time per 1000: 17.500937700271606
    Processing example: 45000
    time per 1000: 18.119248390197754
    Processing example: 46000
    time per 1000: 17.99043583869934
    Processing example: 47000
    time per 1000: 17.735183000564575
    Processing example: 48000
    time per 1000: 18.903116703033447
    Processing example: 49000
    time per 1000: 18.059256076812744
    Processing example: 50000
    time per 1000: 18.150585889816284
    Processing example: 51000
    time per 1000: 18.13495898246765
    Processing example: 52000
    time per 1000: 18.12674856185913
    Processing example: 53000
    time per 1000: 17.845890283584595
    Processing example: 54000
    time per 1000: 18.334473133087158
    Processing example: 55000
    time per 1000: 18.027803659439087
    Processing example: 56000
    time per 1000: 17.905311107635498
    Processing example: 57000
    time per 1000: 17.747634649276733
    Processing example: 58000
    time per 1000: 18.281975507736206
    Processing example: 59000
    time per 1000: 17.63224148750305
    Processing example: 60000
    time per 1000: 18.4889976978302
    Processing example: 61000
    time per 1000: 17.537192583084106
    Processing example: 62000
    time per 1000: 18.301725387573242
    Processing example: 63000
    time per 1000: 17.706255435943604
    Processing example: 64000
    time per 1000: 18.26703381538391
    Processing example: 65000
    time per 1000: 18.21987748146057
    Processing example: 66000
    time per 1000: 18.159775733947754
    Processing example: 67000
    time per 1000: 18.190679788589478
    Processing example: 68000
    time per 1000: 17.921314477920532
    Processing example: 69000
    time per 1000: 18.660756587982178
    Processing example: 70000
    time per 1000: 18.100632667541504
    Processing example: 71000
    time per 1000: 17.699832677841187
    Processing example: 72000
    time per 1000: 18.08768057823181
    Processing example: 73000
    time per 1000: 18.130647897720337
    Processing example: 74000
    time per 1000: 17.599804401397705
    Processing example: 75000
    time per 1000: 17.33307909965515
    Processing example: 76000
    time per 1000: 18.606958389282227
    Processing example: 77000
    time per 1000: 17.928603887557983
    Processing example: 78000
    time per 1000: 18.561939239501953
    Processing example: 79000
    time per 1000: 17.997174739837646
    Processing example: 80000
    time per 1000: 17.839795112609863
    Processing example: 81000
    time per 1000: 18.1271710395813
    Processing example: 82000
    time per 1000: 18.352195501327515
    INFO 2021-07-14 18:58:41,044 launch.py:266] Local processes completed.

