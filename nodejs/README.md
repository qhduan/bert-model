
首先下载数据集合

```bash
$ curl --output train.json https://code.aliyun.com/qhduan/dataset/raw/88b3182c9f9d6185935d4484dfefefc23f50eaa7/LCQMC/train.json
$ curl --output dev.json https://code.aliyun.com/qhduan/dataset/raw/88b3182c9f9d6185935d4484dfefefc23f50eaa7/LCQMC/dev.json
```

下载词表

```bash
$ curl --output vocab.txt https://code.aliyun.com/qhduan/zh-bert/raw/0fb1d96ec2133fe25e66bee12fe387cbe1e52938/vocab.txt
```

下载模型并解压到`bert`目录

```bash
$ mkdir -p bert
$ cd bert
$ curl --output bert.tar.gz https://code.aliyun.com/qhduan/bert/raw/0a53cbdce78a16053ab0034cefe21caa37bdf128/albert_base.tar.gz
$ tar xvzf bert.tar.gz
$ cd ..
```

安装node依赖

```bash
$ npm i
```

运行训练

```bash
$ node train.js
```
