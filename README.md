# 简单中文BERT构建

## 下载并转化TF1训练的BERT模型

模型目录应该有以下文件：

- bert_config.json
- bert_model.ckpt.meta
- bert_model.ckpt.data-00000-of-00001
- vocab.txt
- bert_model.ckpt.index

构建一个TF1的虚拟环境，并转化模型到tensorflow-hub类型

```bash
virtualenv tf1env
tf1env/bin/pip install tensorflow==1.15.0 tensorflow-hub
./convert.sh
```

## 保存TOKENIZER

运行`save_bert_tokenizer.ipynb`

## 合并TOKENIZER和转化好的HUB模型

运行`save_bert.ipynb`

## 打包

运行`./tar.sh`

## 使用

参考： https://colab.research.google.com/drive/1KkjPVn1s6_tSznhox5RxuKF9Igm8VAbE

原模型下载： https://github.com/ymcui/Chinese-BERT-wwm

地址：

```
https://github.com/qhduan/bert-model/releases/download/1.0/bert_wwm_zh_pool.tar.gz
https://github.com/qhduan/bert-model/releases/download/1.0/bert_wwm_zh_seq.tar.gz
https://github.com/qhduan/bert-model/releases/download/1.0/bert_zh_pool.tar.gz
https://github.com/qhduan/bert-model/releases/download/1.0/bert_zh_seq.tar.gz
https://github.com/qhduan/bert-model/releases/download/1.0/roberta_wwm_zh_pool.tar.gz
https://github.com/qhduan/bert-model/releases/download/1.0/roberta_wwm_zh_seq.tar.gz
```