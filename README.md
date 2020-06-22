# 简单中文BERT

## 使用

- 最好配合huggingface/tokenizers 使用（当然中文简单按字分词也差不太多）
- 配合hub食用更佳，当然下载下来只用tf.keras.models.load_model也可以

## 简单的例子

下面是一个简单的文本分类例子

安装依赖

```
$ pip install tensorflow tokenizers tensorflow-hub
```

下载测试数据集与词典

```
$ wget https://code.aliyun.com/qhduan/dataset/raw/88b3182c9f9d6185935d4484dfefefc23f50eaa7/LCQMC/train.json
$ wget https://code.aliyun.com/qhduan/dataset/raw/88b3182c9f9d6185935d4484dfefefc23f50eaa7/LCQMC/dev.json
$ wget https://code.aliyun.com/qhduan/zh-bert/raw/0fb1d96ec2133fe25e66bee12fe387cbe1e52938/vocab.txt
```

代码

```
import os
import json

os.environ['TFHUB_DOWNLOAD_PROGRESS'] = '1'

from tqdm import tqdm
import tensorflow as tf
import tensorflow_hub as hub
import numpy as np
from tokenizers import BertWordPieceTokenizer

# 处理数据集
train = [json.loads(x) for x in open('train.json')]
dev = [json.loads(x) for x in open('dev.json')]
tokenizer = BertWordPieceTokenizer("vocab.txt")


def compose_data(data, batch_size=32):
    X = [
        (tokenizer.encode(x.get('sentence1')).tokens + tokenizer.encode(x.get('sentence2')).tokens[1:])[:512]
        for x in tqdm(data)
    ]
    Y = [int(x.get('label')) for x in data]
    X = tf.ragged.constant(X, tf.string)
    Y = tf.constant(Y, tf.int32)

    @tf.autograph.experimental.do_not_convert
    def _to_tensor(x, y):
        return x.to_tensor(), y

    return tf.data.Dataset.zip((
        tf.data.Dataset.from_tensor_slices(X),
        tf.data.Dataset.from_tensor_slices(Y)
    )).batch(batch_size).map(_to_tensor)


data_train = compose_data(train)
data_dev = compose_data(dev)

# 构建模型

bert = hub.KerasLayer(
    'https://code.aliyun.com/qhduan/zh-roberta-wwm/raw/2c0d7fd709e4719a9ab2ca297f51b24e20586dbe/zh-roberta-wwm-L12.tar.gz',
    output_key='pooled_output',
    trainable=True)


# 用tf.keras.Sequential的话，可能导致模型无法load
inputs = tf.keras.layers.Input(shape=(None,), dtype=tf.string)
m = inputs
m = bert(m)
m = tf.keras.layers.Masking()(m)
m = tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(128))(m)
m = tf.keras.layers.Dense(2, activation='softmax')(m)
model = tf.keras.Model(inputs=inputs, outputs=m)

# 编译训练

model.compile(
    loss=tf.keras.losses.SparseCategoricalCrossentropy(),
    optimizer=tf.keras.optimizers.Adam(1e-3),
    metrics=['acc']
)

model.fit(data_train, epochs=3, validation_data=data_dev)

model.evaluate(data_dev)

```


## 模型下载地址

词表，所有的中文BERT几乎都是基于谷歌最开始发布的词表

```
https://code.aliyun.com/qhduan/zh-bert/raw/0fb1d96ec2133fe25e66bee12fe387cbe1e52938/vocab.txt
```

谷歌最开始的预训练中文，12层，文件中的LX，代表包含几层，L12就是完整模型

```
https://code.aliyun.com/qhduan/zh-bert/raw/0fb1d96ec2133fe25e66bee12fe387cbe1e52938/zh-bert-L1.tar.gz
https://code.aliyun.com/qhduan/zh-bert/raw/0fb1d96ec2133fe25e66bee12fe387cbe1e52938/zh-bert-L3.tar.gz
https://code.aliyun.com/qhduan/zh-bert/raw/0fb1d96ec2133fe25e66bee12fe387cbe1e52938/zh-bert-L6.tar.gz
https://code.aliyun.com/qhduan/zh-bert/raw/0fb1d96ec2133fe25e66bee12fe387cbe1e52938/zh-bert-L9.tar.gz
https://code.aliyun.com/qhduan/zh-bert/raw/0fb1d96ec2133fe25e66bee12fe387cbe1e52938/zh-bert-L12.tar.gz
```

[ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm/)
发布的预训练模型，12层，文件中的LX，代表包含几层，L12就是完整模型

```
https://code.aliyun.com/qhduan/zh-bert-wwm/raw/51f499fb7df00cf4d0f283cefd0a7cb34316b866/zh-bert-wwm-L1.tar.gz
https://code.aliyun.com/qhduan/zh-bert-wwm/raw/51f499fb7df00cf4d0f283cefd0a7cb34316b866/zh-bert-wwm-L3.tar.gz
https://code.aliyun.com/qhduan/zh-bert-wwm/raw/51f499fb7df00cf4d0f283cefd0a7cb34316b866/zh-bert-wwm-L6.tar.gz
https://code.aliyun.com/qhduan/zh-bert-wwm/raw/51f499fb7df00cf4d0f283cefd0a7cb34316b866/zh-bert-wwm-L9.tar.gz
https://code.aliyun.com/qhduan/zh-bert-wwm/raw/51f499fb7df00cf4d0f283cefd0a7cb34316b866/zh-bert-wwm-L12.tar.gz
```

[ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm/)
发布的预训练模型，12层，文件中的LX，代表包含几层，L12就是完整模型

```
https://code.aliyun.com/qhduan/zh-roberta-wwm/raw/2c0d7fd709e4719a9ab2ca297f51b24e20586dbe/zh-roberta-wwm-L1.tar.gz
https://code.aliyun.com/qhduan/zh-roberta-wwm/raw/2c0d7fd709e4719a9ab2ca297f51b24e20586dbe/zh-roberta-wwm-L3.tar.gz
https://code.aliyun.com/qhduan/zh-roberta-wwm/raw/2c0d7fd709e4719a9ab2ca297f51b24e20586dbe/zh-roberta-wwm-L6.tar.gz
https://code.aliyun.com/qhduan/zh-roberta-wwm/raw/2c0d7fd709e4719a9ab2ca297f51b24e20586dbe/zh-roberta-wwm-L9.tar.gz
https://code.aliyun.com/qhduan/zh-roberta-wwm/raw/2c0d7fd709e4719a9ab2ca297f51b24e20586dbe/zh-roberta-wwm-L12.tar.gz
```

[ymcui/Chinese-BERT-wwm](https://github.com/ymcui/Chinese-BERT-wwm/)
发布的预训练模型，24层，文件中的LX，代表包含几层，L12就是完整模型

```
https://code.aliyun.com/qhduan/zh-roberta-wwm-large/raw/47e2d9f33e0fa5f06eeb4a90e68c79c4d3b03ce7/zh-roberta-wwm-large-L12.tar.gz
https://code.aliyun.com/qhduan/zh-roberta-wwm-large/raw/47e2d9f33e0fa5f06eeb4a90e68c79c4d3b03ce7/zh-roberta-wwm-large-L24.tar.gz
```
