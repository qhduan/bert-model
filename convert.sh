#!/bin/bash

set -e

mkdir -p ./hub

tf1env/bin/python export_to_tfhub.py \
        --bert_directory=./pretrained/chinese_roberta_wwm_ext_L-12_H-768_A-12 \
        --export_path=./hub/chinese_roberta_wwm_ext_L-12_H-768_A-12

tf1env/bin/python export_to_tfhub.py \
        --bert_directory=./pretrained/chinese_wwm_ext_L-12_H-768_A-12 \
        --export_path=./hub/chinese_wwm_ext_L-12_H-768_A-12

tf1env/bin/python export_to_tfhub.py \
	--bert_directory=./pretrained/chinese_L-12_H-768_A-12 \
	--export_path=./hub/chinese_L-12_H-768_A-12

tf1env/bin/python export_to_tfhub.py \
	--bert_directory=./pretrained/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16 \
	--export_path=./hub/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16

