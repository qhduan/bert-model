#!/bin/bash

set -e

mkdir -p ./hubs

# for i in {1..11}; do
#         export NUM_HIDDEN_LAYERS=$i
#         echo $NUM_HIDDEN_LAYERS

#         tf1env/bin/python export_to_tfhub.py \
#                 --bert_directory=./pretrained/chinese_roberta_wwm_ext_L-12_H-768_A-12 \
#                 --export_path=./hubs/chinese_roberta_wwm_ext_L-12_H-768_A-12-$i

#         tf1env/bin/python export_to_tfhub.py \
#                 --bert_directory=./pretrained/chinese_wwm_ext_L-12_H-768_A-12 \
#                 --export_path=./hubs/chinese_wwm_ext_L-12_H-768_A-12-$i

#         tf1env/bin/python export_to_tfhub.py \
#                 --bert_directory=./pretrained/chinese_L-12_H-768_A-12 \
#                 --export_path=./hubs/chinese_L-12_H-768_A-12-$i

#         unset NUM_HIDDEN_LAYERS
# done

for i in {1..23}; do
        export NUM_HIDDEN_LAYERS=$i
        echo $NUM_HIDDEN_LAYERS

        tf1env/bin/python export_to_tfhub.py \
	--bert_directory=./pretrained/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16 \
	--export_path=./hubs/chinese_roberta_wwm_large_ext_L-24_H-1024_A-16-$i

        unset NUM_HIDDEN_LAYERS
done



