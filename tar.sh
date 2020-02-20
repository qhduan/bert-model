#!/bin/bash

mkdir -p zip

cd output

cd roberta_wwm_zh_seq
GZIP=-1 tar czf ../../zip/roberta_wwm_zh_seq.tar.gz ./
cd ..

cd roberta_wwm_zh_pool
GZIP=-1 tar czf ../../zip/roberta_wwm_zh_pool.tar.gz ./
cd ..

cd bert_wwm_zh_pool
GZIP=-1 tar czf ../../zip/bert_wwm_zh_pool.tar.gz ./
cd ..

cd bert_wwm_zh_seq
GZIP=-1 tar czf ../../zip/bert_wwm_zh_seq.tar.gz ./
cd ..

cd bert_zh_pool
GZIP=-1 tar czf ../../zip/bert_zh_pool.tar.gz ./
cd ..

cd bert_zh_seq
GZIP=-1 tar czf ../../zip/bert_zh_seq.tar.gz ./
cd ..

