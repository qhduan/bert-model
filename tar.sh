#!/bin/bash

mkdir -p zip

cd output

cd roberta_wwm_zh
GZIP=-1 tar czf ../../zip/roberta_wwm_zh.tar.gz ./
cd ..

cd bert_wwm_zh
GZIP=-1 tar czf ../../zip/bert_wwm_zh.tar.gz ./
cd ..

cd bert_zh
GZIP=-1 tar czf ../../zip/bert_zh.tar.gz ./
cd ..

cd roberta_wwm_large_zh
GZIP=-1 tar czf ../../zip/roberta_wwm_large_zh.tar.gz ./
cd ..

