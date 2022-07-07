#!/usr/bin/env bash

# train
#python inference.py \
# --output_dir ./testv2 \
# --from_pretrained ./pytorch_model_8.bin \
# --lmdb_file ../testv1/item_train_image_feature.lmdb \
# --caption_path ../../Data/item_train_info.jsonl \
# --config_file ./config/capture.json \
# --bert_model bert-base-chinese \
# --predict_feature \
# --train_batch_size 16 \
# --max_seq_length 36

# valid
python inference.py \
 --output_dir ./testv2 \
 --feature_file ./testv2/item_valid_features.tsv \
 --from_pretrained ./pytorch_model_8.bin \
 --lmdb_file ../testv1/item_valid_image_feature.lmdb \
 --caption_path ../../Data/item_valid_info.jsonl \
 --config_file ./config/capture.json \
 --bert_model bert-base-chinese \
 --predict_feature \
 --train_batch_size 16 \
 --max_seq_length 36