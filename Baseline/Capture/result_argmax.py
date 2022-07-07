import os
import csv
import json
import numpy as np
import pandas as pd
from typing import List
from utils.util import *
from numpy.core.fromnumeric import argmax
from sklearn.metrics import precision_recall_curve


# 计算向量相似度
def compute(item_emb_1: List[float], item_emb_2 :List[float]) -> float:
	s = sum([a*b for a,b in zip(item_emb_1, item_emb_2)])
	return s


# 返回最优阈值
def getThreshold(y_true, y_score):
    precision, recall, thresholds = precision_recall_curve(y_true, y_score)
    # target = precision+recall
    target = (2*precision*recall)/(precision+recall)
    index = argmax(target)
    return thresholds[index]


if __name__ == '__main__':

    # 1、加载每个item的vector
    # itemid2emb = {}
    # with open('./testv2/item_features.tsv') as tsv_in_file:
    #     reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=['item_id', 'features'])
    #     for item in reader:
    #         itemid2emb[str(item['item_id'])] = '[' + item['features'] + ']'
    # pickleFile("./cache/itemid2emb_train.pkl", itemid2emb)
    itemid2emb = unPickleFile("./cache/itemid2emb_train.pkl")
    print("1、加载每个item的vector done...")

    # 2、加载每个item的属性
    train_info = []
    with open('../../Data/item_train_info.jsonl', encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            train_info.append(json.loads(line.strip()))
    df_train_info = pd.DataFrame(train_info)
    print("2、加载每个item的属性 done...")

    # 3、统计各属性出现频次
    # feature_dict = {}
    # for key, group in df_train_info.groupby(["sku_pvs"]):
    #     feature_dict[key] = group.shape[0]
    # pv_dict = {}
    # for key, value in sorted(feature_dict.items(), key=lambda x: x[1], reverse=True):
    #     print(key, value)
    #     for property in key.split(";"):
    #         property_name = property.split(":")[0].replace("#", "")
    #         if property_name in pv_dict:
    #             pv_dict[property_name] += 1
    #         else:
    #             pv_dict[property_name] = 1
    # for key, value in sorted(pv_dict.items(), key=lambda x: x[1], reverse=True):
    #     print(key, value)
    # print("3、统计各属性出现频次 done...")

    # 4、根据训练集确定阈值 (all-57741, in-1113, out-56628s)
    y_true = []
    y_score = []
    with open('../../Data/item_train_pair.jsonl', encoding='utf-8', mode='r') as f:
        cnt_all = 0
        cnt_in = 0
        cnt_out = 0
        for line in f.readlines():
            cnt_all += 1
            line = line.strip()
            item = json.loads(line)
            src_item_id = item['src_item_id']
            tgt_item_id = item['tgt_item_id']
            item_label = item['item_label']
            try:
                # 计算两个item的得分
                src_item_emb = itemid2emb[src_item_id]
                tgt_item_emb = itemid2emb[tgt_item_id]
                # score = compute([float(vec) for vec in src_item_emb[1:-1].split(",")], [float(vec) for vec in tgt_item_emb[1:-1].split(",")])
                # y_true.append(float(item_label))
                # y_score.append(float(score))
                cnt_in += 1
            except:
                cnt_out += 1
                continue
    # threshold = getThreshold(y_true, y_score)
    print("训练集概况 - ", cnt_all, cnt_in, cnt_out)
    print("4、根据训练集确定阈值 done...")

    # # 5、确定测试集的阈值
    # f_out = open('./result/XXXX_result.jsonl', encoding='utf-8', mode='w')
    # with open('../../Data/item_valid_pair.jsonl', encoding='utf-8', mode='r') as f:
    #     cnt = 0
    #     for line in f.readlines():
    #         cnt += 1
    #         line = line.strip()
    #         item = json.loads(line)
    #         src_item_id = item['src_item_id']
    #         tgt_item_id = item['tgt_item_id']
    #         src_item_emb = itemid2emb[src_item_id]
    #         tgt_item_emb = itemid2emb[tgt_item_id]
    #         score = compute([float(vec) for vec in src_item_emb[1:-1].split(",")], [float(vec) for vec in tgt_item_emb[1:-1].split(",")])
    #         out_item = {
    #             "src_item_id": src_item_id,
    #             "src_item_emb": src_item_emb,
    #             "tgt_item_id": tgt_item_id,
    #             "tgt_item_emb": tgt_item_emb,
    #             "threshold": threshold
    #         }
    #         f_out.write(json.dumps(out_item) + '\n')
    # f_out.close()





















