import os
import csv
import json
import warnings
import numpy as np
import pandas as pd
from typing import List
from utils.util import *
from sklearn.metrics import f1_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from numpy.core.fromnumeric import argmax
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
warnings.filterwarnings("ignore")


# 计算向量相似度
def compute(item_emb_1: List[float], item_emb_2 :List[float]) -> float:
	s = sum([a*b for a,b in zip(item_emb_1, item_emb_2)])
	return s


# 计算实际值和预测值的结果
def getEffect(y_true, y_pred):
    # 计算auc
    auc = accuracy_score(y_true, y_pred)
    # 计算准确率
    pre = precision_score(y_true, y_pred)
    # 计算召回率
    rec = recall_score(y_true, y_pred)
    # 计算f1值
    f1 = f1_score(y_true, y_pred)
    return auc, pre, rec, f1


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

    # 4、logis回归 (all-57741, in-1113, out-56628s)
    X = []
    Y = []
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
                score = compute([float(vec) for vec in src_item_emb[1:-1].split(",")], [float(vec) for vec in tgt_item_emb[1:-1].split(",")])
                X.append([float(score)])
                Y.append(int(item_label))
                cnt_in += 1
            except:
                cnt_out += 1
                continue
    X_train, X_test, Y_train, Y_test = train_test_split(np.array(X), np.array(Y), test_size=0.2)

    # 自动调整参数
    params = {
        'C': [0.0001, 1, 100, 1000],
        'max_iter': [10000, 20000, 30000, 40000, 100000],
        'class_weight': ['balanced', None],
        'solver': ['liblinear', 'sag', 'lbfgs', 'newton-cg']
    }
    lr = LogisticRegression()
    clf = GridSearchCV(lr, param_grid=params, scoring='f1')
    clf.fit(X_train, Y_train)
    print(clf.best_params_)

    # 加载最优参数
    classifier = LogisticRegression(**clf.best_params_) # 加载最优参数
    classifier.fit(X_train, Y_train)
    Y_pre = classifier.predict(X_test)
    auc, pre, rec, f1 = getEffect(Y_test.tolist(), Y_pre.tolist())
    print("auc - ", auc)
    print("pre - ", pre)
    print("rec - ", rec)
    print("f1 - ", f1)
    print()

    for iter in [10000, 20000, 100000]:
        print("="*20, iter, "="*20)
        # 使用逻辑回归算法
        lg = LogisticRegression(max_iter=iter)
        # 训练
        lg.fit(X_train, Y_train)
        # 预测
        Y_pre = lg.predict(X_test)
        auc, pre, rec, f1 = getEffect(Y_test.tolist(), Y_pre.tolist())
        print("auc - ", auc)
        print("pre - ", pre)
        print("rec - ", rec)
        print("f1 - ", f1)
        print()
    print("4、根据训练集确定阈值 done...", cnt_in, cnt_in, cnt_out)

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





















