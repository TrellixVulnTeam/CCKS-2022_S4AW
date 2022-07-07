import os
import csv
import json
import torch
import warnings
import numpy as np
import pandas as pd
import torch.nn as nn
from sklearn import svm
from typing import List
from utils.util import *
import torch.utils.data as Data
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


# 网络类
class MLP(nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc = nn.Sequential(  # 添加神经元以及激活函数
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Linear(256, 128),
            nn.ReLU(),,
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Linear(64, 2),
        )
        self.mse = nn.CrossEntropyLoss()
        self.optim = torch.optim.Adam(params=self.parameters(), lr=0.001)

    def forward(self, inputs):
        outputs = self.fc(inputs)
        return outputs

    def train(self, x, label):
        out = self.forward(x)       # 正向传播
        loss = self.mse(out, label) # 根据正向传播计算损失
        self.optim.zero_grad()      # 梯度清零
        loss.backward()             # 计算梯度
        self.optim.step()           # 应用梯度更新参数
        return float(loss)

    def test(self, test_):
        return self.fc(test_)


if __name__ == '__main__':

    BATCH_SIZE = 256    # 128-0.8169549515301274-0.088369101、512-0.8083018867924529-0.095467805
    EPOCHS = 30 # 20-0.8093594169543536-0.14584417、30-0.819935985368084-0.081910349、50-0.8140534479536535-0.0671852380

    # 1、加载每个item的vector
    # itemid2emb = {}
    # with open('./testv2/item_train_features.tsv') as tsv_in_file:
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
            # if cnt_all >= 100:
            #     break
            line = line.strip()
            item = json.loads(line)
            src_item_id = item['src_item_id']
            tgt_item_id = item['tgt_item_id']
            item_label = item['item_label']
            try:
                # 计算两个item的得分
                src_item_emb = [float(vec) for vec in itemid2emb[src_item_id][1:-1].split(",")]
                tgt_item_emb = [float(vec) for vec in itemid2emb[tgt_item_id][1:-1].split(",")]
                item_emb = src_item_emb+tgt_item_emb
                X.append(item_emb)
                Y.append(int(item_label))
                cnt_in += 1
            except:
                cnt_out += 1
                continue
    # 划分测试集、训练集
    X_train, X_test, Y_train, Y_test = train_test_split(np.array(X), np.array(Y), test_size=0.2)
    # 加载模型
    model = MLP()
    train_dataset = Data.TensorDataset(torch.from_numpy(X_train).float(), torch.from_numpy(Y_train).long())
    train_loader = Data.DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    # 训练
    for epoch in range(EPOCHS):
        for step, (x, y) in enumerate(train_loader):
            loss = model.train(x, y)
            print('Epoch:', epoch, '| Step:', step, '| Loss:', loss)
    # 测试
    out = model.test(torch.from_numpy(X_test).float())
    Y_test = Y_test.tolist()
    Y_pre = torch.max(out, 1)[1].data.numpy().tolist()  # 1返回index  0返回原值
    auc, pre, rec, f1 = getEffect(Y_test, Y_pre)
    print("auc - ", auc)
    print("pre - ", pre)
    print("rec - ", rec)
    print("f1 - ", f1)
    # # 预测单个样本
    # for i in range(X_test.shape[0]):
    #     x_real = np.array([X_test[i].tolist()])
    #     y_real = [Y_test[i]]
    #     out = model.test(torch.from_numpy(x_real).float())
    #     y_pre = torch.max(out, 1)[1].data.numpy().tolist()  # 1返回index  0返回原值
    #     print(i, " - ", y_real, y_pre)
    print("4、根据训练集确定阈值 done...", cnt_in, cnt_in, cnt_out, "\n")


    # 5、确定测试集的阈值
    # itemid2emb = {}
    # with open('./testv2/item_valid_features.tsv') as tsv_in_file:
    #     reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=['item_id', 'features'])
    #     for item in reader:
    #         itemid2emb[str(item['item_id'])] = '[' + item['features'] + ']'
    # pickleFile("./cache/itemid2emb_valid.pkl", itemid2emb)
    itemid2emb = unPickleFile("./cache/itemid2emb_valid.pkl")

    f_out = open('./result/XXXX_result.jsonl', encoding='utf-8', mode='w')
    with open('../../Data/item_valid_pair.jsonl', encoding='utf-8', mode='r') as f:
        cnt_all = 0
        for line in f.readlines():
            cnt_all += 1
            line = line.strip()
            item = json.loads(line)
            src_item_id = item['src_item_id']
            tgt_item_id = item['tgt_item_id']
            src_item_emb = itemid2emb[src_item_id]
            tgt_item_emb = itemid2emb[tgt_item_id]

            # 计算两个item的得分
            src_item_emb = [float(vec) for vec in itemid2emb[src_item_id][1:-1].split(",")]
            tgt_item_emb = [float(vec) for vec in itemid2emb[tgt_item_id][1:-1].split(",")]
            item_emb = src_item_emb + tgt_item_emb

            # 预测得分
            x_real = np.array([item_emb])
            out = model.test(torch.from_numpy(x_real).float())
            y_pre = torch.max(out, 1)[1].data.numpy().tolist()[0]  # 1返回index  0返回原值

            # 动态调整threshold
            score = compute(src_item_emb, tgt_item_emb)
            if y_pre == 1:
                threshold = max(0, score-0.05)
            else:
                threshold = min(score+0.05, 1)
            out_item = {
                "src_item_id": src_item_id,
                "src_item_emb": itemid2emb[src_item_id],
                "tgt_item_id": tgt_item_id,
                "tgt_item_emb": itemid2emb[tgt_item_id],
                "threshold": float(threshold)
            }
            f_out.write(json.dumps(out_item) + '\n')
    f_out.close()





















