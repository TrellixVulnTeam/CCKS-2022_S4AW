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
            # nn.Linear(1636, 1024),
            # nn.ReLU(),
            # nn.Linear(1024, 512),
            # nn.ReLU(),
            # nn.Linear(512, 256),
            # nn.ReLU(),
            # nn.Linear(256, 128),
            # nn.ReLU(),
            # nn.Linear(128, 64),
            # nn.ReLU(),
            # nn.Linear(64, 2),
            nn.Linear(1536, 1024),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, 256),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(64, 2),
            nn.LogSoftmax(dim=1)
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


def query_relation():
    csv_data = pd.read_csv('./cache/node-relation.csv')
    relation_list = []
    relation_pair = {}
    for i_item, j_item in csv_data.iterrows():
        item_pair = j_item['head_item_id'] + '-' + j_item['tail_item_id']
        relation = j_item['head_pvs_rel'].strip(' ') + ":" + j_item['pvs'].strip(' ')
        # relation = j_item['head_pvs_rel'].strip(' ')

        if relation not in relation_list:
            relation_list.append(relation)
        if item_pair in relation_pair.keys():
            relations = relation_pair[item_pair]
            relations.append(relation)
            relation_pair[item_pair] = relations
        else:
            relation_pair[item_pair] = [relation]
    return relation_list, relation_pair


def query_relation_embedding(src_item_id, tgt_item_id, relation_pair, relation_list, relation_embedding):
    item_id_pair = src_item_id + '-' + tgt_item_id
    input = [0 for i in range(len(relation_list))]
    if item_id_pair in relation_pair.keys():
        relation_pair_list = relation_pair[item_id_pair]
        for relation in relation_pair_list:
            if relation in relation_list:
                index = relation_list.index(relation)
                input[index] = 1

    embedding = relation_embedding(torch.LongTensor(input))
    return embedding


def split_pvs(item):
    try:
        pvs_split = item['item_pvs'].split(";")
        pvs_split = pvs_split + item['sku_pvs'].split(";")
        new_pvs_list = []
        for pvs in pvs_split:
            new_pvs_list.append(pvs.strip(' ').replace('#', ''))

        return new_pvs_list
    except:
        return []

def read_valid_json():
    valid_json_list = []
    with open('../../Data/item_valid_info.jsonl', encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            line = line.strip()
            valid_json_list.append(line)

    return valid_json_list


def query_valid_relation_embedding(src_item_id, tgt_item_id, relation_embedding, relation_list, valid_json_list):
    for line in valid_json_list:
        item = json.loads(line)
        if item['item_id'] == src_item_id:
            src_item = item
        if item['item_id'] == tgt_item_id:
            tgt_item = item
    src_pvs_list = split_pvs(src_item)
    tgt_pvs_list = split_pvs(tgt_item)

    input = [0 for i in range(len(relation_list))]
    for pvs in src_pvs_list:
        if pvs in tgt_pvs_list:
            # pvs_key = pvs.split(":")[0]
            for relation in relation_list:
                if pvs == relation:
                    index = relation_list.index(relation)
                    input[index] = 1
    embedding = relation_embedding(torch.LongTensor(input))
    return embedding


def process_emb(rel_emb_tensor):
    rel_emb_list = rel_emb_tensor.tolist()
    tmp = np.array(rel_emb_list)
    return tmp.mean(axis=0).tolist()


def judge_gategory(src_item_id, tgt_item_id, valid_json_list):
    for line in valid_json_list:
        item = json.loads(line)
        if item['item_id'] == src_item_id:
            src_item = item
        if item['item_id'] == tgt_item_id:
            tgt_item = item
    src_split = src_item['cate_name_path'].split('/')
    if len(src_split) >=2:
        src_gory = src_split[0] + '-' + src_split[1]
    else:
        src_gory = src_split[0]

    tgt_split = tgt_item['cate_name_path'].split('/')
    if len(tgt_split) >=2:
        tgt_gory = tgt_split[0] + '-' + tgt_split[1]
    else:
        tgt_gory = tgt_split[0]

    if src_gory != tgt_gory:
        return False
    else:
        return True

if __name__ == '__main__':

    BATCH_SIZE = 256    # 128-0.8169549515301274-0.088369101、512-0.8083018867924529-0.095467805
    EPOCHS = 100 # 20-0.8093594169543536-0.14584417、30-0.819935985368084-0.081910349、50-0.8140534479536535-0.0671852380

    # 对训练集中的所有关系embedding
    # print("loading relationEmbedding")
    # relation_list, relation_pair = query_relation()
    # relation_embedding = nn.Embedding(len(relation_list), 100)
    # print("loading relationEmbedding done...")

    # 1、加载每个item的vector
    # itemid2emb = {}
    # with open('./testv2/item_train_features.tsv') as tsv_in_file:
    #     reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=['item_id', 'features'])
    #     for item in reader:
    #         itemid2emb[str(item['item_id'])] = '[' + item['features'] + ']'
    # pickleFile("./cache/itemid2emb_train.pkl", itemid2emb)
    itemid2emb = unPickleFile("./cache/itemid2emb_train.pkl")
    print("加载每个item的vector done...")

    # 2、加载每个item的属性
    train_info = []
    with open('../../Data/item_train_info.jsonl', encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            train_info.append(json.loads(line.strip()))
    df_train_info = pd.DataFrame(train_info)
    print("加载每个item的属性 done...")

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
                # 计算节点间的关联信息
                # rel_emb_tensor = query_relation_embedding(src_item_id, tgt_item_id, relation_pair, relation_list, relation_embedding)
                # rel_emb_list = process_emb(rel_emb_tensor)

                item_emb = src_item_emb + tgt_item_emb

                X.append(item_emb)
                Y.append(int(item_label))
                cnt_in += 1
            except:
                cnt_out += 1
                continue

    print("根据训练集确定阈值 done...", cnt_in, cnt_in, cnt_out, "\n")

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
    print("train done...", "\n")


    # 5、确定测试集的阈值
    # itemid2emb = {}
    # with open('./testv2/item_valid_features.tsv') as tsv_in_file:
    #     reader = csv.DictReader(tsv_in_file, delimiter='\t', fieldnames=['item_id', 'features'])
    #     for item in reader:
    #         itemid2emb[str(item['item_id'])] = '[' + item['features'] + ']'
    # pickleFile("./cache/itemid2emb_valid.pkl", itemid2emb)
    itemid2emb = unPickleFile("./cache/itemid2emb_valid.pkl")
    print("loading valid data done...")
    print("predicting wait...")
    valid_json_list = read_valid_json()

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

            same_gory = judge_gategory(src_item_id, tgt_item_id, valid_json_list)
            if same_gory:
                # 计算两个item的得分
                src_item_emb = [float(vec) for vec in itemid2emb[src_item_id][1:-1].split(",")]
                tgt_item_emb = [float(vec) for vec in itemid2emb[tgt_item_id][1:-1].split(",")]
                # 计算节点间的关联信息
                # rel_emb_tensor = query_valid_relation_embedding(src_item_id, tgt_item_id, relation_embedding, relation_list, valid_json_list)
                # rel_emb_list = process_emb(rel_emb_tensor)

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
            else:
                # 动态调整threshold
                src_item_emb = [float(vec) for vec in itemid2emb[src_item_id][1:-1].split(",")]
                tgt_item_emb = [float(vec) for vec in itemid2emb[tgt_item_id][1:-1].split(",")]
                score = compute(src_item_emb, tgt_item_emb)
                threshold = min(score + 0.05, 1)
                out_item = {
                    "src_item_id": src_item_id,
                    "src_item_emb": itemid2emb[src_item_id],
                    "tgt_item_id": tgt_item_id,
                    "tgt_item_emb": itemid2emb[tgt_item_id],
                    "threshold": float(threshold)
                }
                f_out.write(json.dumps(out_item) + '\n')
    f_out.close()
    print("DONE !")





















