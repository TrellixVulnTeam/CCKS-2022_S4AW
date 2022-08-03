import numpy as np
import pandas as pd
import json
import os
from py2neo import Node, Relationship, Graph, Path, Subgraph, NodeMatcher


def read_json(path):
    train_info = []
    with open(path, encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            train_info.append(json.loads(line.strip()))
    return train_info


def link_db():
    neo4j_url = 'bolt://localhost:7687'
    user = 'neo4j'
    pwd = '123456'
    return Graph(neo4j_url, auth=(user, pwd))


def query_relation(src_item_id, tgt_item_id):
    gragh = link_db()
    query = ''' match p=(n)-[*2]-(m) where n.item_id='{}' 
    and m.item_id='{}' return p '''.format(src_item_id, tgt_item_id)
    # print('''====关系查询''', query)
    return gragh.run(query).data()


if __name__ == '__main__':
    pair_data_list = read_json("../Data/item_train_pair.jsonl")
    df = pd.DataFrame()
    node_rel_list = []

    for pair_data in pair_data_list:
        if pair_data['item_label'] == '1':
            src_item_id = pair_data["src_item_id"]
            tgt_item_id = pair_data['tgt_item_id']
            print("query between ", src_item_id, tgt_item_id)
            res = query_relation(src_item_id, tgt_item_id)

            for i in res:
                i = i['p']
                head = i.nodes[0]
                relation = i.nodes[1]
                tail = i.nodes[2]

                one_rel = str(i.relationships[0]).split('[:')[1].split('{}')[0]
                two_rel = str(i.relationships[1]).split('[:')[1].split('{}')[0]

                tripe = [head['item_id'], head['title'], head['industry_name'], head['cate_name_path'],
                         one_rel, relation['pvs_id'], relation['item_pvs'], two_rel,
                         tail['item_id'], tail['title'], tail['industry_name'], tail['cate_name_path']]

                node_rel_list.append(tripe)

    df = pd.DataFrame(node_rel_list, columns=['head_item_id', 'head_title', 'head_industry_name', 'head_cate_name_path',
                   'head_pvs_rel', 'pvs_id', 'pvs', 'tail_pvs_rel',
                   'tail_item_id', 'tail_title', 'tail_industry_name', 'tail_cate_name_path'])
    df.to_csv('./convert/statistics/node-relation.csv', encoding="utf-8-sig", index=False)
