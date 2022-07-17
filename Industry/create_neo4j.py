import numpy as np
import pandas as pd
import json
from py2neo import Node, Relationship, Graph, Path, Subgraph


def link_db():
    neo4j_url = 'bolt://localhost:7687'
    user = 'neo4j'
    pwd = '123456'
    return Graph(neo4j_url, auth=(user, pwd))


if __name__ == '__main__':
    graph = link_db()

    train_info = []
    with open('./大服饰.jsonl', encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            train_info.append(json.loads(line.strip()))
    df_train_info = pd.DataFrame(train_info)

    for i_item, j_item in df_train_info.iterrows():
        item_node = Node("大服饰item", item_id=j_item["item_id"], industry_name=j_item["industry_name"],
                         cate_id=j_item["cate_id"], cate_name=j_item["cate_name"], cate_id_path=j_item["cate_id_path"],
                         cate_name_path=j_item["cate_name_path"], item_image_name=j_item["item_image_name"],
                         title=j_item["title"])
        item_pvs_node = Node("大服饰pvs", item_pvs=j_item["item_pvs"])
        sku_pvs_node = Node("大服饰sku_pvs", sku_pvs=j_item["sku_pvs"])
        graph.create(item_node)
        graph.create(item_pvs_node)
        graph.create(sku_pvs_node)

        pvs_rel = Relationship(item_node, "is a", item_pvs_node)
        sku_rel = Relationship(item_node, "is a", sku_pvs_node)
        graph.create(pvs_rel)
        graph.create(sku_rel)
