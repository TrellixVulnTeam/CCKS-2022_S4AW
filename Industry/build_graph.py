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
    return pd.DataFrame(train_info)


def step_1():
    json_data = read_json('./食品.jsonl')

    for root, dirs, files in os.walk("./Attribute/食品"):
        # root 表示当前正在访问的文件夹路径 dirs 表示该文件夹下的子目录名list files 表示该文件夹下的文件list
        for f in files:
            print("处理", os.path.join(root, f))

            base_name = os.path.splitext(f)[0].replace('|', '/')
            print('------------' + base_name)
            df = pd.DataFrame()
            for i_item, j_item in json_data.iterrows():
                print("+++++++++" + j_item["cate_name_path"])
                if j_item["cate_name_path"] == base_name:
                    df = df.append(j_item, ignore_index=True)
            df.to_csv('./convert/食品/' + f, encoding="utf-8-sig")


def list_base_name(path):
    base_name_list = []
    for root, dirs, files in os.walk(path):
        # root 表示当前正在访问的文件夹路径 dirs 表示该文件夹下的子目录名list files 表示该文件夹下的文件list
        for f in files:
            if f == '.DS_Store':
                continue
            base_name = os.path.splitext(f)[0]
            base_name_split = base_name.split("->")
            if len(base_name_split) >= 2:
                seq = (base_name_split[0], base_name_split[1])
                second_base_name = '->'.join(seq)
                base_name_list.append(second_base_name)
            else:
                base_name_list.append(base_name)

    return list(set(base_name_list))


def merge_data():
    path = "./convert/食品_depart/"
    target_path = "./convert/食品_merge/"
    base_name_list = list_base_name(path)

    for root, dirs, files in os.walk(path):
        for name in base_name_list:
            csv_data_list = []
            for f in files:
                if f == ".DS_Store":
                    continue
                print("处理", os.path.join(root, f))
                base_name = os.path.splitext(f)[0]

                if name in base_name:
                    csv_data = pd.read_csv(os.path.join(root, f), header=0)
                    csv_data_list.append(csv_data)
            data_frame_concat = pd.concat(csv_data_list, axis=0, ignore_index=True)
            data_frame_concat.to_csv(target_path + name + ".csv")


def fix_pvs():
    for root, dirs, files in os.walk("./Attribute/食品"):
        # 读取attribute作为标准
        for f in files:
            if f == ".DS_Store":
                continue
            print("处理", os.path.join(root, f))

            base_name = os.path.splitext(f)[0]
            # print('------------' + base_name)
            csv_data = pd.read_csv(os.path.join(root, f))
            menu_list = []
            for i_item, j_item in csv_data.iterrows():
                newint = float(j_item["比例"].strip("%"))
                if newint > 50.0:
                    menu_list.append(j_item["共有属性"])

            for convert_root, convert_dirs, convert_files in os.walk("./convert/食品"):
                # 读取convert数据
                for convert_f in convert_files:
                    if convert_f == ".DS_Store":
                        continue
                    print("读取", os.path.join(convert_root, convert_f))
                    convert_base_name = os.path.splitext(convert_f)[0]
                    if convert_base_name == base_name:
                        df = pd.DataFrame()
                        convert_csv_data = pd.read_csv(os.path.join(convert_root, convert_f))
                        for convert_i_item, convert_j_item in convert_csv_data.iterrows():
                            pvs_list = str(convert_j_item["item_pvs"]).split(";")
                            sku_list = str(convert_j_item["sku_pvs"]).split(";")
                            final_pvs = []
                            final_sku = []
                            for pvs in pvs_list:
                                pvs_index = pvs.split(":")[0].strip("#")
                                if pvs_index in menu_list:
                                    final_pvs.append(pvs)
                            for sku in sku_list:
                                sku_index = sku.split(":")[0].strip("#")
                                if sku_index in menu_list:
                                    final_sku.append(sku)
                            convert_j_item["item_pvs"] = final_pvs
                            convert_j_item["sku_pvs"] = final_sku
                            # print("=====", convert_j_item)
                            df = df.append(convert_j_item, ignore_index=True)
                        df.to_csv('./convert/食品_depart/' + convert_f, encoding="utf-8-sig")


def create_item(j_item):
    item_list = []
    item_list.append(j_item["item_id"])
    item_list.append(j_item["industry_name"])
    item_list.append(j_item["cate_id"])
    item_list.append(j_item["cate_name"])
    item_list.append(j_item["cate_id_path"])
    item_list.append(j_item["cate_name_path"])
    item_list.append(j_item["item_image_name"])
    item_list.append(j_item["title"])
    item_list.append("item_id")
    return item_list


def analy_pvs_list(j_item):
    item_pvs = eval(j_item["item_pvs"])
    item_pvs.append(eval(j_item["sku_pvs"]))
    return item_pvs


def create_pvs_item(pvs_id, pvs_item, _all_pvs_item_list):
    if pvs_item in _all_pvs_item_list:
        return None, _all_pvs_item_list[pvs_item]
    else:
        _all_pvs_item_list[pvs_item] = pvs_id
        pvs_item_list = [pvs_id, pvs_item, "pvs_id"]
        return pvs_item_list, pvs_id


def create_relation(item_id, pvs_id, relation):
    relation_list = []
    relation_list.append(item_id)
    relation_list.append(pvs_id)
    relation_list.append(relation)
    return relation_list


def create_triple():
    for root, dirs, files in os.walk("./convert/食品_merge"):
        for f in files:
            if f == ".DS_Store":
                continue
            print("处理", os.path.join(root, f))

            base_name = os.path.splitext(f)[0]
            # print('------------' + base_name)
            csv_data = pd.read_csv(os.path.join(root, f))
            all_item_list = []
            all_pvs_list = []
            all_relation_list = []
            index = 1000
            _all_pvs_item_list = {}
            for i_item, j_item in csv_data.iterrows():

                item_id = j_item["item_id"]
                item_list = create_item(j_item)
                all_item_list.append(item_list)

                pvs_list = analy_pvs_list(j_item)
                if len(pvs_list) > 0:
                    for pvs in pvs_list:
                        pvs_split = str(pvs).split(":")
                        pvs_id = item_id + str(index)
                        index = index + 1
                        if len(pvs_split) > 1:
                            relation = pvs_split[0].strip("#").strip("['").strip("['#").strip("']").strip("#']").strip()
                            pvs_item = pvs_split[1].strip("#").strip("['").strip("['#").strip("']").strip("#']").strip()
                            pvs_item_list, pvs_id = create_pvs_item(pvs_id, pvs_item, _all_pvs_item_list)
                            if pvs_item_list is not None:
                                all_pvs_list.append(pvs_item_list)

                            relation_list = create_relation(item_id, pvs_id, relation)
                            all_relation_list.append(relation_list)

            make_item = pd.DataFrame(all_item_list,
                                     columns=['item:ID', 'industry_name', 'cate_id', 'cate_name', 'cate_id_path',
                                              'cate_name_path', 'item_image_name', 'title', ':LABEL'])
            make_item.to_csv(os.path.join('./convert/食品_triple/entity_item', base_name + "_item.csv"),
                             encoding="utf-8-sig", index=False)

            make_item = pd.DataFrame(all_pvs_list,
                                     columns=['pvs:ID', 'pvs', ':LABEL'])
            make_item.to_csv(os.path.join('./convert/食品_triple/entity_pvs', base_name + "_pvs.csv"),
                             encoding="utf-8-sig", index=False)

            make_item = pd.DataFrame(all_relation_list,
                                     columns=[':START_ID', ':END_ID', ':TYPE'])
            make_item.to_csv(os.path.join('./convert/食品_triple/relation', base_name + "_relation.csv"),
                             encoding="utf-8-sig", index=False)


def link_db():
    neo4j_url = 'bolt://localhost:7687'
    user = 'neo4j'
    pwd = '123456'
    return Graph(neo4j_url, auth=(user, pwd))


def import_neo4j():
    db = link_db()
    node_matcher = NodeMatcher(db)
    for root, dirs, files in os.walk("./convert/食品_merge"):
        for f in files:
            if f == ".DS_Store":
                continue
            print("处理", os.path.join(root, f))

            base_name = os.path.splitext(f)[0]
            # print('------------' + base_name)
            csv_data = pd.read_csv(os.path.join(root, f))
            all_item_list = []
            all_pvs_list = []
            all_relation_list = []
            index = 1000
            _all_pvs_item_list = {}
            for i_item, j_item in csv_data.iterrows():

                item_id = j_item["item_id"]
                item_node = Node("食品item", item_id=item_id, industry_name=j_item["industry_name"],
                                 cate_id=j_item["cate_id"], cate_name=j_item["cate_name"],
                                 cate_id_path=j_item["cate_id_path"],
                                 cate_name_path=j_item["cate_name_path"], item_image_name=j_item["item_image_name"],
                                 title=j_item["title"],  type=base_name)
                db.create(item_node)

                pvs_list = analy_pvs_list(j_item)
                if len(pvs_list) > 0:
                    for pvs in pvs_list:
                        pvs_split = str(pvs).split(":")
                        pvs_id = item_id + str(index)
                        index = index + 1
                        if len(pvs_split) > 1:
                            relation = pvs_split[0].strip("#").strip("['").strip("['#").strip("']").strip("#']").strip()
                            pvs_item = pvs_split[1].strip("#").strip("['").strip("['#").strip("']").strip("#']").strip()
                            pvs_item_list, pvs_id = create_pvs_item(pvs_id, pvs_item, _all_pvs_item_list)
                            if pvs_item_list is not None:
                                all_pvs_list.append(pvs_item_list)
                                item_pvs_node = Node("食品pvs", pvs_id=pvs_id, item_pvs=pvs_item_list[1])
                                pvs_rel = Relationship(item_node, relation, item_pvs_node)
                                db.create(item_pvs_node)
                                db.create(pvs_rel)
                            else:
                                matched_node = node_matcher.match("食品pvs").where(pvs_id=pvs_id).first()
                                pvs_rel = Relationship(item_node, relation, matched_node)
                                db.create(pvs_rel)

if __name__ == '__main__':
    import_neo4j()
