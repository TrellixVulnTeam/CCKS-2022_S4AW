import json
import pandas as pd


if __name__ == '__main__':

    # 1、加载每个item的属性
    item_dict = {}
    with open('./Data/item_train_info.jsonl', encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            dict1 = json.loads(line.strip())
            item_dict[dict1["item_id"]] = dict1
    print("1、加载每个item的属性 done...", len(item_dict))

    # 2、加载同款商品
    train_pair = []
    with open('./Data/item_train_pair.jsonl', encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            train_pair.append(json.loads(line.strip()))
    df_train_pair = pd.DataFrame(train_pair)
    print("2、加载同款商品 done...", df_train_pair.shape)


    # 获取item_id的映射关系
    cnt_all = 0
    cnt_same = 0
    for index, row in enumerate(df_train_pair.itertuples()):
        cnt_all += 1
        # src数据详情
        src_data = item_dict[getattr(row, "src_item_id")]
        src_cate_name_path = src_data["cate_name_path"]
        # tgt数据详情
        tgt_data = item_dict[getattr(row, "tgt_item_id")]
        tgt_cate_name_path = tgt_data["cate_name_path"]
        if src_cate_name_path == tgt_cate_name_path:
            cnt_same += 1
        else:
            print("src - ", src_data)
            print("tgt - ", tgt_data)
            print()

    print("累计相似物品对 - ", cnt_all)
    print("累计相似物品对(同路径) - ", cnt_same)
    print("累计相似物品对(不同路径) - ", cnt_all-cnt_same)