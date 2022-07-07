import json
import xlwt
import warnings
import pandas as pd

warnings.filterwarnings('ignore')

# 写如excel的某个sheet
def writeDataIntoExcel(xls_path, data, sheet):
    pd.DataFrame(data).to_excel(xls_path, sheet_name=sheet, index=False)


if __name__ == '__main__':

    # 1、加载每个item的属性
    train_info = []
    with open('./Data/item_train_info.jsonl', encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            train_info.append(json.loads(line.strip()))
    df_train_info = pd.DataFrame(train_info)
    print("1、加载每个item的属性 done...")

    # cnt = 0
    # for index, row in enumerate(df_train_info.itertuples()):
    #     cate_name = getattr(row, "cate_name")
    #     cate_name_path = getattr(row, "cate_name_path").split("->")[-1]
    #     if cate_name != cate_name_path:
    #         cnt += 1
    #         print(cnt, cate_name, cate_name_path, getattr(row, "item_image_name"))
    # print("无法对齐 - ", cnt)
    # print("累计数量 - ", df_train_info.shape)

    # 2、统计各属性出现频次
    # industry_name+cate_name_path (行业组+商品所属的类目路径名称)
    # for key1, group1 in df_train_info.groupby(["industry_name", "cate_name_path"]):
    #     print(key1)
    #
    #     # 统计该类目的item_pvs属性
    #     item_pvs_dict = {}
    #     for row in group1.itertuples():
    #         item_pvs = getattr(row, "item_pvs").split(";")
    #         for item_pv in item_pvs:
    #             p, v = item_pv.split(":")[0].replace("#", ""), item_pv.split(":")[-1].replace("#", "")
    #             if p not in item_pvs_dict:
    #                 item_pvs_dict[p] = {}
    #             if v in item_pvs_dict[p]:
    #                 item_pvs_dict[p][v] += 1
    #             else:
    #                 item_pvs_dict[p][v] = 1
    #     for key, value in item_pvs_dict.items():
    #         item_pvs_dict[key] = sorted(value.items(), key=lambda x: x[1], reverse=True)
    #         print(key, " : ", item_pvs_dict[key])
    #
    #     # 统计该类目的item_pvs属性
    #     item_pvs_dict = {}
    #     for row in group1.itertuples():
    #         item_pvs = getattr(row, "item_pvs").split(";")
    #         for item_pv in item_pvs:
    #             p, v = item_pv.split(":")[0].replace("#", ""), item_pv.split(":")[-1].replace("#", "")
    #             if p not in item_pvs_dict:
    #                 item_pvs_dict[p] = {}
    #             if v in item_pvs_dict[p]:
    #                 item_pvs_dict[p][v] += 1
    #             else:
    #                 item_pvs_dict[p][v] = 1
    #     for key, value in item_pvs_dict.items():
    #         item_pvs_dict[key] = sorted(value.items(), key=lambda x: x[1], reverse=True)
    #         print(key, " : ", item_pvs_dict[key])
    #
    #     # 统计该类目的sku_pvs属性
    #     sku_pvs_dict = {}
    #     for row in group1.itertuples():
    #         sku_pvs = getattr(row, "sku_pvs").split(";")
    #         for sku_pv in sku_pvs:
    #             p, v = sku_pv.split(":")[0].replace("#", ""), sku_pv.split(":")[-1].replace("#", "")
    #             if p not in sku_pvs_dict:
    #                 sku_pvs_dict[p] = {}
    #             if v in sku_pvs_dict[p]:
    #                 sku_pvs_dict[p][v] += 1
    #             else:
    #                 sku_pvs_dict[p][v] = 1
    #     for key, value in sku_pvs_dict.items():
    #         sku_pvs_dict[key] = sorted(value.items(), key=lambda x: x[1], reverse=True)
    #         print(key, " : ", sku_pvs_dict[key])
    #
    #     break

    for key1, group1 in df_train_info.groupby(["industry_name"]):
        print("\n", key1, group1.shape)

        for key2, group2 in group1.groupby(["cate_name_path"]):
            print(key2, group2.shape)

            pvs = []
            atts = []
            enums = []
            # 统计该类目的item_pvs属性
            item_pvs_dict = {}
            for row in group2.itertuples():
                item_pvs = str(getattr(row, "item_pvs")).split(";")
                for item_pv in item_pvs:
                    p, v = item_pv.split(":")[0].replace("#", ""), item_pv.split(":")[-1].replace("#", "")
                    if p not in item_pvs_dict:
                        item_pvs_dict[p] = {}
                    if v in item_pvs_dict[p]:
                        item_pvs_dict[p][v] += 1
                    else:
                        item_pvs_dict[p][v] = 1
            for key, value in item_pvs_dict.items():
                item_pvs_dict[key] = sorted(value.items(), key=lambda x: x[1], reverse=True)
                pvs.append("item_pvs")
                atts.append(key)
                enums.append("、".join([str(data[0])+"_"+str(data[1]) for data in item_pvs_dict[key]])[:32767])

            # 统计该类目的sku_pvs属性
            sku_pvs_dict = {}
            for row in group2.itertuples():
                sku_pvs = getattr(row, "sku_pvs").split(";")
                for sku_pv in sku_pvs:
                    p, v = sku_pv.split(":")[0].replace("#", ""), sku_pv.split(":")[-1].replace("#", "")
                    if p not in sku_pvs_dict:
                        sku_pvs_dict[p] = {}
                    if v in sku_pvs_dict[p]:
                        sku_pvs_dict[p][v] += 1
                    else:
                        sku_pvs_dict[p][v] = 1
            for key, value in sku_pvs_dict.items():
                sku_pvs_dict[key] = sorted(value.items(), key=lambda x: x[1], reverse=True)
                pvs.append("sku_pvs")
                atts.append(key)
                enums.append("、".join([str(data[0]) + "_" + str(data[1]) for data in sku_pvs_dict[key]])[:32767])

            pd.DataFrame({"pvs":pvs, "atts":atts, "enums":enums}).to_excel("./Industry/"+key1+"/"+str(key2).replace("/", "|")+"_"+str(group2.shape[0])+".xls", index=False)

