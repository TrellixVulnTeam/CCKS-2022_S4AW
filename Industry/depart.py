
import json
import pandas as pd

if __name__ == '__main__':
    train_info = []
    with open('../Data/item_train_info.jsonl', encoding='utf-8', mode='r') as f:
        for line in f.readlines():
            train_info.append(json.loads(line.strip()))
    df_train_info = pd.DataFrame(train_info)
    industry_list = list(set(df_train_info['industry_name']))

    for industry in industry_list:
        fileName = industry + '.jsonl'
        with open(fileName, 'w', encoding='utf-8')as file:
            for i_item, j_item in df_train_info.iterrows():
                if j_item["industry_name"] == industry:
                    str_item = str(j_item.to_json(orient="columns", force_ascii=False))
                    file.write(str(str_item))
                    file.write('\n')
