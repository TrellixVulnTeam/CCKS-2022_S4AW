import os
import csv
import json
import time
import pickle
import calendar
import datetime
import pandas as pd
from numpy import *
from itertools import islice


# 获取文件夹下所有文件
def getAllFiles(path):  # 获取所有文件
    all_file = []
    for f in os.listdir(path):  #listdir返回文件中所有目录
        all_file.append(f)
    return all_file



# 序列化变量到磁盘中
def pickleFile(pickle_path, my_object):
    """
    序列化变量到磁盘中
    :return:
    """
    with open(pickle_path, "wb") as file:
        pickle.dump(my_object, file)


# 序列化多个变量到磁盘中
def pickleFiles(pickle_path, my_objects):
    """
    序列化变量到磁盘中
    :return:
    """
    with open(pickle_path, "wb") as file:
        for my_object in my_objects:
            pickle.dump(my_object, file, pickle.HIGHEST_PROTOCOL)


# 反序列化变量到内存中
def unPickleFile(pickle_path):
    """
    反序列化变量到内存中
    :return:
    """
    with open(pickle_path, "rb") as file:
        return pickle.load(file)


# 反序列化变量到内存中
def unPickleFiles(pickle_path):
    """
    反序列化变量到内存中
    :return:
    """
    with open(pickle_path, "rb") as file:
        my_objects = []
        while True:
            try:
                my_object = pickle.load(file)
                my_objects.append(my_object)
            except:
                break
        return my_objects

# 写入csv
def writeCSV(csv_path, data_list):
    """
    写入csv文件
    :param path:
    :param data_list:
    :return:
    """
    with open(csv_path, "w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerows(data_list)


# 读取csv
def readCSV(csv_path):
    """
    读取csv文件
    :param csv_path:
    :return:
    """
    csv_file = open(csv_path, "r", encoding="utf-8")
    reader = csv.reader(csv_file)
    result = []
    for item in reader:
        # 忽略第一行
        if reader.line_num == 1:
            continue
        result.append(item)
    csv_file.close()
    return result


# 带参数装饰器，用于打印日志
def logRunTime(desc=""):
    def metric(func):
        def wrapper(*args, **kw):
            start_time = time.time()
            result = func(*args, **kw)
            print('{} [{}] 运行耗时: {}s.'.format(desc, func.__name__, time.time()-start_time))
            return result
        return wrapper
    return metric


# 写入neo4j属性字符串清洗
def neo4jAttributeClean(str_old):
    str_new = str(str_old).strip()
    if str_old == "/":
        str_new = "-"
    str_new = str_new.replace(",", "，")
    str_new = str_new.replace("\"", "")
    str_new = str_new.replace("/.html", "-")
    return str_new


# 映射到neo4j存储的节点属性
def getNeo4jAttribute(old_dict, specified):
    if specified == "line":
        specified_dict = ATTRIBUTE_NODE_LINE
    elif specified == "model":
        specified_dict = ATTRIBUTE_NODE_MODEL
    else:
        print("请指定该节点属性集")
        return
    new_dict = {}
    for key, value in old_dict.items():
        key = key.replace("、", "").replace("/", "_")
        if key in specified_dict:
            new_dict[key] = neo4jAttributeClean(value)
    return new_dict


# 将元素为dict的list写入csv
def jsonArryToCSV(dict_list, path):
    df = pd.DataFrame(dict_list)
    df = df.fillna("-")
    try:
        df[df==''] = "-"
    except:
        pass
    df.to_csv(path,index=None)


# 将元素为dict的list写入csv
def setArryToCSV(set_list, path):
    set_new = set()
    for data in set_list:
        if len(data) > 0:
            for element in data:
                set_new.add(element)
    data_list = [["name"]]
    for element in set_new:
        data_list.append([element])
    writeCSV(path, data_list)


# 获取产品型号对应的卖点集合
def getSellPoint(attribute_dict, SELL_POINT):
    concept_set = set()
    for key, value in attribute_dict.items():
        if key == "卖点":
            value = value.upper()
            for word, triggers in SELL_POINT.items():
                if word in value:
                    for trigger in triggers:
                        concept_set.add(trigger)
            break
    return concept_set


# 计算两个日期之间差多少秒
def getMinuteDifference(time1, time2):
    """
    计算两个日期之间差多少秒
    :param date1:
    :param date2:
    :return:
    """
    time1 = datetime.datetime.strptime(time1, "%Y-%m-%d %H:%M:%S")
    time2 = datetime.datetime.strptime(time2, "%Y-%m-%d %H:%M:%S")
    return abs((time2 - time1).seconds)


# 计算距今天差多少天
def getDayDifferenceAgo(time_last):
    """
    计算两个日期之间差多少秒
    :param date1:
    :param date2:
    :return:
    """
    year, month, day = time.strptime(time_last, "%Y-%m-%d")[:3]
    day_last = datetime.date(year, month, day)
    day_now = datetime.date.today()
    return abs((day_now - day_last).days)


# 计算日期列表平均日期间隔
def getDayDifferenceAverage(times):
    """
    计算日期列表平均日期间隔
    :param date1:
    :param date2:
    :return:
    """
    if len(times) < 1:
        return 0, 0, 0
    elif len(times) == 1:
        return 0, 0, 0
    else:
        diffs = []
        for index in range(len(times)-1):
            year_now, month_now, day_now = time.strptime(times[index], "%Y-%m-%d")[:3]
            day_now = datetime.date(year_now, month_now, day_now)
            year_next, month_next, day_next = time.strptime(times[index+1], "%Y-%m-%d")[:3]
            day_next = datetime.date(year_next, month_next, day_next)
            diffs.append(abs((day_next-day_now).days))
        return min(diffs), int(mean(diffs)), max(diffs)


# 带参数装饰器，用于打印日志
def logRunTime(desc=""):
    def metric(func):
        def wrapper(*args, **kw):
            start_time = time.time()
            result = func(*args, **kw)
            print('{} [{}] 运行耗时: {}s.\n'.format(desc, func.__name__, time.time()-start_time))
            return result
        return wrapper
    return metric


# 返回某年某月的所有日期
def getMothDate(year, month):
    """
    返回某年某月的所有日期
    :param year:
    :param month:
    :return:
    """
    date_list = []
    for i in range(calendar.monthrange(year, month)[1] + 1)[1:]:
        str1 = str(year) + str("%02d" % month) + str("%02d" % i)
        date_list.append(str1)
    return date_list
