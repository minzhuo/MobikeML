# -*- coding: utf-8 -*-

import numpy as np
import pandas as pd
import datetime


# 对geohash进行编码
geobase = '0123456789bcdefghjkmnpqrstuvwxyz'
geodecodemap = {}
for i in range(len(geobase)):
    geodecodemap[geobase[i]] = i
del i


# 按位计算起始坐标和终点坐标的编码值之差
def loc_dif(x, y):
    xn = [geodecodemap[c] for c in x]
    yn = [geodecodemap[c] for c in y]
    return list(map(lambda a: a[0] - a[1], zip(xn, yn)))


# 数据预处理，返回可供训练的train_x, train_y
def preprocess():

    # --------------------------------------
    # 读入原始数据
    dataf = pd.read_csv('data/train.csv')
    traindata = np.array(dataf)

    # --------------------------------------
    # 使用初始地点和结束地点的geohash编码按位计算的差值，作为训练的y
    # 这样在最后提交的时候，把原始的编码加上差值，得到最终地点
    # 好处是可以避免转换成经纬坐标再转换回去引起的误差

    train_bLoc = traindata[:, -2]  # 起点坐标
    train_eLoc = traindata[:, -1]  # 终点坐标

    train_y = list(map(loc_dif, train_bLoc, train_eLoc))
    train_y = np.matrix(train_y)
    # --------------------------------------
    # 以下整理出训练使用的x
    # 可以用时间数据的“时”作为一个输入特征

    train_timer = traindata[:, 4]
    train_hour = [int(d.split()[1].split(':')[0]) for d in train_timer]
    train_hour = np.matrix(train_hour).T

    # 把时间数据的日期转换成星期，然后用一个one-hot的7位向量表示
    # 原始训练数据中只有17年5月的{10, 11, 12, 13, 14, 15, 16, 18, 19, 20, 21, 22, 23, 24}这几个日期的数据
    train_date = [datetime.datetime.strptime(t, '%Y-%m-%d %H:%M:%S') for t in train_timer]
    train_week = [d.weekday() for d in train_date]
    train_one_hot_week = np.zeros((len(train_date), 7))
    train_one_hot_week = np.matrix(train_one_hot_week)

    for j in range(len(train_date)):
        train_one_hot_week[i, train_week[i]] = 1
    del j

    # 初始的地点翻译成一个7位的向量表示
    train_loc = np.matrix(list(map(lambda a: [geodecodemap[b] for b in a], train_bLoc)))

    # 使用星期几、几点和初始坐标作为训练用的x
    train_x = np.concatenate((train_one_hot_week, train_hour, train_loc), axis=1)

    return train_x, train_y


