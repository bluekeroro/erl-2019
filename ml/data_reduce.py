# -*- coding:UTF-8 -*-
"""
@File    : data_reduce.py
@Time    : 2019/6/4 14:18
@Author  : Blue Keroro
"""
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

import random

from tqdm import tqdm

from util import loadData


def get_train_data(data_path):
    train_data = []
    tmp = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line == '\n':
                train_data.append(tmp)
                tmp = []
            else:
                tmp.append(line.strip('\n'))
    # random.shuffle(train_data)
    train_data = [[j.split('@$$@') for j in i] for i in train_data]
    train_x = [[token[0] for token in sen] for sen in train_data]
    train_y = [[token[1] for token in sen] for sen in train_data]
    return train_x, train_y


def reduce(line, labels):
    label_list = [labels[0] for i in line['text']]
    for ment in line['mention_data']:
        label_list[int(ment['offset'])] = labels[1]
        for i in range(len(ment['mention']) - 1):
            label_list[int(ment['offset']) + i + 1] = labels[2]
    ret = []
    for index in range(len(line['text'])):
        ret.append(line['text'][index] + '@$$@' + label_list[index])
    return ret


def predict_reduce(test_x, pred_y, labels):
    index = -1
    entitys = list()
    while (1):
        try:
            index = pred_y.index(labels[1], index + 1)
            entity = test_x[index]
            for i in range(index + 1, len(test_x)):
                if pred_y[i] != labels[2]:
                    break
                entity += test_x[i]
            entitys.append([entity, index])
        except Exception:
            break
    return entitys


def loadBertPred(path):
    data = []
    with open(path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            line = eval(line)
            line = [str(i) for i in line]
            data.append(line)
    return data
def loadDevData(path):
    data = loadData(path)
    ret=[]
    for sub in data:
        ret.append(list(sub['text']))
    return ret

if __name__ == '__main__':
    labels = ['O', 'B-ment', 'I-ment']
    data = loadData('data/train_pre.json')
    with open('data/train_text.txt', 'w', encoding='utf-8') as file:
        for line in tqdm(data):
            line = reduce(line, labels)
            file.write('\n'.join(line))
            file.write('\n\n')
    # train_x, train_y = get_train_data('data/train_text.txt')
    # print(train_x)
    # print(train_y)
    # print(predict_reduce(train_x[0], train_y[0],labels))
