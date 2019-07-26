# -*- coding:UTF-8 -*-
"""
@File    : test.py
@Time    : 2019/5/27 8:32
@Author  : Blue Keroro
"""
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from util import loadDataBase, loadData
import json
if __name__ == '__main__':
    cnt = 0
    data = loadData('data/train.json')
    delete = list()
    for sub in data:
        map = [0 for i in sub['text']]
        length = 0
        for ment in sub['mention_data']:
            length += len(ment['mention'])
            for index in range(int(ment['offset']), int(ment['offset']) + len(ment['mention'])):
                map[index] = 1
        for i in map:
            if i == 1:
                length -= 1
        if length != 0:
            print(sub)
            delete.append(sub)
            cnt += 1
    print("条数：", cnt)  # 722
    for i in delete:
        data.remove(i)
    with open('data/train_pre.json', 'w', encoding='utf-8') as file:
        for i in data:
            file.write(json.dumps(i,ensure_ascii=False) + '\n')
