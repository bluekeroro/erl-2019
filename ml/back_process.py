# -*- coding:UTF-8 -*-
"""
@File    : back_process.py
@Time    : 2019/7/14 18:21
@Author  : Blue Keroro
"""
import json
import re

from tqdm import tqdm

from util import loadData, is_word, loadDataBase
import numpy as np
from random import choice, choices


def process(data):
    for i, l in enumerate(tqdm(data)):
        md = set()
        text = l['text']
        arr = np.zeros(len(text))
        for ment in l['mention_data']:
            arr[int(ment['offset']):int(ment['offset']) + len(ment['mention'])] = 1
        for ment in l['mention_data']:
            if (not is_word(ment['mention'])):
                print(l)
                print(ment['mention'])
                print('')
                continue
            md.add((ment['kb_id'], ment['mention'], ment['offset']))

            index = 0
            while text.find(ment['mention'], index) != -1:
                index = text.find(ment['mention'], index)
                if len(np.nonzero(arr[index:index + len(ment['mention'])])[0]) == 0:
                    md.add((ment['kb_id'], ment['mention'], str(index)))
                    arr[index:index + len(ment['mention'])] = 1
                index += 1
        l['mention_data'] = sorted([{'kb_id': _[0], 'mention': _[1], 'offset': _[2]} for _ in md],
                                   key=lambda x: int(x['offset']))
    return data


dataByAlias, dataBySubjectId = loadDataBase('data/kb_data')


def merge(data):
    for l in data:
        t = l['text']
        d = {int(m['offset']): m['mention'] for m in l['mention_data']}
        mt = {(_['kb_id'], _['mention'], _['offset']) for _ in l['mention_data']}
        for s in d:
            m = d[s]
            s1 = s + len(m)
            if s1 in d and (m + d[s1]) in dataByAlias:
                mt = {_ for _ in mt if not ((_[1] == d[s1] and _[2] == str(s1)) or (_[1] == m and _[2] == str(s)))}
                mt.add(('NIL', m + d[s1], str(s)))
        l['mention_data'] = [{"kb_id": _[0], "mention": _[1], 'offset': _[2]} for _ in mt]


if __name__ == '__main__':
    pass
