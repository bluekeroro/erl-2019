# -*- coding:UTF-8 -*-
"""
@File    : my_bert_crf_model.py
@Time    : 2019/7/3 1:11
@Author  : Blue Keroro
"""
import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)
import json
import numpy as np
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer, get_model
import re, os
import codecs
from keras.callbacks import Callback
from tqdm import tqdm

from bert.data_reduce import get_train_data
from util import loadData, loadDataBase
import numpy as np

path = os.path.dirname(__file__) if len(os.path.dirname(__file__)) != 0 else '.'
config_path = path + '/' + '../bert/bertModel/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = path + '/' + '../bert/bertModel/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = path + '/' + '../bert/bertModel/chinese_L-12_H-768_A-12/vocab.txt'

token_dict = {}

with codecs.open(dict_path, 'r', 'utf8') as reader:
    for line in reader:
        token = line.strip()
        token_dict[token] = len(token_dict)


class OurTokenizer(Tokenizer):
    def _tokenize(self, text):
        R = []
        for c in text:
            if c in self._token_dict:
                R.append(c)
            elif self._is_space(c):
                R.append('[unused1]')  # space类用未经训练的[unused1]表示
            else:
                R.append('[UNK]')  # 剩余的字符是[UNK]
        return R


tokenizer = OurTokenizer(token_dict)
char_size = 512  # 768
data = loadData(path + '/' + 'data/train_pre.json')
# data=[ for line in data]
train_data = data[:int(len(data) * 0.8)]
valid_data = data[int(len(data) * 0.8):]
dataByAlias, dataBySubjectId = loadDataBase(path + '/' + 'data/kb_data')


def seq_padding(X, padding=0):
    L = [len(x) for x in X]
    ML = max(L)
    return np.array([
        np.concatenate([x, [padding] * (ML - len(x))]) if len(x) < ML else x for x in X
    ])


class data_generator:
    def __init__(self, data, batch_size=4):
        self.data = data
        self.batch_size = batch_size
        length = 0
        for i in range(len(self.data)):
            for ment in self.data[i]['mention_data']:
                if ment['mention'].lower() not in dataByAlias or ment['kb_id'] == 'NIL' or ment[
                    'kb_id'] not in dataBySubjectId:
                    continue
                length += 2 if len(dataByAlias[ment['mention'].lower()]) > 1 else 1
        self.steps = length // self.batch_size
        if length % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = range(len(self.data))
            # np.random.shuffle(idxs)
            X1, X2, S, Y = [], [], [], []
            for i in idxs:
                for ment in self.data[i]['mention_data']:
                    text = data[i]['text']
                    if ment['mention'].lower() not in dataByAlias or ment['kb_id'] == 'NIL' or ment[
                        'kb_id'] not in dataBySubjectId:
                        continue
                    for l in range(2):
                        if l == 0:
                            l = choice(dataByAlias[ment['mention'].lower()])
                            while l['subject_id'] == ment['kb_id'] and len(dataByAlias[ment['mention'].lower()]) > 1:
                                l = choice(dataByAlias[ment['mention'].lower()])
                        elif len(dataByAlias[ment['mention'].lower()]) > 1:
                            l = dataBySubjectId[ment['kb_id']]
                        if l == 1 or l == 2:
                            continue
                        subject_desc = '\n'.join(l['type'])
                        subject_desc += '\n\n' + '\n'.join(
                            [_['predicate'] + ':' + _['object'] for _ in l['data']]).lower()
                        x1, x2 = tokenizer.encode(first=text.lower() + '\n\n' + subject_desc, max_len=char_size)
                        s1 = np.zeros(len(text) + 2)
                        s1[int(ment['offset']) + 1: int(ment['offset']) + 1 + len(ment['mention'])] = 1
                        if ment['kb_id'] == l['subject_id']:
                            y = [1]
                        else:
                            y = [0]
                        X1.append(x1)
                        X2.append(x2)
                        S.append(s1)
                        Y.append(y)
                        if len(X1) == self.batch_size or i == idxs[-1]:
                            X1 = seq_padding(X1, padding=-1)
                            X2 = seq_padding(X2, padding=-1)
                            S = seq_padding(S, padding=-1)
                            Y = seq_padding(Y, padding=-1)
                            yield [X1, X2], Y
                            [X1, X2, S, Y] = [], [], [], []
                            # evaluator.evaluate()


class Evaluate(Callback):
    def __init__(self):
        self.F1 = []
        self.best = 0.
        self.best_ment = 0.
        self.test_x = valid_data
        self.test_y = [[(ment['mention'], ment['offset'], ment['kb_id']) for ment in ments['mention_data'] if
                        ment['kb_id'] != 'NIL' and ment['kb_id'] in dataBySubjectId]
                       for ments in valid_data]

    def on_epoch_end(self, epoch, logs=None):
        [f1, precision, recall], [f1_ment, precision_ment, recall_ment] = self.evaluate()
        self.F1.append(f1)
        print("F1 list:", evaluator.F1)
        if f1 > self.best:
            self.best = f1
            t_model.save_weights(path + '/' + 'best_model.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))
        if f1_ment > self.best_ment:
            self.best_ment = f1_ment
        print('实体提取：f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (
            f1_ment, precision_ment, recall_ment, self.best_ment))

    def evaluate(self):
        A, B, C = 1e-10, 1e-10, 1e-10
        A1, B1, C1 = 1e-10, 1e-10, 1e-10
        for x, y in tqdm(list(zip(self.test_x, self.test_y))):
            R = set(self.extract_items(x))
            T = set(y)
            A += len(R & T)
            B += len(R)
            C += len(T)
            R1 = set([(_[0], _[1]) for _ in R])
            T1 = set([(_[0], _[1]) for _ in T])
            A1 += len(R1 & T1)
            B1 += len(R1)
            C1 += len(T1)
        return [2 * A / (B + C), A / B, A / C], [2 * A1 / (B1 + C1), A1 / B1, A1 / C1]

    def extract_items(self, inputs):
        rtn = []
        for ment in inputs['mention_data']:
            X1, X2, S = [], [], []
            kb_ids = []
            text = inputs['text']
            if ment['mention'].lower() not in dataByAlias:
                continue
            for l in dataByAlias[ment['mention'].lower()]:
                subject_desc = '\n'.join(l['type'])
                subject_desc += '\n\n' + '\n'.join([_['predicate'] + ':' + _['object'] for _ in l['data']]).lower()
                # x1, x2 = tokenizer.encode(first=text, second=subject_desc)
                x1, x2 = tokenizer.encode(first=text.lower() + '\n\n' + subject_desc, max_len=char_size)
                s1 = np.zeros(len(text) + 2)
                s1[int(ment['offset']) + 1: int(ment['offset']) + 1 + len(ment['mention'])] = 1
                X1.append(x1)
                X2.append(x2)
                S.append(s1)
                kb_ids.append(l['subject_id'])
            X1 = seq_padding(X1, padding=-1)
            X2 = seq_padding(X2, padding=-1)
            S = seq_padding(S, padding=-1)
            scores = t_model.predict([X1, X2])[:, 0]
            index = np.argmax(scores)
            rtn.append((ment['mention'], ment['offset'], kb_ids[index]))
        return rtn


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras_contrib.layers import CRF

bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

for l in bert_model.layers:
    l.trainable = True

x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))
s_in = Input(shape=(None,))
y_in = Input(shape=(None,))
x1, x2, s, y = x1_in, x2_in, s_in, y_in
x = bert_model([x1, x2])
x = Lambda(lambda x: x[:, 0])(x)
pt = Dense(1, activation='sigmoid')(x)
t_model = Model([x1_in, x2_in], pt)
t_model.compile(
    loss='binary_crossentropy',
    optimizer=Adam(1e-5),  # 用足够小的学习率
    metrics=['accuracy']
)
t_model.summary()
train_D = data_generator(train_data)
evaluator = Evaluate()


def pred(data):
    R = evaluator.extract_items(data)  # [{'kb_id': r[2], 'mention': r[0], 'offset': r[1]} for r in R]
    rtn = []
    for r in R:
        d = {}
        d['mention'] = r[0]
        d['content'] = dataBySubjectId[r[2]]
        rtn.append(d)
    return rtn


if __name__ == '__main__':
    t_model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=4,
        callbacks=[evaluator]
    )
    print('预测eval:')
    with open('data/eval_output_ment.json', 'r', encoding='utf-8')as f1, \
            open('data/eval_output.json', 'w', encoding='utf-8') as f2:
        for l in tqdm(f1.readlines()):
            l = json.loads(l)
            R = set(evaluator.extract_items(l))
            R = [{'kb_id': r[2], 'mention': r[0], 'offset': r[1]} for r in R]
            R.sort(key=lambda x: int(x['offset']))
            l['mention_data'] = R
            f2.write(json.dumps(l, ensure_ascii=False) + '\n')

else:
    t_model.load_weights(path + '/' + 'model/0.8949_findId_best_model.weights')
    # 听说这样可以解决web中keras加载调用的问题
    evaluator.extract_items({'text': "南京南站:坐高铁在南京南站下。南京南站", 'mention_data': [{'mention': '南京南站', 'offset': '0'}]})

# print('预测eval:')
# with open('../data/eval_output24_backprocess_merge.json', 'r', encoding='utf-8')as f1, \
#         open('output_data/eval_output24_backprocess_merge_findId.json', 'w', encoding='utf-8') as f2:
#     for l in tqdm(f1.readlines()):
#         try:
#             l = json.loads(l.strip())
#         except Exception as e:
#             print(str(l))
#             raise e
#         m1 = [_ for _ in l['mention_data'] if _['kb_id'] == 'NIL']
#         m2 = [_ for _ in l['mention_data'] if _['kb_id'] != 'NIL']
#         l['mention_data']=m1
#         R = set(evaluator.extract_items(l))
#         R = [{'kb_id': r[2], 'mention': r[0], 'offset': r[1]} for r in R]+m2
#         R.sort(key=lambda x: int(x['offset']))
#         l['mention_data'] = R
#         f2.write(json.dumps(l, ensure_ascii=False) + '\n')
