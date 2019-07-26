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

from keras_contrib import losses
import json
import numpy as np
from random import choice
from keras_bert import load_trained_model_from_checkpoint, Tokenizer
import re, os
import codecs
from keras.callbacks import Callback
from tqdm import tqdm

from bert.data_reduce import get_train_data
from util import loadData, loadDataBase
import numpy as np

maxlen = 100
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

labels = ['O', 'B-ment', 'I-ment']
x, y = get_train_data(path + '/' + 'data/train_text.txt')
y = [[labels.index(tag) for tag in tags] for tags in y]
data = list(zip(x, y))
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
    def __init__(self, data, batch_size=64):
        self.data = data
        self.batch_size = batch_size
        self.steps = len(self.data) // self.batch_size
        if len(self.data) % self.batch_size != 0:
            self.steps += 1

    def __len__(self):
        return self.steps

    def __iter__(self):
        while True:
            idxs = range(len(self.data))
            # np.random.shuffle(idxs)
            X1, X2, Y = [], [], []
            for i in idxs:
                d = self.data[i]
                text = d[0]
                x1, x2 = tokenizer.encode(first=text.lower())
                y = [-2] + d[1] + [-2]
                X1.append(x1)
                X2.append(x2)
                Y.append(y)
                if len(X1) == self.batch_size or i == idxs[-1]:
                    X1 = seq_padding(X1)
                    X2 = seq_padding(X2)
                    Y = seq_padding(Y, padding=-1)
                    Y = np.expand_dims(Y, 2)
                    yield [X1, X2], Y
                    [X1, X2, Y] = [], [], []


class Evaluate(Callback):
    def __init__(self):
        self.F1 = []
        self.best = 0.
        self.best_ment = 0.
        data = loadData(path + '/' + "data/train_pre.json")
        self.test_x = [i['text'] for i in data[int(len(data) * 0.8):]]
        self.test_y = [[(ment['mention'], ment['offset']) for ment in ments['mention_data'] if ment['kb_id'] != 'NIL']
                       for ments in data[int(len(data) * 0.8):]]

    def on_epoch_end(self, epoch, logs=None):
        [f1, precision, recall] = self.evaluate()
        self.F1.append(f1)
        if f1 > self.best:
            self.best = f1
            model.save_weights(path + '/' + 'best_model.weights')
        print('f1: %.4f, precision: %.4f, recall: %.4f, best f1: %.4f\n' % (f1, precision, recall, self.best))
        print("F1 list:", evaluator.F1)

    def evaluate(self):
        A, B, C = 1e-10, 1e-10, 1e-10
        for x, y in tqdm(list(zip(self.test_x, self.test_y))):
            R = set(self.extract_items(x))
            T = set(y)
            A += len(R & T)
            B += len(R)
            C += len(T)
        return [2 * A / (B + C), A / B, A / C]

    def extract_items(self, inputs):
        x1, x2 = tokenizer.encode(first=inputs.lower())
        raw = model.predict([np.array([x1]), np.array([x2])])[0]
        raw = raw[1:1 + len(inputs)]
        result = [np.argmax(row) for row in raw]
        result_tags = [labels[int(i)] for i in result]
        ments = []
        for index, (s, t) in enumerate(zip(inputs, result_tags)):
            if t == 'B-ment':
                ment = s
                offset = index
                for index1, (s1, t1) in enumerate(list(zip(inputs, result_tags))[index + 1:]):
                    if t1 == 'I-ment':
                        ment += s1
                    else:
                        break
                if ment in dataByAlias or ment.lower() in dataByAlias:
                    ments.append((ment, str(offset)))
        return ments


from keras.layers import *
from keras.models import Model
import keras.backend as K
from keras.optimizers import Adam
from keras_contrib.layers import CRF

bert_model = load_trained_model_from_checkpoint(config_path, checkpoint_path)

for l in bert_model.layers:
    l.trainable = True
BiRNN_UNITS = 768  # 768
x1_in = Input(shape=(None,))
x2_in = Input(shape=(None,))
x = bert_model([x1_in, x2_in])
crf = CRF(len(labels), sparse_target=True)
p = crf(x)

model = Model([x1_in, x2_in], p)
model.compile(optimizer=Adam(1e-5), loss=losses.crf_loss, metrics=[crf.accuracy])
model.summary()

train_D = data_generator(train_data)
evaluator = Evaluate()


def pred(text):
    R = evaluator.extract_items(text)
    R = [{'mention': r[0], 'offset': r[1]} for r in R]
    R.sort(key=lambda x: int(x['offset']))
    l = {}
    l['text'] = text
    l['mention_data'] = R
    return l


if __name__ == '__main__':
    model.fit_generator(
        train_D.__iter__(),
        steps_per_epoch=len(train_D),
        epochs=10,
        callbacks=[evaluator]
    )
    print('预测eval:')
    with open('data/eval.json', 'r', encoding='utf-8')as f1, \
            open('data/eval_output_ment.json', 'w', encoding='utf-8') as f2:
        for l in tqdm(f1.readlines()):
            l = json.loads(l)
            R = set(evaluator.extract_items(l['text']))
            R = [{'mention': r[0], 'offset': r[1]} for r in R]
            R.sort(key=lambda x: int(x['offset']))
            l['mention_data'] = R
            f2.write(json.dumps(l, ensure_ascii=False) + '\n')

else:
    model.load_weights(path + '/' + 'model/0.82_ment_best_model.weights')
    evaluator.extract_items("听说这样可以解决web中keras加载调用的问题")
    print('加载权重', path + '/' + 'model/0.82_ment_best_model.weights')

# print('预测train:')
# with open('../bert/data/train_pre.json', 'r', encoding='utf-8')as f1, \
#         open('output_data/train_output_ment.json', 'w', encoding='utf-8') as f2:
#     for l in tqdm(f1.readlines()):
#         l = json.loads(l)
#         R = set(evaluator.extract_items(l['text']))
#         R = [{'mention': r[0], 'offset': r[1]} for r in R]
#         R.sort(key=lambda x: int(x['offset']))
#         l['mention_data'] = R
#         f2.write(json.dumps(l, ensure_ascii=False) + '\n')
# print('预测dev:')
# with open('../data/develop.json', 'r', encoding='utf-8')as f1, \
#         open('output_data/dev_output_ment.json', 'w', encoding='utf-8') as f2:
#     for l in tqdm(f1.readlines()):
#         l = json.loads(l)
#         R = set(evaluator.extract_items(l['text']))
#         R = [{'mention': r[0], 'offset': r[1]} for r in R]
#         R.sort(key=lambda x: int(x['offset']))
#         l['mention_data'] = R
#         f2.write(json.dumps(l, ensure_ascii=False) + '\n')
# print('预测eval:')
# with open('../data/eval722.json', 'r', encoding='utf-8')as f1, \
#         open('output_data/eval_output_ment.json', 'w', encoding='utf-8') as f2:
#     for l in tqdm(f1.readlines()):
#         l = json.loads(l)
#         R = set(evaluator.extract_items(l['text']))
#         R = [{'mention': r[0], 'offset': r[1]} for r in R]
#         R.sort(key=lambda x: int(x['offset']))
#         l['mention_data'] = R
#         f2.write(json.dumps(l, ensure_ascii=False) + '\n')
