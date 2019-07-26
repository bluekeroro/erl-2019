# -*- coding:UTF-8 -*-
"""
@File    : util.py
@Time    : 2019/5/19 19:46
@Author  : Blue Keroro
"""

import sys
import os

curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
sys.path.append(rootPath)

from tqdm import tqdm
import json
# import jieba.analyse
# jieba.load_userdict('data/nerDict.txt')


def loadData(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            data.append(eval(line))
    return data


def getAliasfromAbstract(input_text: str):
    key_words = ['简称', '别称', '又称']
    ret = set()
    for key_word in key_words:
        start = 0
        # if key_word in input_text:
        while input_text.find(key_word, start) != -1:
            start = input_text.index(key_word, start) + len(key_word)
            end = start
            for index, ch in enumerate(input_text[start:]):
                if is_Chinese(ch) is not True and str(ch).isalpha() is not True and str(ch).isdigit() is not True:
                    end = index
                    break
            ret.add(input_text[start:start + end])
    return ret


def loadDataBase(file_path,model=0):
    print('加载数据库 start')
    dataByAlias = {}
    dataBySubjectId = {}
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            line = eval(line)
            if line['subject'] not in line['alias']:
                line['alias'].append(line['subject'])
                # raise Exception('subject not in alias',"subject_id:",line["subject_id"])
            # try:
            #     words = getAliasfromAbstract(line['data'][0]['object'])
            #     for word in words:
            #         if word not in line['alias']:
            #             line['alias'].append(word)
            # except Exception:
            #     print('数据格式异常', 'subject_id:', line["subject_id"])
            if line["subject_id"] in dataBySubjectId:
                raise Exception('可能有重复的subject_id', "subject_id:", line["subject_id"])
            dataBySubjectId[line["subject_id"]] = line
            tmp = set()
            for alia in line['alias']:
                tmp.add(alia)
                # alia=alia.replace('《', '').replace('》', '')# 别名去除书名号
                # tmp.add(alia)
                tmp.add(alia.lower())
            line['alias'] = list(tmp)
            for alia in line['alias']:
                if alia not in dataByAlias:
                    dataByAlias[alia] = []
                dataByAlias[alia].append(line)
    if model==1:
        path = os.path.dirname(__file__) if len(os.path.dirname(__file__)) != 0 else '.'
        train = loadData(path + '/' +'bert/data/train_pre.json')
        for l in tqdm(train):
            for i in l['mention_data']:
                if i['mention'] not in dataByAlias and i['kb_id'] in dataBySubjectId:
                    dataByAlias[i['mention']]=[dataBySubjectId[i['kb_id']]]
                if i['mention'] in dataByAlias and i['kb_id'] in dataBySubjectId:
                    for _ in dataByAlias[i['mention']]:
                        if _['subject_id'] ==i['kb_id']:
                            break
                    else:
                        dataByAlias[i['mention']].append(dataBySubjectId[i['kb_id']])
                # elif i['mention'] in dataByAlias and i['kb_id'] in dataBySubjectId and i['kb_id'] not in dataByAlias[i['mention']]:
                #     dataByAlias[i['mention']].append(dataBySubjectId[i['kb_id']])
    print('加载数据库 end')
    return dataByAlias, dataBySubjectId


def is_word(word: str):
    for ch in word:
        if is_Chinese(ch) or str(ch).isdigit() or str(ch).isalpha():
            return True
    return False


def is_Chinese(ch):
    # 中文符号也会返回False
    if '\u4e00' <= ch <= '\u9fff':
        return True
    return False


def get_special_mention(input_text: str):
    special_chars = [['《', '》'], ['[', ']'], ['【', '】'], ['≪', '≫'], ['(', ')'], ['（', '）'], ['{', '}'], ['〖', '〗'],
                     ['『', '』']]
    split_chars = set(['|', '┆', '┇', '︳', '︱', '▕', '‖', '│', '▎'
                          , 'Ⅰ', '▏', '▍', '┋', '།', '¦', '┇', '┆', 'Ⅱ', '|', '▋', '▌', '┊'])
    ret_set = set()
    for special_char in special_chars:
        text_list = input_text.split(special_char[0])
        for t in text_list:
            if special_char[1] in t:
                t = t.split(special_char[1])
                ret_set.add(t[0])
    ret = set()
    for ment in ret_set:
        for split_char in split_chars:
            ret |= set(ment.split(split_char))
    return ret


def get_mention_by_alias(input_text: str, aliasSet):
    ret = set()
    for alia in aliasSet:
        if alia in input_text:
            ret.add(alia)
        # elif alia in input_text.replace('·',''): # 对“·”进行处理
        #     text=input_text.replace('·','')
        #     index=text.index(alia)
        #     cnt=-1
        #     start=-1
        #     end=0
        #     for i in range(len(input_text)):
        #         if input_text[i]!='·':
        #             cnt+=1
        #             if cnt==index:
        #                 start=i
        #             if cnt-index+1==len(alia):
        #                 end=i+1
        #                 break
        #     if start>-1 and end>start:
        #         ret.add(input_text[start:end])
    return ret


def fenci(input_text: str):
    words = jieba.analyse.extract_tags(input_text, topK=20, withWeight=True)
    ret_words = []
    for word in words:
        for ch in word[0]:
            if is_Chinese(ch) or str(ch).isdigit() or str(ch).isalpha():
                ret_words.append(word)
                break
    return ret_words


def save_mention(mentionDict: dict, save_file_path, raw_data_path):
    print("save_mention start")
    data = loadData(raw_data_path)
    with open(save_file_path, 'w', encoding='utf-8') as file:
        for d in tqdm(data):
            d['mention_data'] = []
            if d['text_id'] in mentionDict:
                for mention in mentionDict[d['text_id']]:
                    if mention == '':
                        continue
                    offset = 0
                    while str(d['text']).find(mention, offset) != -1:
                        offset = str(d['text']).index(mention, offset)
                        d['mention_data'].append({'mention': mention, 'offset': str(offset)})
                        offset += 1
            file.write(json.dumps(d,ensure_ascii=False) + '\n')
    print("save_mention end")


def precision_of_special_word():
    data = loadData('data/train.json')
    cnt = 0
    cntSum = 0
    for d in data:
        pred_mention = get_special_mention(d['text'])
        cntSum += len(d['mention_data'])
        ment = set()
        for mention_data in d['mention_data']:
            ment.add(mention_data['mention'])
        cnt += len(pred_mention & ment)
    print("预测mention的准确性：{}%".format(cnt / cntSum * 100))


def fit(dataByAlias, save_file_path, raw_data_path):
    data = loadData(raw_data_path)
    mentionDict = {}
    for d in tqdm(data):
        pred_mention = set()
        # pred_mention = get_special_mention(d['text'])
        pred_mention |= get_mention_by_alias(d['text'], dataByAlias.keys())
        mentionDict[d['text_id']] = pred_mention
    save_mention(mentionDict, save_file_path, raw_data_path)


def merge_mentions(mention_data: list):
    mention_data.sort(key=lambda x: int(x['offset']))
    ret_ments = []
    cnt = 0
    for i in range(len(mention_data)):
        i += cnt
        if i >= len(mention_data):
            break
        mentStr = mention_data[i]['mention']
        while i + 1 < len(mention_data) \
                and int(mention_data[i]['offset']) + len(mentStr) == int(mention_data[i + 1]['offset']):
            mentStr += mention_data[i + 1]['mention']
            cnt += 1
        mention_data[i]['mention'] = mentStr
        ret_ments.append(mention_data[i])
    return ret_ments


def is_contact_char(mention: dict, text: str):
    # if not mention['mention'][-1].encode('UTF-8').isalpha() and :
    #     return False
    if mention['mention'][0].encode('UTF-8').isalpha() \
            and int(mention['offset']) > 0 \
            and text[int(mention['offset']) - 1].encode('UTF-8').isalpha():
        return True
    if mention['mention'][-1].encode('UTF-8').isalpha() \
            and int(mention['offset']) + len(mention['mention']) < len(text) \
            and text[int(mention['offset']) + len(mention['mention'])].encode('UTF-8').isalpha():
        return True
    return False


def process_mentions(mention_data: list, text: str):
    mention_data.sort(key=lambda x: len(x['mention']), reverse=True)
    mention_data.sort(key=lambda x: int(x['offset']))
    delete_ments = []
    # if '品茗施工课堂' in text:
    #     a=1
    for ment in mention_data:
        if is_word(ment['mention']) is not True \
                or len(ment['mention']) < 2 \
                or ment['mention'].isdigit() \
                or ment['mention'].isnumeric() \
                or is_contact_char(ment, text):
            delete_ments.append(ment)
    for ment in delete_ments:
        mention_data.remove(ment)
    if len(mention_data) == 0:
        return mention_data
    map_list = [0 for i in range(int(mention_data[-1]['offset']) + 50)]
    valid_ment = []
    for ment in mention_data:  # 最大匹配模式
        if map_list[int(ment['offset'])] == 0:
            valid_ment.append(ment)
            for i in range(int(ment['offset']), int(ment['offset']) + len(ment['mention'])):
                map_list[i] = 1
    return valid_ment
    # return merge_mentions(valid_ment)


def save_user_word(dataByAlias):
    with open("data/user_word.txt",'w',encoding='utf-8') as file:
        for alia in dataByAlias:
            file.write(alia + '\n')


if __name__ == '__main__':
    pass
    # print('start')
    # from time import time
    #
    # dataByAlias, dataBySubjectId = loadDataBase('data/kb_data',model=1)
    #
    # save_user_word(dataByAlias)
    # start = time()
    # save_train_file_path = 'data/save_train_mention.json'
    # save_dev_file_path = 'data/save_dev_mention.json'
    # train_file_path = 'bert/data/train_pre.json'
    # dev_file_path = 'data/develop.json'
    # print('train fit')
    # fit(dataByAlias, save_train_file_path, train_file_path)
    # print('dev fit')
    # fit(dataByAlias, save_dev_file_path, dev_file_path)
    # print("end", 'use time', time() - start)
