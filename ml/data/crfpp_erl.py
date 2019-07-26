import codecs
import json
import os


# 0 install crf++ https://taku910.github.io/crfpp/
# 1 train data in
# 2 test data in
# 3 crf train
# 4 crf test
# 5 submit test

def loadData(file_path):
    data = []
    with open(file_path, 'r', encoding='utf-8') as file:
        for line in file:
            line = line.strip()
            data.append(eval(line))
    return data


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


def get_train_data1(data_path):
    train_data = []
    tmp = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            if line == '\n':
                train_data.append(tmp)
                tmp = []
            else:
                tmp.append(line.strip('\n'))
    train_data = [[j.split('\t') for j in i] for i in train_data]
    train_x = [[token[0] for token in sen] for sen in train_data]
    train_y = [[token[1] for token in sen] for sen in train_data]
    return train_x, train_y


train_x, train_y = get_train_data('train_text.txt')
data = list(zip(train_x, train_y))
train_data = data[:int(len(data) * 0.8)]
test_data = data[int(len(data) * 0.8):]

with codecs.open('erl_train.txt', 'w', encoding='utf-8') as f_out:
    for d in train_data:
        for i in range(len(d[0])):
            if d[0][i] == ' ':
                d[0][i] = '_'
            f_out.write(d[0][i] + '\t' + d[1][i] + '\n')
        f_out.write('\n')

# step 2 test data in
with codecs.open('erl_test.txt', 'w', encoding='utf-8') as f_out:
    for d in (train_data+test_data):
        for i in range(len(d[0])):
            if d[0][i] == ' ':
                d[0][i] = '_'
            f_out.write(d[0][i] + '\n')
        f_out.write('\n')

# 3 crf train
crf_train = "crfpp\crf_learn -f 3 template.txt erl_train.txt erl_model"
os.system(crf_train)

# 4 crf test
crf_test = "crfpp\crf_test -m erl_model erl_test.txt -o erl_result.txt"
os.system(crf_test)

raw_data = loadData('train_pre.json')
data_x, data_y = get_train_data1('erl_result.txt')
result = list(zip(data_x, data_y))
with open('erl_crfpp_train_output.json', 'w', encoding='utf-8') as f:
    for index in range(len(result)):
        mt = set()
        text = raw_data[index]['text']
        labels = result[index][1]
        for i, (s, t) in enumerate(zip(text, labels)):
            if t == 'B-ment':
                ment = s
                offset = i
                for index1, (s1, t1) in enumerate(list(zip(text, labels))[i + 1:]):
                    if t1 == 'I-ment':
                        ment += s1
                    else:
                        break
                mt.add((ment, str(offset)))
        ms = [{'mention': _[0], 'offset': _[1]} for _ in mt]
        line = {'text': text, 'mention_data': ms}
        f.write(json.dumps(line, ensure_ascii=False) + '\n')
