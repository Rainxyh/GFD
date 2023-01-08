import pickle
import csv
import pandas as pd
import json
import os


with open(f'{os.getcwd()}/results/{self.hparams.model_name}/outputs.tsv', 'rb+', newline='') as f:
f = open('C:/Users/Administrator/Desktop/first.pickle', 'rb+')
    info = pickle.load(f)
    # print(info[0])
    # for i in range(len(info)):
    #     info[i].insert(0, 500+i)

# 词向量
with open(r'C:/Users/Administrator/Desktop/vector.tsv', 'w', newline='') as f:
    tsv_w = csv.writer(f, delimiter='\t')
    tsv_w.writerows(info)  # 多行写入
    f.index = [i for i in range(1, 268)]

# vector = pd.read_csv('Vector1142.tsv', sep='\t', header=None)
# vector.index = [i for i in range(1, 436)]
# print(vector)

# 标签
word2index = json.load(open("./dataset/word2index.json", 'r', encoding='utf8'))
for i in word2index:
    with open(r'C:/Users/Administrator/Desktop/index_col.tsv', 'w', newline='') as f:
        tsv_w = csv.writer(f, delimiter='\t')
        tsv_w.writerows([key] for key in word2index)  # 多行写入
# vector = pd.read_csv('Index_col1142.tsv', sep='\t', header=None)
# print(vector)