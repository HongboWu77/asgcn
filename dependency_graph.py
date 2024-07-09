# -*- coding: utf-8 -*-

import numpy as np
import spacy
import pickle

from spacy.tokens import Doc

class WhitespaceTokenizer(object):
    def __init__(self, vocab):
        self.vocab = vocab

    def __call__(self, text):
        words = text.split()
        # All tokens 'own' a subsequent space character in this tokenizer
        spaces = [True] * len(words)
        return Doc(self.vocab, words=words, spaces=spaces)

nlp = spacy.load('en_core_web_sm')
nlp.tokenizer = WhitespaceTokenizer(nlp.vocab)

# 构建句子的依赖邻接矩阵
def dependency_adj_matrix(text):
    # https://spacy.io/docs/usage/processing-text
    tokens = nlp(text)
    words = text.split()
    matrix = np.zeros((len(words), len(words))).astype('float32')
    assert len(words) == len(list(tokens))

    for token in tokens:
        matrix[token.i][token.i] = 1
        for child in token.children:
            #  "The cat sat on the mat" 中，"sat" 是谓语，"cat" 是主语，"on the mat" 是介词短语作为状语修饰 "sat"。
            #  "sat" 的子代可能包括 "cat" 和 "on the mat"。
            matrix[token.i][child.i] = 1
            matrix[child.i][token.i] = 1

    return matrix

def process(filename):
    fin = open(filename, 'r', encoding='utf-8', newline='\n', errors='ignore')
    # 读取所有行
    lines = fin.readlines()
    fin.close()
    idx2graph = {}
    fout = open(filename+'.graph', 'wb')
    for i in range(0, len(lines), 3):  # 三行为一次循环
        text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
        aspect = lines[i + 1].lower().strip()
        # 将第i+1行的数据替换到第i行的$T$处，并传入方法中构建该句子的依赖邻接矩阵
        adj_matrix = dependency_adj_matrix(text_left+' '+aspect+' '+text_right)
        # 存入idx2graph
        idx2graph[i] = adj_matrix
    # 转存到.graph文件中
    pickle.dump(idx2graph, fout)        
    fout.close()

if __name__ == '__main__':
    process('./datasets/acl-14-short-data/train.raw')
    process('./datasets/acl-14-short-data/test.raw')
    process('./datasets/semeval14/restaurant_train.raw')
    process('./datasets/semeval14/restaurant_test.raw')
    process('./datasets/semeval14/laptop_train.raw')
    process('./datasets/semeval14/laptop_test.raw')
    process('./datasets/semeval15/restaurant_train.raw')
    process('./datasets/semeval15/restaurant_test.raw')
    process('./datasets/semeval16/restaurant_train.raw')
    process('./datasets/semeval16/restaurant_test.raw')