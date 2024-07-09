# -*- coding: utf-8 -*-

import os
import pickle
import numpy as np

# 从一个文件中加载预训练的词向量
def load_word_vec(path, word2idx=None, embed_dim=300):
    fin = open(path, 'r', encoding='utf-8', newline='\n', errors='ignore')
    word_vec = {}
    for line in fin:
        tokens = line.rstrip().split()
        #
        word, vec = ' '.join(tokens[:-embed_dim]), tokens[-embed_dim:]
        if word in word2idx.keys():
            word_vec[word] = np.asarray(vec, dtype='float32')
    return word_vec

# 用于构建一个词嵌入矩阵
def build_embedding_matrix(word2idx, embed_dim, type):
    embedding_matrix_file_name = '{0}_{1}_embedding_matrix.pkl'.format(str(embed_dim), type)
    # 如果有该数据集的词嵌入矩阵，那么直接加载到程序中来
    if os.path.exists(embedding_matrix_file_name):
        print('loading embedding_matrix:', embedding_matrix_file_name)
        embedding_matrix = pickle.load(open(embedding_matrix_file_name, 'rb'))
    # 如果没有该数据集的词嵌入矩阵，那么就生成该数据集的词嵌入矩阵
    else:
        print('loading word vectors ...')
        embedding_matrix = np.zeros((len(word2idx), embed_dim))  # idx 0 and 1 are all-zeros
        # 为索引 1 的行生成一个小的随机向量，这通常是为了确保模型有一个好的初始状态
        embedding_matrix[1, :] = np.random.uniform(-1/np.sqrt(embed_dim), 1/np.sqrt(embed_dim), (1, embed_dim))
        # 设置预训练词向量文件的路径
        fname = './glove/glove.840B.300d.txt'
        # 调用 load_word_vec 函数加载词向量
        word_vec = load_word_vec(fname, word2idx=word2idx, embed_dim=embed_dim)
        #
        print('building embedding_matrix:', embedding_matrix_file_name)
        for word, i in word2idx.items():
            vec = word_vec.get(word)
            if vec is not None:
                # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = vec
        pickle.dump(embedding_matrix, open(embedding_matrix_file_name, 'wb'))
    return embedding_matrix

# 将文本数据转换为可用于机器学习模型的数值序列，用于管理数据集的词汇字典
class Tokenizer(object):
    def __init__(self, word2idx=None):
        if word2idx is None:
            self.word2idx = {}
            self.idx2word = {}
            self.idx = 0
            self.word2idx['<pad>'] = self.idx
            self.idx2word[self.idx] = '<pad>'
            self.idx += 1
            self.word2idx['<unk>'] = self.idx
            self.idx2word[self.idx] = '<unk>'
            self.idx += 1
        else:
            self.word2idx = word2idx
            self.idx2word = {v:k for k,v in word2idx.items()}

    # 用于根据给定文本更新单词到索引的映射
    def fit_on_text(self, text):
        text = text.lower()
        words = text.split()
        for word in words:
            if word not in self.word2idx:
                self.word2idx[word] = self.idx
                self.idx2word[self.idx] = word
                self.idx += 1

    # 用于将文本转换为数值序列
    def text_to_sequence(self, text):
        text = text.lower()
        words = text.split()
        unknownidx = 1
        sequence = [self.word2idx[w] if w in self.word2idx else unknownidx for w in words]
        if len(sequence) == 0:
            sequence = [0]
        return sequence


# 用于处理和访问某个特定数据集
class ABSADataset(object):
    def __init__(self, data):
        self.data = data

    def __getitem__(self, index):
        return self.data[index]

    def __len__(self):
        return len(self.data)


#
class ABSADatesetReader:
    # 用于从一个或多个文件中读取文本数据，并将其组合成单个字符串
    @staticmethod
    def __read_text__(fnames):
        text = ''
        for fname in fnames:
            fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
            lines = fin.readlines()
            fin.close()
            for i in range(0, len(lines), 3):
                text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
                aspect = lines[i + 1].lower().strip()
                text_raw = text_left + " " + aspect + " " + text_right
                text += text_raw + " "
        return text

    # 读取数据集中的数据
    @staticmethod
    def __read_data__(fname, tokenizer):
        # 获取对应数据集的raw，graph，tree
        fin = open(fname, 'r', encoding='utf-8', newline='\n', errors='ignore')
        lines = fin.readlines()
        fin.close()
        fin = open(fname+'.graph', 'rb')
        idx2graph = pickle.load(fin)
        fin.close()
        fin = open(fname+'.tree', 'rb')
        idx2tree = pickle.load(fin)
        fin.close()

        # 建立空列表，用来存储全部数据
        all_data = []
        for i in range(0, len(lines), 3): # 同构建graph和tree时一样，3行一次循环
            # 对数据集内容进行处理，得到需要的数据
            text_left, _, text_right = [s.lower().strip() for s in lines[i].partition("$T$")]
            aspect = lines[i + 1].lower().strip()
            polarity = lines[i + 2].strip()

            # 进一步处理得到数据，将字符串全部处理为数值序列
            text_indices = tokenizer.text_to_sequence(text_left + " " + aspect + " " + text_right)
            context_indices = tokenizer.text_to_sequence(text_left + " " + text_right)
            aspect_indices = tokenizer.text_to_sequence(aspect)
            left_indices = tokenizer.text_to_sequence(text_left)
            polarity = int(polarity)+1
            dependency_graph = idx2graph[i]
            dependency_tree = idx2tree[i]

            # 构建data
            data = {
                'text_indices': text_indices,
                'context_indices': context_indices,
                'aspect_indices': aspect_indices,
                'left_indices': left_indices,
                'polarity': polarity,
                'dependency_graph': dependency_graph,
                'dependency_tree': dependency_tree,
            }
            # 将data放入all_data
            all_data.append(data)

        # 返回全部数据
        return all_data

    def __init__(self, dataset='twitter', embed_dim=300):
        print("preparing {0} dataset ...".format(dataset))
        fname = {
            'twitter': {
                'train': './datasets/acl-14-short-data/train.raw',
                'test': './datasets/acl-14-short-data/test.raw'
            },
            'rest14': {
                'train': './datasets/semeval14/restaurant_train.raw',
                'test': './datasets/semeval14/restaurant_test.raw'
            },
            'lap14': {
                'train': './datasets/semeval14/laptop_train.raw',
                'test': './datasets/semeval14/laptop_test.raw'
            },
            'rest15': {
                'train': './datasets/semeval15/restaurant_train.raw',
                'test': './datasets/semeval15/restaurant_test.raw'
            },
            'rest16': {
                'train': './datasets/semeval16/restaurant_train.raw',
                'test': './datasets/semeval16/restaurant_test.raw'
            }
        }
        # 首先读取测试集和训练集的所有文本，将其转化为一段字符串
        text = ABSADatesetReader.__read_text__([fname[dataset]['train'], fname[dataset]['test']])
        # 如果已经存在该数据集的单词字典，那么直接将文件加载到程序中来
        if os.path.exists(dataset+'_word2idx.pkl'):
            print("loading {0} tokenizer...".format(dataset))
            with open(dataset+'_word2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 tokenizer = Tokenizer(word2idx=word2idx)
        # 如果不存在该数据集的单词字典，那么需要根据text的内容生成该数据集的单词字典，并将它持久化到文件中
        else:
            tokenizer = Tokenizer()
            tokenizer.fit_on_text(text)
            with open(dataset+'_word2idx.pkl', 'wb') as f:
                 pickle.dump(tokenizer.word2idx, f)
        # 根据该数据集的单词字典，来进行词嵌入矩阵的构建
        self.embedding_matrix = build_embedding_matrix(tokenizer.word2idx, embed_dim, dataset)
        self.train_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['train'], tokenizer))
        self.test_data = ABSADataset(ABSADatesetReader.__read_data__(fname[dataset]['test'], tokenizer))
    