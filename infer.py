# -*- coding: utf-8 -*-

import os
import pickle
import torch
import torch.nn.functional as F

from data_utils import ABSADatesetReader, ABSADataset, Tokenizer, build_embedding_matrix
from models import LSTM, ASGCN, ASCNN
from dependency_graph import dependency_adj_matrix

# 基线方法对比
class Inferer:
    """A simple inference example"""
    def __init__(self, opt):
        self.opt = opt
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
            },
        }
        # 确认是否有该数据集的单词字典
        if os.path.exists(opt.dataset+'_word2idx.pkl'):
            print("loading {0} tokenizer...".format(opt.dataset))
            with open(opt.dataset+'_word2idx.pkl', 'rb') as f:
                 word2idx = pickle.load(f)
                 self.tokenizer = Tokenizer(word2idx=word2idx)
        else:
            print("reading {0} dataset...".format(opt.dataset))
            text = ABSADatesetReader.__read_text__([fname[opt.dataset]['train'], fname[opt.dataset]['test']])
            self.tokenizer = Tokenizer()
            self.tokenizer.fit_on_text(text)
            with open(opt.dataset+'_word2idx.pkl', 'wb') as f:
                 pickle.dump(self.tokenizer.word2idx, f)
        # 构建词嵌入矩阵
        embedding_matrix = build_embedding_matrix(self.tokenizer.word2idx, opt.embed_dim, opt.dataset)
        # 创建模型实例
        self.model = opt.model_class(embedding_matrix, opt).to(opt.device)
        print('loading model {0} ...'.format(opt.model_name))
        # 加载之前保存的模型状态字典
        self.model.load_state_dict(torch.load(opt.state_dict_path))
        #
        # self.model = self.model
        # switch model to evaluation mode
        # 调用模型的 eval() 方法将模型设置为评估模式。在评估模式下，某些特定的层（如 Dropout 和 Batch Normalization）会以测试模式运行，这通常用于模型的推理和评估。
        self.model.eval()
        # 禁用了 PyTorch 的梯度计算功能。
        # 在评估模式下，通常不需要计算梯度，因为这会增加计算负担并且没有必要。
        torch.autograd.set_grad_enabled(False)

    # 用于效果验证
    def evaluate(self, raw_text, aspect):
        # 使用分词器（tokenizer）将 raw_text 转换为序列，并将文本转换为小写
        text_seqs = [self.tokenizer.text_to_sequence(raw_text.lower())]
        aspect_seqs = [self.tokenizer.text_to_sequence(aspect.lower())]
        left_seqs = [self.tokenizer.text_to_sequence(raw_text.lower().split(aspect.lower())[0])]
        # 文本序列转换为 PyTorch 张量（tensor），数据类型为 64 位整数。
        text_indices = torch.tensor(text_seqs, dtype=torch.int64)
        aspect_indices = torch.tensor(aspect_seqs, dtype=torch.int64)
        left_indices = torch.tensor(left_seqs, dtype=torch.int64)
        # 生成依赖图的邻接矩阵，并将其转换为 PyTorch 张量
        dependency_graph = torch.tensor([dependency_adj_matrix(raw_text.lower())])
        # 构建data
        data = {
            'text_indices': text_indices, 
            'aspect_indices': aspect_indices,
            'left_indices': left_indices, 
            'dependency_graph': dependency_graph
        }
        # 创建一个列表 t_inputs，其中包含 data 字典中对应于 self.opt.inputs_cols 的值，并将它们移动到配置选项 opt 指定的设备上。
        t_inputs = [data[col].to(opt.device) for col in self.opt.inputs_cols]
        # 将输入数据传递给模型，获取模型的输出
        t_outputs = self.model(t_inputs)
        # 对模型输出应用 softmax 函数，以获得概率分布，然后将输出转换为 CPU 张量并转换为 NumPy 数组。
        t_probs = F.softmax(t_outputs, dim=-1).cpu().numpy()
        return t_probs


def print_status(mood_status):
    if mood_status==0:
        print("消极")
    elif mood_status==1:
        print("中性")
    elif mood_status==2:
        print("积极")


if __name__ == '__main__':
    # 数据集的名称
    dataset = 'rest14'
    # set your trained models here
    # 其中包含不同模型的状态字典文件路径，这些文件包含了训练后的模型参数
    model_state_dict_paths = {
        'lstm': 'state_dict/lstm_'+dataset+'.pkl',
        'ascnn': 'state_dict/ascnn_'+dataset+'.pkl',
        'asgcn': 'state_dict/asgcn_'+dataset+'.pkl',
    }
    # 模型选择
    model_classes = {
        'lstm': LSTM,
        'ascnn': ASCNN,
        'asgcn': ASGCN,
    }
    # 不同模型需要的输入列
    input_colses = {
        'lstm': ['text_indices'],
        'ascnn': ['text_indices', 'aspect_indices', 'left_indices'],
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
    }
    # 定义一个Option类
    class Option(object): pass
    # 创建实例并复制
    opt = Option()
    opt.model_name = 'asgcn'
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.dataset = dataset
    opt.state_dict_path = model_state_dict_paths[opt.model_name]
    # 嵌入层维度
    opt.embed_dim = 300
    # 隐藏层维度
    opt.hidden_dim = 300
    # 情感极性维度
    opt.polarities_dim = 3
    # 设备选择，优先使用cuda（gpu）
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(torch.cuda.is_available())
    # 依据配置创建对比实例
    inf = Inferer(opt)
    # 执行判定, 0代表消极，1代表中性， 2代表积极
    # 打印预测概率中最大值的索引，即最可能的情感极性
    t_probs = (inf.evaluate('While there \'s a decent menu , it should n\'t take ten minutes to get your drinks and 45 for a menu ..', 'menu'))
    print_status(t_probs.argmax(axis=-1)[0])
    t_probs = (inf.evaluate('The bread is top notch as well ..', 'bread'))
    print_status(t_probs.argmax(axis=-1)[0])
    t_probs = (inf.evaluate('Our waiter was horrible ; so rude and disinterested .', 'waiter'))
    print_status(t_probs.argmax(axis=-1)[0])
