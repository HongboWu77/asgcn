# -*- coding: utf-8 -*-


import torch
import torch.nn as nn
import numpy as np

class DynamicLSTM(nn.Module):

    def __init__(self, input_size, hidden_size, num_layers=1, bias=True, batch_first=True, dropout=0,
                 bidirectional=False, only_use_last_hidden_state=False, rnn_type = 'LSTM'):
        """
        LSTM which can hold variable length sequence, use like TensorFlow's RNN(input, length...).

        :param input_size:The number of expected features in the input x
        :param hidden_size:The number of features in the hidden state h
        :param num_layers:Number of recurrent layers.
        :param bias:If False, then the layer does not use bias weights b_ih and b_hh. Default: True
        :param batch_first:If True, then the input and output tensors are provided as (batch, seq, feature)
        :param dropout:If non-zero, introduces a dropout layer on the outputs of each RNN layer except the last layer
        :param bidirectional:If True, becomes a bidirectional RNN. Default: False
        :param rnn_type: {LSTM, GRU, RNN}
        input_size: 输入特征的数量。
        hidden_size: 隐藏状态的特征数量。
        num_layers: 循环层的数量，默认为 1。
        bias: 是否使用偏置权重，默认为 True。
        batch_first: 输入和输出张量的格式是否为 (batch, seq, feature)，默认为 True。
        dropout: 除最后一层外，每层输出上的丢弃层概率，默认为 0。
        bidirectional: 是否为双向 RNN，默认为 False。
        only_use_last_hidden_state: 是否仅使用最后一个隐藏状态，默认为 False。
        rnn_type: RNN 类型，可以是 'LSTM', 'GRU', 或 'RNN'。
        """
        super(DynamicLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.bias = bias
        self.batch_first = batch_first
        self.dropout = dropout
        self.bidirectional = bidirectional
        self.only_use_last_hidden_state = only_use_last_hidden_state
        self.rnn_type = rnn_type
        
        if self.rnn_type == 'LSTM': 
            self.RNN = nn.LSTM(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)  
        elif self.rnn_type == 'GRU':
            self.RNN = nn.GRU(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        elif self.rnn_type == 'RNN':
            self.RNN = nn.RNN(
                input_size=input_size, hidden_size=hidden_size, num_layers=num_layers,
                bias=bias, batch_first=batch_first, dropout=dropout, bidirectional=bidirectional)
        

    def forward(self, x, x_len, h0=None):
        """
        sequence -> sort -> pad and pack ->process using RNN -> unpack ->unsort

        :param x: sequence embedding vectors
        :param x_len: numpy/tensor list
        :return:
        """
        """sort"""
        x_sort_idx = torch.argsort(-x_len)
        x_unsort_idx = torch.argsort(x_sort_idx).long()
        x_len = x_len[x_sort_idx]
        x = x[x_sort_idx.long()]
        """pack"""
        x_emb_p = torch.nn.utils.rnn.pack_padded_sequence(x, x_len, batch_first=self.batch_first)
        
        # process using the selected RNN
        if self.rnn_type == 'LSTM':
            if h0 is None: 
                out_pack, (ht, ct) = self.RNN(x_emb_p, None)
            else:
                out_pack, (ht, ct) = self.RNN(x_emb_p, (h0, h0))
        else: 
            if h0 is None:
                out_pack, ht = self.RNN(x_emb_p, None)
            else:
                out_pack, ht = self.RNN(x_emb_p, h0)
            ct = None
        """unsort: h"""
        ht = torch.transpose(ht, 0, 1)[
            x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
        ht = torch.transpose(ht, 0, 1)

        if self.only_use_last_hidden_state:
            return ht
        else:
            """unpack: out"""
            out = torch.nn.utils.rnn.pad_packed_sequence(out_pack, batch_first=self.batch_first)  # (sequence, lengths)
            out = out[0]  #
            out = out[x_unsort_idx]
            """unsort: out c"""
            if self.rnn_type =='LSTM':
                ct = torch.transpose(ct, 0, 1)[
                    x_unsort_idx]  # (num_layers * num_directions, batch, hidden_size) -> (batch, ...)
                ct = torch.transpose(ct, 0, 1)

            return out, (ht, ct)
