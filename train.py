# -*- coding: utf-8 -*-

import os
import math
import argparse
import random
import numpy
import torch
import torch.nn as nn
from bucket_iterator import BucketIterator
from sklearn import metrics
from data_utils import ABSADatesetReader
from models import LSTM, ASCNN, ASGCN, ASTCN

class Instructor:
    def __init__(self, opt):
        self.opt = opt
        # 获得数据集读取实例
        absa_dataset = ABSADatesetReader(dataset=opt.dataset, embed_dim=opt.embed_dim)
        # 得到训练集
        self.train_data_loader = BucketIterator(data=absa_dataset.train_data, batch_size=opt.batch_size, shuffle=True)
        # 得到测试集
        self.test_data_loader = BucketIterator(data=absa_dataset.test_data, batch_size=opt.batch_size, shuffle=False)
        # 创建模型实例
        self.model = opt.model_class(absa_dataset.embedding_matrix, opt).to(opt.device)
        # 调用方法来打印训练参数和配置。
        self._print_args()
        # 初始化一个变量来存储全局 F1 分数
        self.global_f1 = 0.
        # 检查 CUDA（GPU）是否可用。
        if torch.cuda.is_available():
            # 如果 CUDA 可用，打印已分配的 CUDA 内存。
            print('cuda memory allocated:', torch.cuda.memory_allocated(device=opt.device.index))

    # 打印模型的参数和配置选项
    def _print_args(self):
        # 初始化两个变量，分别用于计数可训练参数和不可训练参数的数量
        n_trainable_params, n_nontrainable_params = 0, 0
        # 遍历模型的所有参数
        for p in self.model.parameters():
            # 计算每个参数的总元素数量
            n_params = torch.prod(torch.tensor(p.shape)).item()
            # 检查参数是否需要梯度，即是否是可训练的
            if p.requires_grad:
                n_trainable_params += n_params
            else:
                n_nontrainable_params += n_params
        print('n_trainable_params: {0}, n_nontrainable_params: {1}'.format(n_trainable_params, n_nontrainable_params))
        print('> training arguments:')
        for arg in vars(self.opt):
            print('>>> {0}: {1}'.format(arg, getattr(self.opt, arg)))

    # 确保模型的所有可训练参数都使用适当的初始化方法进行初始化
    def _reset_params(self):
        #  遍历模型 self.model 中的所有参数
        for p in self.model.parameters():
            # 检查参数 p 是否需要梯度，即是否是可训练的参数
            if p.requires_grad:
                # 检查参数的形状是否多于一个维度，即参数是否是一个权重矩阵（通常权重矩阵有多个维度）
                if len(p.shape) > 1:
                    self.opt.initializer(p)
                else:
                    stdv = 1. / math.sqrt(p.shape[0])
                    torch.nn.init.uniform_(p, a=-stdv, b=stdv)

    # 训练并评估模型
    def _train(self, criterion, optimizer):
        # 用来存储在训练过程中遇到的最高的测试准确率和 F1 分数
        max_test_acc = 0
        max_test_f1 = 0
        # 初始化全局步骤计数器 global_step，用于跟踪训练过程中的步骤数
        global_step = 0
        # 初始化 continue_not_increase 计数器，用于跟踪模型性能在连续多个 epoch 中没有提升的次数
        continue_not_increase = 0
        for epoch in range(self.opt.num_epoch):
            print('>' * 100)
            print('epoch: ', epoch)
            # 用来计数正确的预测次数和总的预测次数
            n_correct, n_total = 0, 0
            # 用于标记在当前 epoch 中模型性能是否有所提升
            increase_flag = False
            for i_batch, sample_batched in enumerate(self.train_data_loader):
                global_step += 1

                # switch model to training mode, clear gradient accumulators
                # 将模型设置为训练模式
                self.model.train()
                # 清除优化器中的梯度累积
                optimizer.zero_grad()
                #
                inputs = [sample_batched[col].to(self.opt.device) for col in self.opt.inputs_cols]
                targets = sample_batched['polarity'].to(self.opt.device)
                outputs = self.model(inputs)
                # 计算输出和目标之间的损失
                loss = criterion(outputs, targets)
                # 进行反向传播，计算梯度
                loss.backward()
                # 根据梯度更新模型参数
                optimizer.step()
                # 如果全局步骤数是 self.opt.log_step 的倍数
                if global_step % self.opt.log_step == 0:
                    # 更新正确预测的次数
                    n_correct += (torch.argmax(outputs, -1) == targets).sum().item()
                    # 更新总预测的次数
                    n_total += len(outputs)
                    # 计算当前的训练准确率
                    train_acc = n_correct / n_total
                    # 调用 _evaluate_acc_f1 方法评估模型在测试集上的准确率和 F1 分数
                    test_acc, test_f1 = self._evaluate_acc_f1()
                    # 检查测试准确率是否超过了之前记录的最高准确率
                    if test_acc > max_test_acc:
                        max_test_acc = test_acc
                    # 检查测试 F1 分数是否超过了之前记录的最高 F1 分数
                    if test_f1 > max_test_f1:
                        increase_flag = True
                        max_test_f1 = test_f1
                        # 如果配置选项 save 为 True 并且测试 F1 分数超过了全局记录的 F1 分数
                        if self.opt.save and test_f1 > self.global_f1:
                            self.global_f1 = test_f1
                            # 保存模型的状态字典到文件
                            torch.save(self.model.state_dict(), 'state_dict/'+self.opt.model_name+'_'+self.opt.dataset+'.pkl')
                            print('>>> best model saved.')
                    # 打印当前批次的损失、训练准确率、测试准确率和测试 F1 分数
                    print('loss: {:.4f}, acc: {:.4f}, test_acc: {:.4f}, test_f1: {:.4f}'.format(loss.item(), train_acc, test_acc, test_f1))
            if increase_flag == False:
                continue_not_increase += 1
                # 如果模型性能连续五轮没有提升
                if continue_not_increase >= 5:
                    print('early stop.')
                    break
            else:
                continue_not_increase = 0    
        return max_test_acc, max_test_f1

    # 用于评估模型在测试集上的准确率和 F1 分数
    def _evaluate_acc_f1(self):
        # switch model to evaluation mode
        # 将模型设置为评估模式，这将禁用模型中的某些特定层的行为，比如丢弃层（Dropout）
        self.model.eval()
        # 初始化测试集上正确预测的样本数 n_test_correct 和总样本数 n_test_total
        n_test_correct, n_test_total = 0, 0
        # 初始化测试集上正确预测的样本数 n_test_correct 和总样本数 n_test_total
        t_targets_all, t_outputs_all = None, None
        # 禁用梯度计算
        with torch.no_grad():
            for t_batch, t_sample_batched in enumerate(self.test_data_loader):
                t_inputs = [t_sample_batched[col].to(opt.device) for col in self.opt.inputs_cols]
                t_targets = t_sample_batched['polarity'].to(opt.device)
                t_outputs = self.model(t_inputs)
                # 更新测试集上正确预测的样本数
                n_test_correct += (torch.argmax(t_outputs, -1) == t_targets).sum().item()
                # 更新测试集的总样本数
                n_test_total += len(t_outputs)
                # 检查 t_targets_all 是否为 None
                if t_targets_all is None:
                    # 初始化它
                    t_targets_all = t_targets
                    t_outputs_all = t_outputs
                else:
                    # 将当前批次的目标和输出与之前累积的目标和输出进行拼接
                    t_targets_all = torch.cat((t_targets_all, t_targets), dim=0)
                    t_outputs_all = torch.cat((t_outputs_all, t_outputs), dim=0)
        # 计算测试集的准确率
        test_acc = n_test_correct / n_test_total
        # 使用 f1_score 函数计算 F1 分数
        # 'macro' 平均是计算每个类别的 F1 分数，然后计算它们的平均值，这不考虑类别的支持度（样本数量）
        f1 = metrics.f1_score(t_targets_all.cpu(), torch.argmax(t_outputs_all, -1).cpu(), labels=[0, 1, 2], average='macro')
        return test_acc, f1

    # 执行模型的训练和验证过程，并通过重复多次来计算平均性能指标
    def run(self, repeats=3):
        # Loss and Optimizer
        # 交叉熵损失（CrossEntropyLoss）
        criterion = nn.CrossEntropyLoss()
        # 创建日志目录
        if not os.path.exists('log/'):
            os.mkdir('log/')
        # 根据模型和数据集名称创建日志文件
        f_out = open('log/'+self.opt.model_name+'_'+self.opt.dataset+'_val.txt', 'w', encoding='utf-8')
        # 用于存储多次重复中测试准确率和 F1 分数的最大值的平均值
        max_test_acc_avg = 0
        max_test_f1_avg = 0
        # for num in range(10):
        for i in range(repeats):
            # 打印控制台并写入日志
            print('repeat: ', (i+1))
            f_out.write('repeat: '+str(i+1)+"\n")
            # 初始化模型参数
            self._reset_params()
            # 使用 filter 函数和 lambda 表达式来筛选出模型中需要梯度的参数
            _params = filter(lambda p: p.requires_grad, self.model.parameters())
            # 定义优化器，使用重置后的参数
            # 设置了优化器的学习率，这是训练过程中用于更新模型参数的步长
            # 设置了 L2 正则化项，防止模型过拟合
            optimizer = self.opt.optimizer(_params, lr=self.opt.learning_rate, weight_decay=self.opt.l2reg)
            # 调用 _train 方法来进行训练，并获取测试准确率和 F1 分数
            max_test_acc, max_test_f1 = self._train(criterion, optimizer)
            # 打印控制台并写入日志
            print('max_test_acc: {0}     max_test_f1: {1}'.format(max_test_acc, max_test_f1))
            f_out.write('max_test_acc: {0}, max_test_f1: {1}\n'.format(max_test_acc, max_test_f1))
            # 将当前的测试准确率和 F1 分数累加到平均值变量中
            max_test_acc_avg += max_test_acc
            max_test_f1_avg += max_test_f1
            # 打印分割线
            print('#' * 100)

        # 计算并打印测试准确率和 F1 分数的平均值
        print("max_test_acc_avg:", max_test_acc_avg / repeats)
        print("max_test_f1_avg:", max_test_f1_avg / repeats)

            #
            # f_out.write('\n')

        f_out.close()


if __name__ == '__main__':
    # Hyper Parameters
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_name', default='lstm', type=str)
    parser.add_argument('--dataset', default='twitter', type=str, help='twitter, rest14, lap14, rest15, rest16')
    parser.add_argument('--optimizer', default='adam', type=str)
    parser.add_argument('--initializer', default='xavier_uniform_', type=str)
    parser.add_argument('--learning_rate', default=0.001, type=float)
    parser.add_argument('--l2reg', default=0.00001, type=float)
    # 100*?(25*5)*32
    parser.add_argument('--num_epoch', default=100, type=int)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--log_step', default=5, type=int)
    parser.add_argument('--embed_dim', default=300, type=int)
    parser.add_argument('--hidden_dim', default=300, type=int)
    parser.add_argument('--polarities_dim', default=3, type=int)
    parser.add_argument('--save', default=False, type=bool)
    parser.add_argument('--seed', default=776, type=int)
    parser.add_argument('--device', default=None, type=str)
    opt = parser.parse_args()

    model_classes = {
        'lstm': LSTM,
        'ascnn': ASCNN,
        'asgcn': ASGCN,
        'astcn': ASTCN,
    }
    input_colses = {
        'lstm': ['text_indices'],
        'ascnn': ['text_indices', 'aspect_indices', 'left_indices'],
        'asgcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_graph'],
        'astcn': ['text_indices', 'aspect_indices', 'left_indices', 'dependency_tree'],
    }
    initializers = {
        'xavier_uniform_': torch.nn.init.xavier_uniform_,
        'xavier_normal_': torch.nn.init.xavier_normal_,
        'orthogonal_': torch.nn.init.orthogonal_,
    }
    optimizers = {
        'adadelta': torch.optim.Adadelta,  # default lr=1.0
        'adagrad': torch.optim.Adagrad,  # default lr=0.01
        'adam': torch.optim.Adam,  # default lr=0.001
        'adamax': torch.optim.Adamax,  # default lr=0.002
        'asgd': torch.optim.ASGD,  # default lr=0.01
        'rmsprop': torch.optim.RMSprop,  # default lr=0.01
        'sgd': torch.optim.SGD,
    }
    opt.model_name = 'asgcn'
    opt.dataset = 'rest14'
    opt.model_class = model_classes[opt.model_name]
    opt.inputs_cols = input_colses[opt.model_name]
    opt.initializer = initializers[opt.initializer]
    opt.optimizer = optimizers[opt.optimizer]
    opt.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu') \
        if opt.device is None else torch.device(opt.device)

    if opt.seed is not None:
        random.seed(opt.seed)
        numpy.random.seed(opt.seed)
        torch.manual_seed(opt.seed)
        torch.cuda.manual_seed(opt.seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    ins = Instructor(opt)
    ins.run()
