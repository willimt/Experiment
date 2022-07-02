import numpy as np
import pandas as pd
import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import Dataset, DataLoader
from transformers import BertModel
from tqdm import tqdm
import os
import time
from transformers import BertTokenizer
from transformers import logging
import processing
from sklearn import metrics
import warnings

logging.set_verbosity_error()

os.environ["CUDA_VISIBLE_DEVICES"] = "0"


# 通过继承nn.Module类自定义符合自己需求的模型
class MyBertModel(nn.Module):

    # 初始化类
    def __init__(self, class_size, pretrained_name='bert-base-chinese'):
        """
        Args:
            class_size  :指定分类模型的最终类别数目，以确定线性分类器的映射维度
            pretrained_name :用以指定bert的预训练模型
        """
        # 类继承的初始化，固定写法
        super(MyBertModel, self).__init__()
        # 加载HuggingFace的BertModel
        # BertModel的最终输出维度默认为768
        # return_dict=True 可以使BertModel的输出具有dict属性，即以 bert_output['last_hidden_state'] 方式调用
        self.bert = BertModel.from_pretrained(pretrained_name,
                                              return_dict=True)
        # 通过一个线性层将[CLS]标签对应的维度：768->class_size
        # class_size 在SST-2情感分类任务中设置为：2
        self.classifier = nn.Linear(768, class_size)

    def forward(self, inputs):
        # 获取DataLoader中已经处理好的输入数据：
        # input_ids :tensor类型，shape=batch_size*max_len   max_len为当前batch中的最大句长
        # input_tyi :tensor类型，
        # input_attn_mask :tensor类型，因为input_ids中存在大量[Pad]填充，attention mask将pad部分值置为0，让模型只关注非pad部分
        input_ids, input_tyi, input_attn_mask = inputs['input_ids'], inputs[
            'token_type_ids'], inputs['attention_mask']
        # 将三者输入进模型，如果想知道模型内部如何运作，前面的蛆以后再来探索吧~
        output = self.bert(input_ids, input_tyi, input_attn_mask)
        # bert_output 分为两个部分：
        #   last_hidden_state:最后一个隐层的值
        #   pooler output:对应的是[CLS]的输出,用于分类任务
        # 通过线性层将维度：768->2
        # categories_numberic：tensor类型，shape=batch_size*class_size，用于后续的CrossEntropy计算
        categories_numberic = self.classifier(output.pooler_output)
        return categories_numberic


def save_pretrained(model, path):
    # 保存模型，先利用os模块创建文件夹，后利用torch.save()写入模型文件
    os.makedirs(path, exist_ok=True)
    torch.save(model, os.path.join(path, 'model.pth'))


"""
torch提供了优秀的数据加载类Dataloader，可以自动加载数据。
1. 想要使用torch的DataLoader作为训练数据的自动加载模块，就必须使用torch提供的Dataset类
2. 一定要具有__len__和__getitem__的方法，不然DataLoader不知道如何如何加载数据
这里是固定写法，是官方要求，不懂可以不做深究，一般的任务这里都通用
"""


class BertDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset
        self.data_size = len(dataset)

    def __len__(self):
        return self.data_size

    def __getitem__(self, index):
        # 这里可以自行定义，Dataloader会使用__getitem__(self, index)获取数据
        # 这里我设置 self.dataset[index] 规定了数据是按序号取得，序号是多少DataLoader自己算，用户不用操心
        return self.dataset[index]


pretrained_model_name = 'bert-base-uncased'
tokenizer = BertTokenizer.from_pretrained(pretrained_model_name)

def coffate_fn(examples):
    inputs, targets = [], []
    for sent, polar in examples:
        inputs.append(sent)
        targets.append(polar)
    inputs = tokenizer(inputs,
                       padding=True,
                       truncation=True,
                       return_tensors="pt",
                       max_length=80)
    targets = torch.tensor(targets)
    return inputs, targets


def load_bert_data(train_data, test_data, batch_size):
    train_dataset = BertDataset(train_data)
    test_dataset = BertDataset(test_data)
    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, collate_fn=coffate_fn, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=coffate_fn)

    return train_dataloader, test_dataloader


def train(model, train_dataloader, device, optimizer, loss, epoch):
    model.train()
    # 记录当前epoch的总loss
    total_loss = 0
    # tqdm用以观察训练进度，在console中会打印出进度条
    for batch in tqdm(train_dataloader, desc=f"Training Epoch {epoch}", colour='red'):
        # tqdm(train_dataloader, desc=f"Training Epoch {epoch}") 会自动执行DataLoader的工作流程，
        # 想要知道内部如何工作可以在debug时将断点打在 coffate_fn 函数内部，查看数据的处理过程
        # 对batch中的每条tensor类型数据，都执行.to(device)，
        # 因为模型和数据要在同一个设备上才能运行
        inputs, targets = [x.to(device) for x in batch]
        # 清除现有的梯度
        optimizer.zero_grad()
        # 模型前向传播，model(inputs)等同于model.forward(inputs)
        bert_output = model(inputs)
        # 计算损失，交叉熵损失计算可参考：https://zhuanlan.zhihu.com/p/159477597
        l = loss(bert_output, targets)
        # 梯度反向传播
        l.backward()
        # 根据反向传播的值更新模型的参数
        optimizer.step()
        # 统计总的损失，.item()方法用于取出tensor中的值
        total_loss += l.item()
        #测试过程
        # acc统计模型在测试数据上分类结果中的正确个数

def test(model, test_dataloader, device, epoch, file_name):
    true_y, pred_y = [], []
    for batch in tqdm(test_dataloader, desc=f"Testing", colour='green'):
        inputs, targets = [x.to(device) for x in batch]
        # with torch.no_grad(): 为固定写法，
        # 这个代码块中的全部有关tensor的操作都不产生梯度。目的是节省时间和空间，不加也没事
        with torch.no_grad():
            bert_output = model(inputs)
            pred_y.extend(bert_output.argmax(dim=1).tolist())
            true_y.extend(targets.tolist())

            # acc_score = metrics.accuracy_score(true_y, pred_y)
            # precision_score = metrics.precision_score(
            #     true_y, pred_y, average='binary')
            # recall_score = metrics.recall_score(
            #     true_y, pred_y, average='binary')
            # f1_score = metrics.f1_score(true_y, pred_y, average='binary')
            # acc += acc_score
            # recall += recall_score
            # precision += precision_score
            # f1 += f1_score

        #输出在测试集上的准确率
        # print(f"Acc: {acc / len(test_dataloader):.6f}\t", f"Precision: {precision / len(test_dataloader):.6f}\t",
        #       f"Recall: {recall / len(test_dataloader):.6f}\t", f"F1: {f1 / len(test_dataloader):.6f}\t"
        #       )
    fp = open(file_name, 'a+')
    print(metrics.confusion_matrix(true_y, pred_y))
    print(f'epochs : {epoch}', metrics.confusion_matrix(true_y, pred_y), file = fp)
    print(f'epochs : {epoch}\n', metrics.classification_report(true_y, pred_y), file=fp)
    print(metrics.classification_report(true_y, pred_y))
    print(f'Acc : {metrics.accuracy_score(true_y, pred_y)}\t F1: {metrics.f1_score(true_y, pred_y)}')
    print(f'Acc : {metrics.accuracy_score(true_y, pred_y)}\t F1: {metrics.f1_score(true_y, pred_y)}\n', file=fp)
    fp.close()