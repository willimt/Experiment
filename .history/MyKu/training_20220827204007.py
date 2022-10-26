from numpy import record
import torch
from torch import nn
from d2l import torch as d2l
from tqdm import tqdm
from sklearn import metrics
import numpy as np

class Accumulator:
    """For accumulating sums over `n` variables."""
    def __init__(self, n):
        """Defined in :numref:`sec_softmax_scratch`"""
        self.data = [0.0] * n

    def add(self, *args):
        self.data = [a + float(b) for a, b in zip(self.data, args)]

    def reset(self):
        self.data = [0.0] * len(self.data)

    def __getitem__(self, idx):
        return self.data[idx]


def accuracy(y_hat, y):
    """Compute the number of correct predictions.

    Defined in :numref:`sec_softmax_scratch`"""
    if len(y_hat.shape) > 1 and y_hat.shape[1] > 1:
        y_hat = d2l.argmax(y_hat, axis=1)
    cmp = d2l.astype(y_hat, y.dtype) == y
    return float(d2l.reduce_sum(d2l.astype(cmp, y.dtype)))

def train_batch(net, X, y, loss, trainer, devices):
    if isinstance(X, list):
        X = [x.to(devices[0]) for x in X]
    else:
        X = X.to(devices[0])
    y = y.to(devices[0])
    net.train()
    trainer.zero_grad()
    pred = net(X)
    l = loss(pred, y)
    l.sum().backward()
    trainer.step()
    train_loss_sum = l.sum()
    train_acc_sum = accuracy(pred, y)
    return train_loss_sum, train_acc_sum

def evaluate_acciracy_test(net, test_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    records = Accumulator(2)
    with torch.no_grad():
        for X, y in test_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            records.add(accuracy(net(X), y), len(y))
    return records[0] / records[1]

def evaluate_f1_test(net, test_iter, device=None):
    if isinstance(net, nn.Module):
        net.eval()
        if not device:
            device = next(iter(net.parameters())).device
    f1_score = 0.0
    with torch.no_grad():
        for X, y in test_iter:
            if isinstance(X, list):
                X = [x.to(device) for x in X]
            else:
                X = X.to(device)
            y = y.to(device)
            temp = metrics.f1_score(np.argmax(net(X).cpu(), axis=1).numpy(), y.cpu().numpy(), average='macro')
            if temp > f1_score:
                f1_score = temp
    return f1_score

def train(net, train_iter, test_iter, loss, trainer, epoch, devices=d2l.try_all_gpus()):
    net = nn.DataParallel(net, device_ids=devices).to(devices[0])
    # for epoch in tqdm(range(num_epochs)):
    #     records = Accumulator(4)
    #     for features, labels in train_iter:
    #         l, acc = train_batch(net, features, labels, loss, trainer, devices)
    #         records.add(l, acc, labels.shape[0], labels.numel())
    #     test_acc = evaluate_acciracy_test(net, test_iter)
    #     test_f1 = evaluate_f1_test(net, test_iter)
    #     print(f'epoch number:{epoch + 1}', f'  loss {records[0] / records[2]:.3f}, train acc '
    #       f'{records[1] / records[3]:.3f}, test acc {test_acc:.3f} , test macro F1 {test_f1:.3f}')
    for batch in tqdm(train_iter, desc=f"Training Epoch {epoch}", colour='red'):
        # tqdm(train_dataloader, desc=f"Training Epoch {epoch}") 会自动执行DataLoader的工作流程，
        # 想要知道内部如何工作可以在debug时将断点打在 coffate_fn 函数内部，查看数据的处理过程
        # 对batch中的每条tensor类型数据，都执行.to(device)，
        # 因为模型和数据要在同一个设备上才能运行
        inputs, targets = [x.to(devices[]) for x in batch]
        # 清除现有的梯度
        trainer.zero_grad()
        # 模型前向传播，model(inputs)等同于model.forward(inputs)
        output = net(inputs)
        # 计算损失，交叉熵损失计算可参考：https://zhuanlan.zhihu.com/p/159477597
        l = loss(output, targets)
        # 梯度反向传播
        l.backward()
        # 根据反向传播的值更新模型的参数
        trainer.step()
        # 统计总的损失，.item()方法用于取出tensor中的值
        total_loss += l.item()
    true_y, pred_y = [], []
    for batch in tqdm(test_iter, desc=f"Testing", colour='green'):
        inputs, targets = [x.to(devices) for x in batch]
        # with torch.no_grad(): 为固定写法，
        # 这个代码块中的全部有关tensor的操作都不产生梯度。目的是节省时间和空间，不加也没事
        with torch.no_grad():
            output = net(inputs)
            pred_y.extend(output.argmax(dim=1).tolist())
            true_y.extend(targets.tolist())
    print(metrics.confusion_matrix(true_y, pred_y))
    print(f'epochs : {epoch}', metrics.confusion_matrix(
        true_y, pred_y))
    # print(f'epochs : {epoch}\n', metrics.classification_report(
    #     true_y, pred_y))
    print(metrics.classification_report(true_y, pred_y))
    print(
        f'Acc : {metrics.accuracy_score(true_y, pred_y)}\t F1: {metrics.f1_score(true_y, pred_y)}')
    # print(f'Acc : {metrics.accuracy_score(true_y, pred_y)}\t F1: {metrics.f1_score(true_y, pred_y)}\n',)



def calculate_measure(tp, fn, fp):
    # avoid nan
    if tp == 0:
        return 0, 0, 0

    p = tp * 1.0 / (tp + fp)
    r = tp * 1.0 / (tp + fn)
    if (p + r) > 0:
        f1 = 2.0 * (p * r) / (p + r)
    else:
        f1 = 0
    return p, r, f1


class Measure(object):
    def __init__(self, num_classes, target_class):
        """

        Args:
            num_classes: number of classes.
            target_class: target class we focus on, used to print info and do early stopping.
        """
        self.num_classes = num_classes
        self.target_class = target_class
        self.true_positives = {}
        self.false_positives = {}
        self.false_negatives = {}
        self.target_best_f1 = 0.0
        self.target_best_f1_epoch = 0
        self.reset_info()

    def reset_info(self):
        """
            reset info after each epoch.
        """
        self.true_positives = {cur_class: []
                               for cur_class in range(self.num_classes)}
        self.false_positives = {cur_class: []
                                for cur_class in range(self.num_classes)}
        self.false_negatives = {cur_class: []
                                for cur_class in range(self.num_classes)}

    def append_measures(self, predictions, labels):
        predicted_classes = predictions.argmax(
            dim=1)  # 返回每行最大值的索引prediction是一个向量结果
        # predicted_classes = predictions#返回每行最大值的索引prediction是一个向量结果
        for cl in range(self.num_classes):
            cl_indices = (labels == cl)
            pos = (predicted_classes == cl)
            hits = (predicted_classes[cl_indices] == labels[cl_indices])

            tp = hits.sum()  # 对应起来相等的结果数量
            fn = hits.size(0) - tp  # 没预测到
            fp = pos.sum() - tp  # 预测错误

            self.true_positives[cl].append(tp.cpu())
            self.false_negatives[cl].append(fn.cpu())
            self.false_positives[cl].append(fp.cpu())

    def get_total_measure(self):
        tp = sum(self.true_positives[self.target_class])
        fn = sum(self.false_negatives[self.target_class])
        fp = sum(self.false_positives[self.target_class])

        p, r, f1 = calculate_measure(tp, fn, fp)
        #对整体做计算
        return p, r, f1

    def get_avg_measure(self):
        tp = sum(self.true_positives[self.target_class])
        fn = sum(self.false_negatives[self.target_class])
        fp = sum(self.false_positives[self.target_class])

        p, r, f1 = calculate_measure(tp, fn, fp)
        #对整体做计算
        return p, r, f1

    def update_best_f1(self, cur_f1, cur_epoch):
        if cur_f1 > self.target_best_f1 and cur_f1 != 1:  # 重点改动
            self.target_best_f1 = cur_f1
            self.target_best_f1_epoch = cur_epoch
