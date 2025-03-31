# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 17:38:22 2018

@author: lenovo
"""
import numpy as np
import torch
import pandas as pd
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt



def gen_pseudo_labels(model, ssv_dataset):
    model.eval()
    if hasattr(ssv_dataset, 'X_test'):
        X = ssv_dataset.X_test
    elif hasattr(ssv_dataset, 'data'):
        X = ssv_dataset.data
    X = torch.tensor(X).float()
    X.resize_(X.size()[0], 1, X.size()[1])
    X = X.cuda()
    y = model(X)
    y = torch.argmax(y, dim=1)
    y = y.to("cpu").detach().numpy()
    ssv_dataset.y = y
    return ssv_dataset


###=====check the acc of model on loader, if error_analysis return confuseMatrix====
def check_accuracy(model, loader, device, error_analysis=False):
    # save the errors samples predicted by model
    ys = np.array([])
    y_preds = np.array([])
    confuse_matrix = None
    # correct counts
    num_correct = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    with torch.no_grad():
        # one batch
        for x, y in loader:
            x.resize_(x.size()[0], 1, x.size()[1])
            x, y = x.float(), y.long()
            x, y = x.to(device), y.to(device)
            # predictions
            scores = model(x)
            preds = scores.max(1, keepdim=True)[1]
            # accumulate the corrects
            num_correct += preds.eq(y.view_as(preds)).sum().item()
            # confuse matrix: labels and preds
            if error_analysis:
                ys = np.append(ys, np.array(y.cpu()))
                y_preds = np.append(y_preds, np.array(preds.cpu()))
    acc = float(num_correct) / len(loader.dataset)
    # confuse matrix 
    if error_analysis:
        confuse_matrix = pd.crosstab(y_preds, ys, margins=True)
    print('Got %d / %d correct (%.2f)' % (num_correct, len(loader.dataset), 100 * acc))
    return acc, confuse_matrix


def check_accuracy_final(model, loader, device,ssv_method,aug_method, error_analysis=False, filename="test_cm"):
    # 保存模型预测错误的样本
    classes_idx = ['1', '2', '3', '4', '5', '6', '7']
    ys = []  # 初始化为列表，确保可以累积每个批次的真实标签
    y_preds = []  # 初始化为列表，确保可以累积每个批次的预测结果
    confuse_matrix = None
    # 正确预测的数量
    num_correct = 0
    model.eval()  # 将模型设置为评估模式（与模型的训练模式相反）
    with torch.no_grad():
        # 遍历每个批次
        for x, y in loader:
            x.resize_(x.size()[0], 1, x.size()[1])
            x, y = x.float(), y.long()
            x, y = x.to(device), y.to(device)
            # 进行预测
            scores = model(x)
            preds = scores.max(1, keepdim=True)[1]
            # 累积正确的预测结果
            num_correct += preds.eq(y.view_as(preds)).sum().item()
            print(f"批次大小: {y.size(0)}")  # 检查实际的批次大小
            # 混淆矩阵：真实标签和预测标签
            if error_analysis:
                y_preds.extend(preds.squeeze().cpu().numpy())  # 将预测结果添加到 y_preds
                ys.extend(y.squeeze().cpu().numpy())  # 将真实标签添加到 ys
                print(f"ys 的长度: {len(ys)}")  # 打印每个批次后 ys 的长度

    acc = float(num_correct) / len(loader.dataset)

    cm = None  # 确保 cm 总是被初始化
    # 如果启用错误分析
    if error_analysis:
        cm = confusion_matrix(ys, y_preds)  # 计算混淆矩阵
        print("Confusion Matrix:\n", cm)

        # 定义类别名称
        classes_idx = ['0', '5', '6', '7','8', '9', '10']

        cm_df = pd.DataFrame(cm,
                             index=classes_idx,  # e.g., ['1','2','3','4','5','6','7']
                             columns=classes_idx)

        # 绘制混淆矩阵热力图
        fig = plt.figure(figsize=(6.5, 5))
        sns.heatmap(cm_df, annot=True, fmt='d', cmap='cubehelix_r')
        plt.ylabel('True label')
        plt.xlabel('Predicted label')
        plt.tight_layout()  # 防止标签被截断

        # 保存热力图为 PDF
        heatmap_pdf = f'{filename}.pdf'
        fig.savefig(heatmap_pdf)
        print(f"Heatmap saved to {heatmap_pdf}")
        # plt.show()  # 可选：显示图像

        # 计算每个类别的准确率
        class_accuracy = 100 * cm.diagonal() / cm.sum(1)
        print("Per-class Accuracy (%):", class_accuracy)

    # 将预测和标签展开，用于生成分类报告
    y_preds_flatten = y_preds  # y_preds 已经是一个平坦的列表，不需要进一步展开
    y_trues_flatten = ys  # ys 也是一个平坦的列表，不需要进一步展开

    # 定义分类报告的类别名称
    target_names_7 = [
        "1",  # label 0 (原始标签 0)
        "5",  # label 1 (原始标签 5)
        "6",  # label 2 (原始标签 6)
        "7",  # label 3 (原始标签 7)
        "8",  # label 4 (原始标签 8)
        "9",  # label 5 (原始标签 9)
        "10"  # label 6 (原始标签 10)
    ]
    y_trues_flatten_tensor = torch.tensor(y_trues_flatten)
    y_preds_flatten_tensor = torch.tensor(y_preds_flatten)
    report = classification_report(
        y_trues_flatten_tensor.cpu().numpy(), y_preds_flatten_tensor.cpu().numpy(),
        labels=list(range(7)),
        target_names=target_names_7
    )

    print(report)

    # -------------------------------
    # 保存评估结果到带有时间戳的文件中
    # -------------------------------
    import datetime
    import os

    # 获取当前时间戳
    now = datetime.datetime.now()
    timestamp = now.strftime('%Y_%m_%d_%H_%M_%S')

    exp_name = 'class_result'
    # 在 exp_name 目录下创建一个名为 "class_result" 的子文件夹
    result_dir = os.path.join(exp_name, ssv_method, aug_method)
    os.makedirs(result_dir, exist_ok=True)

    # 保存结果的文件路径
    result_file = os.path.join(result_dir, f"{filename}_{timestamp}.txt")
    with open(result_file, "w") as f:
        f.write("Confusion Matrix:\n")
        if cm is not None:
            f.write(np.array2string(cm))  # 如果计算了混淆矩阵，则写入文件
        f.write("\n\n")
        f.write("Per-class Accuracy (%):\n")
        f.write(np.array2string(class_accuracy))
        f.write("\n\n")
        f.write("Classification Report:\n")
        f.write(report)
    print(f"Saved evaluation results to {result_file}")

    # 保存混淆矩阵热力图到同一目录下
    heatmap_file = os.path.join(result_dir, f"{filename}_{timestamp}_heatmap.pdf")
    fig.savefig(heatmap_file)
    print(f"Saved confusion matrix heatmap to {heatmap_file}")

    return acc, cm



def check_class_accuracy(model, loader, device, error_analysis=False):
    import numpy as np
    import pandas as pd
    import torch

    ys = np.array([])
    y_preds = np.array([])
    confuse_matrix = None

    # 保存每个类别的正确数和总数
    class_correct = {}
    class_total = {}

    # 模型预测的总正确数
    num_correct = 0

    model.eval()  # 将模型设置为评估模式
    with torch.no_grad():
        for x, y in loader:
            # 调整输入尺寸
            x.resize_(x.size()[0], 1, x.size()[1])
            x, y = x.float(), y.long()
            x, y = x.to(device), y.to(device)
            y=y.squeeze()

            # 模型预测
            scores = model(x)
            preds = scores.max(1, keepdim=False)[1]  # 获得预测的类别

            # 总正确数
            num_correct += preds.eq(y).sum().item()

            # 按类别统计正确数和总数
            for label in y.unique():
                label = label.item()
                if label not in class_correct:
                    class_correct[label] = 0
                    class_total[label] = 0
                class_correct[label] += (preds[y == label] == label).sum().item()
                class_total[label] += (y == label).sum().item()

            # 如果需要混淆矩阵，保存所有预测值和真实值
            if error_analysis:
                ys = np.append(ys, np.array(y.cpu()))
                y_preds = np.append(y_preds, np.array(preds.cpu()))

    # 总准确率
    acc = float(num_correct) / len(loader.dataset)

    # 计算每个类别的准确率
    class_accuracies = []
    for label in sorted(class_correct.keys()):
        class_acc = float(class_correct[label]) / class_total[label] if class_total[label] > 0 else 0.0
        class_accuracies.append(class_acc)

    # 混淆矩阵
    if error_analysis:
        confuse_matrix = pd.crosstab(y_preds, ys, margins=True)

    print('Got %d / %d correct (%.2f)' % (num_correct, len(loader.dataset), 100 * acc))
    return acc, confuse_matrix, class_accuracies


###=====check the acc of model on loader, if error_analysis return confuseMatrix====
def check_semi_accuracy(model, loader, device, error_analysis=False):
    # save the errors samples predicted by model
    ys = np.array([])
    y_preds = np.array([])
    confuse_matrix = None
    # correct counts
    num_correct = 0
    model.eval()  # Put the model in test mode (the opposite of model.train(), essentially)
    with torch.no_grad():
        # one batch
        for x, y, _ in loader:
            x.resize_(x.size()[0], 1, x.size()[1])
            x, y = x.float(), y.long()
            x, y = x.to(device), y.to(device)
            # predictions
            scores = model(x)
            preds = scores.max(1, keepdim=True)[1]
            # accumulate the corrects
            num_correct += preds.eq(y.view_as(preds)).sum().item()
            # confuse matrix: labels and preds
            if error_analysis:
                ys = np.append(ys, np.array(y.cpu()))
                y_preds = np.append(y_preds, np.array(preds.cpu()))
    acc = float(num_correct) / len(loader.dataset)
    # confuse matrix
    if error_analysis:
        confuse_matrix = pd.crosstab(y_preds, ys, margins=True)
    print('Got %d / %d correct (%.2f)' % (num_correct, len(loader.dataset), 100 * acc))
    return acc, confuse_matrix

# Copyright (c) 2017-present, Facebook, Inc.
# All rights reserved.
#
# This source code is licensed under the license found in the
# LICENSE file in the root directory of this source tree.
#
import os
import pickle

import numpy as np
import torch
from torch.utils.data.sampler import Sampler

import models


def load_model(path):
    """Loads model and return it without DataParallel table."""
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path)

        # size of the top layer
        N = checkpoint['state_dict']['top_layer.bias'].size()

        # build skeleton of the model
        sob = 'sobel.0.weight' in checkpoint['state_dict'].keys()
        model = models.__dict__[checkpoint['arch']](sobel=sob, out=int(N[0]))

        # deal with a dataparallel table
        def rename_key(key):
            if not 'module' in key:
                return key
            return ''.join(key.split('.module'))

        checkpoint['state_dict'] = {rename_key(key): val
                                    for key, val
                                    in checkpoint['state_dict'].items()}

        # load weights
        model.load_state_dict(checkpoint['state_dict'])
        print("Loaded")
    else:
        model = None
        print("=> no checkpoint found at '{}'".format(path))
    return model


class UnifLabelSampler(Sampler):
    """Samples elements uniformly across pseudolabels.
    Args:
        N (int): size of returned iterator.
        dataset_y (list): list of labels for the dataset.
    """

    def __init__(self, N, dataset_y):
        self.N = N
        self.dataset_y = dataset_y
        self.indexes = self.generate_indexes_epoch()

    def generate_indexes_epoch(self):
        # 获取所有唯一的标签
        unique_labels = np.unique(self.dataset_y)
        nmb_non_empty_clusters = len(unique_labels)

        # 每个标签的样本数量
        size_per_label = int(self.N / nmb_non_empty_clusters) + 1
        res = np.array([])

        for label in unique_labels:
            # 获取当前标签的所有索引
            label_indexes = np.where(self.dataset_y == label)[0]

            # 采样
            if len(label_indexes) > 0:
                sampled_indexes = np.random.choice(
                    label_indexes,
                    size_per_label,
                    replace=(len(label_indexes) <= size_per_label)
                )
                res = np.concatenate((res, sampled_indexes))

        # 随机化结果
        np.random.shuffle(res)
        res = list(res.astype('int'))

        # 保证返回的索引数量等于 N
        if len(res) >= self.N:
            return res[:self.N]
        res += res[: (self.N - len(res))]
        return res

    def __iter__(self):
        return iter(self.indexes)

    def __len__(self):
        return len(self.indexes)



class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


def learning_rate_decay(optimizer, t, lr_0):
    for param_group in optimizer.param_groups:
        lr = lr_0 / np.sqrt(1 + lr_0 * param_group['weight_decay'] * t)
        param_group['lr'] = lr


class Logger(object):
    """ Class to update every epoch to keep trace of the results
    Methods:
        - log() log and save
    """

    def __init__(self, path):
        self.path = path
        self.data = []

    def log(self, train_point):
        self.data.append(train_point)
        with open(os.path.join(self.path), 'wb') as fp:
            pickle.dump(self.data, fp, -1)


def accuracy(output, target, topk=(1,)):
    """Computes the accuracy over the k top predictions for the specified values of k"""
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)

        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.view(1, -1).expand_as(pred))

        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res

import shutil
def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')


def pareto_sample(total_sample=500, beta=100, class_num=10):
    import numpy as np

    # 帕累托分布参数
    from scipy.optimize import fsolve

    # 定义方程组，根据 R 和 n 求解 alpha
    def equation(alpha, R, n):
        p1 = 1 - 2 ** (-alpha)
        pn = n ** (-alpha) - (n + 1) ** (-alpha)
        return p1 / pn - R

    # 初始猜测 alpha 值
    alpha_guess = 1.0

    # 使用 fsolve 求解 alpha
    alpha = fsolve(equation, alpha_guess, args=(beta, class_num))[0]

    # 定义区间边界
    bins = np.arange(1, class_num + 2)  # 区间 [0, 1], [1, 2], ..., [class_num-1, class_num]

    # 定义帕累托分布的累积分布函数 F(x)
    def pareto_cdf(x, alpha):
        return 1 - x ** (-alpha)

    # 计算每个区间的积分
    cdf_values = pareto_cdf(bins, alpha)  # 从 1 开始计算 CDF 值 # 在 0 位置插入 0（F(0) = 0）
    interval_probs = np.diff(cdf_values)  # 每个区间的积分

    # 确保占比总和为 1
    interval_probs /= interval_probs.sum()

    return np.round(total_sample * interval_probs).astype(int)


import numpy as np
from scipy.fftpack import fft


def compute_time_domain_features(signal):
    N = signal.shape[1]
    mean_val = np.mean(signal, axis=1)
    std_dev = np.std(signal, axis=1, ddof=1)
    sqrt_amplitude = (np.mean(np.sqrt(np.abs(signal)), axis=1)) ** 2
    abs_mean_val = np.mean(np.abs(signal), axis=1)
    skewness = np.mean(signal ** 3, axis=1)
    kurtosis = np.mean(signal ** 4, axis=1)
    variance = np.var(signal, axis=1)
    kurtosis_index = kurtosis / (np.sqrt(variance) ** 2)
    peak_index = np.max(np.abs(signal), axis=1) / std_dev
    waveform_index = std_dev / abs_mean_val
    pulse_index = np.max(np.abs(signal), axis=1) / abs_mean_val
    skewness_index = skewness / (np.sqrt(variance) ** 3)

    return np.column_stack([
        mean_val, std_dev, sqrt_amplitude, abs_mean_val, skewness, kurtosis, variance,
        kurtosis_index, peak_index, waveform_index, pulse_index, skewness_index
    ])


def compute_frequency_domain_features(signal, fs):
    N = signal.shape[1]
    freq_spectrum = np.abs(fft(signal, axis=1))[:, :N // 2]
    freqs = np.fft.fftfreq(N, d=1 / fs)[:N // 2]

    mean_freq = np.mean(freq_spectrum, axis=1)
    var_freq = np.var(freq_spectrum, axis=1)
    skewness_freq = np.mean((freq_spectrum - mean_freq[:, None]) ** 3, axis=1) / (var_freq ** (3 / 2))
    steepness_freq = np.mean((freq_spectrum - mean_freq[:, None]) ** 4, axis=1) / (var_freq ** 2)
    gravity_freq = np.sum(freqs * freq_spectrum, axis=1) / np.sum(freq_spectrum, axis=1)
    std_freq = np.sqrt(
        np.sum((freqs - gravity_freq[:, None]) ** 2 * freq_spectrum, axis=1) / np.sum(freq_spectrum, axis=1))
    rms_freq = np.sqrt(np.sum(freqs ** 2 * freq_spectrum, axis=1) / np.sum(freq_spectrum, axis=1))
    avg_freq = np.sum(freqs ** 4 * freq_spectrum, axis=1) / np.sum(freqs ** 2 * freq_spectrum, axis=1)
    reg_degree = np.sum(freqs ** 2 * freq_spectrum, axis=1) / (
                np.sqrt(np.sum(freq_spectrum, axis=1)) * np.sqrt(np.sum(freqs ** 4 * freq_spectrum, axis=1)))
    var_param = std_freq / gravity_freq
    eighth_moment = np.sum((freqs - gravity_freq[:, None]) ** 3 * freq_spectrum, axis=1) / (N * std_freq ** 3)
    sixteenth_moment = np.sum((freqs - gravity_freq[:, None]) ** 4 * freq_spectrum, axis=1) / (N * std_freq ** 4)

    return np.column_stack([
        mean_freq, var_freq, skewness_freq, steepness_freq, gravity_freq, std_freq, rms_freq,
        avg_freq, reg_degree, var_param, eighth_moment, sixteenth_moment
    ])


if __name__ == '__main__':
    betas = range(1, 10)
    for beta in betas:
        print(pareto_sample(beta=beta))