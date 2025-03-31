# -*- coding: utf-8 -*-
"""
Created on Thu Apr 19 10:31:08 2018

@author: rlk
"""

from config import Config
import torch
opt = Config()
from data import data_process_1D_uav as data_preprocess
from models import costumed_model_uav as costumed_model
import math
import random

import torch.backends.cudnn as cudnn
import os
import argparse
from sklearn.metrics import confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser(description='PyTorch ImageNet Training')
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--pretrained_model', metavar='DIR', help='path to dataset',
                    default=r"checkpoints\byol\checkpoint_byol_2D_0199_batchsize_0128.pth.tar")
parser.add_argument('--output_dir', default='./data', type=str)
parser.add_argument('--output_filename', default='pseudo_labeled_cwru.pth', type=str)
parser.add_argument('--pretrained', action='store_true', default=True)
parser.add_argument('--requires_grad', action='store_true', default=False)
parser.add_argument('--semi_requires_grad', action='store_true', default=False)
parser.add_argument('-b', '--batch-size', default=32, type=int,
                    metavar='N',
                    help='mini-batch size (default: 32), this is the total '
                         'batch size of all GPUs on the current node when '
                         'using Data Parallel or Distributed Data Parallel')
parser.add_argument('--lr', '--learning-rate', default=0.05, type=float,
                    metavar='LR', help='initial (base) learning rate', dest='lr')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum of SGD solver')
parser.add_argument('--wd', '--weight-decay', default=1e-4, type=float,
                    metavar='W', help='weight decay (default: 1e-4)',
                    dest='weight_decay')
parser.add_argument('-p', '--print-freq', default=30, type=int,
                    metavar='N', help='print frequency (default: 10)')
parser.add_argument('--seed', default=42, type=int,
                    help='seed for initializing training. ')
parser.add_argument('--gpu', default=0, type=int,
                    help='GPU id to use.')
parser.add_argument('--ssv_size', default=100, type=int,
                    help='ssv_set size (default: 200)')
parser.add_argument('--normal_size', default=100, type=int,
                    help='normal_size size (default: 200)')
parser.add_argument('--excep_size', default=10, type=int,
                    help='excep_size size (default: 200)')
parser.add_argument('--omega', default=1.0, type=float,
                    help='weight of non labeled data')
parser.add_argument('--num_classes', default=7, type=int)

import data.ssv_data_uav_syn as ssv_data
import torchvision.transforms as transforms




def main():
    args = parser.parse_args()
    ssv_method=set_ssv_method_based_on_pretrained(args)
    aug_method = ssv_data.set_aug_method()
    print('aug_method:  ',aug_method)
    print('ssv_method:  ',ssv_method)
    if args.seed is not None:
        random.seed(args.seed)
        torch.manual_seed(args.seed)
        cudnn.deterministic = True
    use_cuda = torch.cuda.is_available()
    if use_cuda:
        print('CUDA is available')
    #betas = [1, 10, 50, 100]    不平衡因子
    betas = [1]
    results = []
    class_acc_results = []
    loop = 5 # 运行loop次，取平均值
    for beta in betas:
        nonLabelCWRUData = ssv_data.NonLabelSSVData()
        fine_tune_acc = 0
        semi_acc = 0
        semi_marc_acc = 0
        class_acc_saved = []       #记录各个类别的准确率
        for i in range(loop):
            data_ssv = nonLabelCWRUData.get_ssv()
            data_train = nonLabelCWRUData.get_train()
            data_test = nonLabelCWRUData.get_test()

            loop_flag=False
            if i == loop-1:
                loop_flag=True

            model = costumed_model.StackedCNNEncoderWithPooling(num_classes=7)     ###神经网络  提取特征
            '''阶段一：微调阶段'''
            '''微调阶段 用的是训练集、测试集数据'''
            from train_uav import train as fine_tune
            fine_tune_acc_ret, class_accs = fine_tune(model, nonLabelCWRUData.get_train(), nonLabelCWRUData.get_test(), args,ssv_method, aug_method,loop_flag)
            class_acc_saved.append({"fine_tune": class_accs})

            '''阶段二：半监督学习阶段(Semi)'''
            "半监督阶段用的是有标签+ 无标签(生成为标签)  构成半监督学习数据集  "
            from utils_uav import gen_pseudo_labels

            ssv_dataset = gen_pseudo_labels(model, nonLabelCWRUData.get_ssv())      ###用训练好的模型为无标签数据生成为伪标签
            semiCWRU = data_preprocess.SemiSupervisedImbalanceCWRU(nonLabelCWRUData.get_train(), ssv_dataset,        ###有标签+ 无标签(生成为标签)  构成半监督学习数据集
                                                                   omega=args.omega)
            from train_semi_uav import train_semi
            semi_acc_ret, class_accs = train_semi(model, semiCWRU, nonLabelCWRUData.get_test(), args,ssv_method, aug_method,loop_flag)     ##半监督学习，训练部分冻结的模型
            class_acc_saved.append({"semi": class_accs})

            '''阶段三：决策面调整阶段（Marc）'''
            "半监督阶段用的是有标签+ 无标签(生成为标签)  构成半监督学习数据集 "
            from models.marc import Marc
            from train_marc_uav import marc
            model_marc = Marc(model, args.num_classes)
            acc_ret, class_accs = marc(model_marc, semiCWRU, nonLabelCWRUData.get_test(), args,ssv_method, aug_method,loop_flag)
            class_acc_saved.append({"marc": class_accs})
            fine_tune_acc += fine_tune_acc_ret
            semi_acc += semi_acc_ret
            semi_marc_acc += acc_ret
        fine_tune_acc /= loop
        semi_acc /= loop
        semi_marc_acc /= loop
        results.append({"beta":beta, "fine_tune_acc":fine_tune_acc, "semi_acc":semi_acc, "semi_marc_acc": semi_marc_acc})
        class_acc_results.append({"beta":beta, "class_acc":class_acc_saved})
        print(results)
    with open("train_results.txt", "a") as f:
        f.write(str(args.pretrained_model) + "_semi_marc:" + str(results))
        f.write("\n")
        # f.write(str(args.pretrained_model) + "_semi_marc:" + str(class_acc_results))
        # f.write("\n")


def set_ssv_method_based_on_pretrained(args):
    # 检查 args.pretrained_model 是否包含 'byol'
    if 'byol' in args.pretrained_model:
        ssv_method = 'byol'
    else:
        ssv_method = 'default'  # 如果没有 'byol'，可以设置为其他默认值

    return ssv_method


def adjust_learning_rate(optimizer, init_lr, epoch, epochs):
    """Decay the learning rate based on schedule"""
    cur_lr = init_lr * 0.5 * (1. + math.cos(math.pi * epoch / epochs))
    for param_group in optimizer.param_groups:
        if 'fix_lr' in param_group and param_group['fix_lr']:
            param_group['lr'] = init_lr
        else:
            param_group['lr'] = cur_lr

def save_checkpoint(state, is_best, filename='checkpoint.pth.tar'):
    import shutil
    torch.save(state, filename)
    if is_best:
        shutil.copyfile(filename, 'model_best.pth.tar')

if __name__ == "__main__":
    main()