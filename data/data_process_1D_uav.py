# load mitbih dataset

import os
import numpy as np
import pandas as pd
import sys
from tqdm import tqdm

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from sklearn.utils import resample

# Ignore warnings
import warnings
import h5py
import pandas as pd
import numpy as np
import os
import scipy.io as scio
import torch.utils.data as data



warnings.filterwarnings("ignore")

cls_dit = {'Non-Ectopic Beats': 0, 'Superventrical Ectopic': 1, 'Ventricular Beats': 2,
           'Unknown': 3, 'Fusion Beats': 4}


class mitbih_train(Dataset):
    def __init__(self, filename=r'D:\Users\Administrator\Desktop\Semi-Marc-SimSiam-main\Semi-Marc-SimSiam-main\data\train_result.csv', n_samples=20000, oneD=False, transform=None):
        self.transform = transform
        data_train = pd.read_csv(filename)

        # making the class labels for our dataset
        data_0 = data_train[data_train['fault_type_id'] == 0]
        data_1 = data_train[data_train['fault_type_id'] == 5]
        data_2 = data_train[data_train['fault_type_id'] == 6]
        data_3 = data_train[data_train['fault_type_id'] == 7]
        data_4 = data_train[data_train['fault_type_id'] == 8]
        data_5 = data_train[data_train['fault_type_id'] == 9]
        data_6 = data_train[data_train['fault_type_id'] == 10]

        data_0_resample = resample(data_0, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_1_resample = resample(data_1, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_2_resample = resample(data_2, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_3_resample = resample(data_3, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_4_resample = resample(data_4, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_5_resample = resample(data_5, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_6_resample = resample(data_6, n_samples=n_samples,
                                   random_state=123, replace=True)

        train_dataset = pd.concat((data_0_resample, data_1_resample,
                                   data_2_resample, data_3_resample, data_4_resample, data_5_resample, data_6_resample))

        self.X_train = train_dataset.iloc[:, :-1].values

        if oneD:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
        else:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, 1, self.X_train.shape[1])
        #self.y_train = train_dataset['fault_type_id'].values


        # 原标签（不连续的，如 0,5,6,7,8,9,10）
        # 获取第 69 列的数据
        raw_labels = train_dataset.iloc[:, 69].values

        # 定义映射字典，将 [0,5,6,7,8,9,10] 转为 [0,1,2,3,4,5,6]
        label_map = {0: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6}
        # 对原标签进行映射
        mapped_labels = np.array([label_map[val] for val in raw_labels])
        y_train = mapped_labels
        self.y_train = y_train.reshape(-1, 1)



        self.X_train = self.X_train.squeeze()  # Removing two dimensions (1, 1)
        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        print(
            f'The dataset including {len(data_0_resample)} class 0, {len(data_1_resample)} class 1, {len(data_2_resample)} class 2, {len(data_3_resample)} class 3, {len(data_4_resample)} class 4')

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        x = self.X_train[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y_train[idx]


class mitbih_test(Dataset):
    def __init__(self, filename=r'D:\Users\Administrator\Desktop\Semi-Marc-SimSiam-main\Semi-Marc-SimSiam-main\data\test_result.csv', n_samples=1000, oneD=False, transform=None):
        self.transform = transform
        data_test = pd.read_csv(filename)

        # making the class labels for our dataset
        # making the class labels for our dataset
        data_0 = data_test[data_test['fault_type_id'] == 0]
        data_1 = data_test[data_test['fault_type_id'] == 5]
        data_2 = data_test[data_test['fault_type_id'] == 6]
        data_3 = data_test[data_test['fault_type_id'] == 7]
        data_4 = data_test[data_test['fault_type_id'] == 8]
        data_5 = data_test[data_test['fault_type_id'] == 9]
        data_6 = data_test[data_test['fault_type_id'] == 10]




        test_dataset = pd.concat((data_0, data_1,
                                  data_2, data_3, data_4, data_5, data_6))

        self.X_test = test_dataset.iloc[:, :-1].values
        self.X_test = self.X_test.squeeze() # Removing two dimensions (1, 1)
        if oneD:
            self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.X_test.shape[1])
        else:
            self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, 1, self.X_test.shape[1])
        #self.y_test = test_dataset['fault_type_id'].values


        # 原标签（不连续的，如 0,5,6,7,8,9,10）
        # 获取第 69 列的数据
        raw_labels = test_dataset.iloc[:, 69].values

        # 定义映射字典，将 [0,5,6,7,8,9,10] 转为 [0,1,2,3,4,5,6]
        label_map = {0: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6}
        # 对原标签进行映射
        mapped_labels = np.array([label_map[val] for val in raw_labels])
        y_test = mapped_labels

        # 将 y_test 转换为 (7000, 1) 的 ndarray，每个值被包装在一个列表中
        self.y_test = y_test.reshape(-1, 1)


        self.X_test = self.X_test.squeeze()  # Removing two dimensions (1, 1)
        print(f'X_test shape is {self.X_test.shape}')
        print(f'y_test shape is {self.y_test.shape}')
        print(
            f'The dataset including {len(data_0)} class 0, {len(data_1)} class 1, {len(data_2)} class 2, {len(data_3)} class 3, {len(data_4)} class 4')

    def __len__(self):
        return len(self.y_test)

    def __getitem__(self, idx):
        x = self.X_test[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y_test[idx]


class mitbih_ssv(Dataset):
    def __init__(self, filename=r'D:\Users\Administrator\Desktop\Semi-Marc-SimSiam-main\Semi-Marc-SimSiam-main\data\ssv_result.csv', n_samples=1000, oneD=False, transform=None):
        self.transform = transform
        data_test = pd.read_csv(filename)

        # making the class labels for our dataset
        # making the class labels for our dataset
        data_0 = data_test[data_test['fault_type_id'] == 0]
        data_1 = data_test[data_test['fault_type_id'] == 5]
        data_2 = data_test[data_test['fault_type_id'] == 6]
        data_3 = data_test[data_test['fault_type_id'] == 7]
        data_4 = data_test[data_test['fault_type_id'] == 8]
        data_5 = data_test[data_test['fault_type_id'] == 9]
        data_6 = data_test[data_test['fault_type_id'] == 10]

        data_0_resample = resample(data_0, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_1_resample = resample(data_1, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_2_resample = resample(data_2, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_3_resample = resample(data_3, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_4_resample = resample(data_4, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_5_resample = resample(data_5, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_6_resample = resample(data_6, n_samples=n_samples,
                                   random_state=123, replace=True)


        test_dataset = pd.concat((data_0_resample, data_1_resample,
                                  data_2_resample, data_3_resample, data_4_resample, data_5_resample, data_6_resample))

        self.X_test = test_dataset.iloc[:, :-1].values
        self.X_test = self.X_test.squeeze()# Removing two dimensions (1, 1)
        if oneD:
            self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.X_test.shape[1])
        else:
            self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, 1, self.X_test.shape[1])



        # 将 y_test 转换为 (7000, 1) 的 ndarray，每个值被包装在一个列表中

        # 原标签（不连续的，如 0,5,6,7,8,9,10）
        # 获取第 69 列的数据
        raw_labels = test_dataset.iloc[:, 69].values

        # 定义映射字典，将 [0,5,6,7,8,9,10] 转为 [0,1,2,3,4,5,6]
        label_map = {0: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6}
        # 对原标签进行映射
        mapped_labels = np.array([label_map[val] for val in raw_labels])
        y_test = mapped_labels

        # 将 y_test 转换为 (7000, 1) 的 ndarray，每个值被包装在一个列表中
        self.y_test = y_test.reshape(-1, 1)



        self.X_test = self.X_test.squeeze()  # Removing two dimensions (1, 1)

        print(f'X_test shape is {self.X_test.shape}')
        print(f'y_test shape is {self.y_test.shape}')
        print(
            f'The dataset including {len(data_0)} class 0, {len(data_1)} class 1, {len(data_2)} class 2, {len(data_3)} class 3, {len(data_4)} class 4')

    def __len__(self):
        return len(self.y_test)

    def __getitem__(self, idx):
        x = self.X_test[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.y_test[idx]


def get_attr(obj, *attr_names):
    """
    尝试按顺序获取 obj 中第一个存在的属性，如果都不存在，则抛出异常。
    """
    for attr in attr_names:
        if hasattr(obj, attr):
            return getattr(obj, attr)
    raise AttributeError(f"{obj} 中找不到属性：{attr_names}")




class SemiSupervisedImbalanceCWRU(data.Dataset):         ####将有标签数据 和无标签（生成伪代码） 的数据  合成 一个半监督学习可用数据集
    def __init__(self, train_set, ssv_set, omega = 0.9, transform=None):
        X_ssv = get_attr(ssv_set, 'X_test', 'data')
        y_ssv = get_attr(ssv_set, 'y_test', 'labels')



        # 之后可以进行合并
        self.X = np.vstack((train_set.X_train, X_ssv))
        self.y = np.concatenate((train_set.y_train, y_ssv))

        self.omega = np.ones(len(self.y))
        self.omega[len(train_set.y_train):] *= omega
        self.transform = transform
    def __len__(self):
        return self.X.shape[0]

    def __getitem__(self, idx):
        sample = self.X[idx]
        label = self.y[idx]
        omega = self.omega[idx]
        # Apply transformation if available
        if self.transform is not None:
            sample = self.transform(sample)

        return sample, label, omega

if __name__ == "__main__":
    # 示例：用 chunk_size=100, n_chunk_samples=20 测试
    dataset_test = mitbih_train()
    dataset_test1 = mitbih_test()
    dataset_test2 = mitbih_ssv()
