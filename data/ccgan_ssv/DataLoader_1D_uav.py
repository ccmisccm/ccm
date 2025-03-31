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

warnings.filterwarnings("ignore")

cls_dit = {'Non-Ectopic Beats': 0, 'Superventrical Ectopic': 1, 'Ventricular Beats': 2,
           'Unknown': 3, 'Fusion Beats': 4}


class mitbih_train(Dataset):
    def __init__(self, filename='./unlabeled_result.csv', n_samples=20000, oneD=False):
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

        # 检查归一化后是否存在 NaN 值
        if np.isnan(self.X_train).any():
            print("Warning: 归一化后的数据中存在 NaN 值！")
        else:
            print("归一化后的数据中没有 NaN 值。")



        if oneD:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1])
        else:
            self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, 1, self.X_train.shape[1])


        # 原标签（不连续的，如 0,5,6,7,8,9,10）
        raw_labels = train_dataset['fault_type_id'].values

        # 定义映射字典，将 [0,5,6,7,8,9,10] 转为 [0,1,2,3,4,5,6]
        label_map = {0: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6}
        # 对原标签进行映射
        mapped_labels = np.array([label_map[val] for val in raw_labels])
        self.y_train = mapped_labels




        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')
        print(
            f'The dataset including {len(data_0_resample)} class 0, {len(data_1_resample)} class 1, {len(data_2_resample)} class 2, {len(data_3_resample)} class 3, {len(data_4_resample)} class 4')

    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class mitbih_test(Dataset):
    def __init__(self, filename='./train_result.csv', n_samples=1000, oneD=False):
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
        if oneD:
            self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.X_test.shape[1])
        else:
            self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, 1, self.X_test.shape[1])

        # 检查归一化后是否存在 NaN 值
        if np.isnan(self.X_test).any():
            print("Warning: 归一化后的数据中存在 NaN 值！")
        else:
            print("归一化后的数据中没有 NaN 值。")
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.X_test.shape[1], self.X_test.shape[2])

        # 原标签（不连续的，如 0,5,6,7,8,9,10）
        raw_labels = test_dataset['fault_type_id'].values

        # 定义映射字典，将 [0,5,6,7,8,9,10] 转为 [0,1,2,3,4,5,6]
        label_map = {0: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6}
        # 对原标签进行映射
        mapped_labels = np.array([label_map[val] for val in raw_labels])
        self.y_test = mapped_labels





        print(f'X_test shape is {self.X_test.shape}')
        print(f'y_test shape is {self.y_test.shape}')
        print(
            f'The dataset including {len(data_0)} class 0, {len(data_1)} class 1, {len(data_2)} class 2, {len(data_3)} class 3, {len(data_4)} class 4')

    def __len__(self):
        return len(self.y_test)

    def __getitem__(self, idx):
        return self.X_test[idx], self.y_test[idx]




if __name__ == "__main__":
    # 示例：用 chunk_size=100, n_chunk_samples=20 测试
    dataset_test = mitbih_train()
    dataset_test1 = mitbih_test()
