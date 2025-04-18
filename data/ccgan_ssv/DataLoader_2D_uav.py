# load mitbih dataset

import os 
import numpy as np
import pandas
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

cls_dit = {'Non-Ectopic Beats':0, 'Superventrical Ectopic':1, 'Ventricular Beats':2,
                                                'Unknown':3, 'Fusion Beats':4}


def chunk_to_3d(df, chunk_size=100):
    """
    将 df (形状 (N, 188)) 按行分割成若干块，每块大小为 chunk_size。
    不足 chunk_size 的部分丢弃，然后 reshape 成 (n_full_chunks, chunk_size, 188)。
    返回一个 np.ndarray。
    """
    n_rows = len(df)
    n_full_chunks = n_rows // chunk_size
    cutoff = n_full_chunks * chunk_size

    # 丢弃多余行
    df_cut = df.iloc[:cutoff].reset_index(drop=True)
    # 转成 numpy
    arr = df_cut.values.astype(np.float16)  # 形状 (cutoff, 188)

    # reshape 成 (n_full_chunks, chunk_size, 188)
    arr_3d = arr.reshape(n_full_chunks, chunk_size, 70)
    return arr_3d


def normalize_features(X, skip_last_column=True):
    """
    对输入的 DataFrame X 的数值型特征进行归一化处理，使用全局均值和标准差归一化。
    如果 skip_last_column=True，则假设最后一列为标签，不参与归一化，
    最后将归一化结果重新转换为 DataFrame，并保留原始的列名。
    """
    # 如果跳过最后一列，则分离出特征部分和标签部分
    if skip_last_column:
        features = X.iloc[:, :-1].to_numpy(dtype=np.float64)
        labels = X.iloc[:, -1].to_numpy().reshape(-1, 1)
    else:
        features = X.to_numpy(dtype=np.float64)
        labels = None

    # 将无穷大替换为 NaN
    features = np.where(np.isinf(features), np.nan, features)

    # 计算每一列的均值和标准差（忽略 NaN 值）
    mu = np.nanmean(features, axis=0)
    sigma = np.nanstd(features, axis=0)

    # 定义一个非常小的阈值 epsilon，防止标准差过小
    epsilon = 1e-8
    # 对于标准差小于 epsilon 或为 NaN 的列，将 sigma 设为 1（避免除零错误）
    sigma_fixed = np.where((sigma < epsilon) | (np.isnan(sigma)), 1, sigma)

    # 归一化处理
    features_norm = (features - mu) / sigma_fixed

    # 对于常数列（sigma 很小或为 NaN），将归一化后的结果设为 0
    constant_cols = (sigma < epsilon) | (np.isnan(sigma))
    features_norm[:, constant_cols] = 0

    # 如果跳过最后一列，则将标签拼接回去
    if skip_last_column:
        X_norm = np.concatenate((features_norm, labels), axis=1)
        # 构建新的 DataFrame，保持原始列名（最后一列为标签）
        new_columns = list(X.columns[:-1]) + [X.columns[-1]]
        X_norm = pd.DataFrame(X_norm, columns=new_columns)
    else:
        X_norm = pd.DataFrame(features_norm, columns=X.columns)

    return X_norm





class mitbih_train(Dataset):
    def __init__(self, filename='./unlabeled_result.csv', n_samples=200, oneD=False):
        data_train = pd.read_csv(filename)
        data_train=normalize_features(data_train,skip_last_column=True)

        # making the class labels for our dataset
        data_0 = data_train[data_train['fault_type_id'] == 0]
        data_1 = data_train[data_train['fault_type_id'] == 5]
        data_2 = data_train[data_train['fault_type_id'] == 6]
        data_3 = data_train[data_train['fault_type_id'] == 7]
        data_4 = data_train[data_train['fault_type_id'] == 8]
        data_5 = data_train[data_train['fault_type_id'] == 9]
        data_6 = data_train[data_train['fault_type_id'] == 10]

        data_0_3d = chunk_to_3d(data_0.iloc[:, :70], chunk_size=100)
        data_1_3d = chunk_to_3d(data_1.iloc[:, :70], chunk_size=100)
        data_2_3d = chunk_to_3d(data_2.iloc[:, :70], chunk_size=100)
        data_3_3d = chunk_to_3d(data_3.iloc[:, :70], chunk_size=100)
        data_4_3d = chunk_to_3d(data_4.iloc[:, :70], chunk_size=100)
        data_5_3d = chunk_to_3d(data_5.iloc[:, :70], chunk_size=100)
        data_6_3d = chunk_to_3d(data_6.iloc[:, :70], chunk_size=100)


        data_0_resample = resample(data_0_3d, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_1_resample = resample(data_1_3d, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_2_resample = resample(data_2_3d, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_3_resample = resample(data_3_3d, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_4_resample = resample(data_4_3d, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_5_resample = resample(data_5_3d, n_samples=n_samples,
                                   random_state=123, replace=True)
        data_6_resample = resample(data_6_3d, n_samples=n_samples,
                                   random_state=123, replace=True)


        train_dataset = np.concatenate((data_0_resample, data_1_resample,
                                  data_2_resample, data_3_resample, data_4_resample, data_5_resample, data_6_resample))

        self.X_train = train_dataset[:, :,:-1]

        # 检查归一化后是否存在 NaN 值
        if np.isnan(self.X_train).any():
            print("Warning: 归一化后的数据中存在 NaN 值！")
        else:
            print("归一化后的数据中没有 NaN 值。")

        self.X_train = self.X_train.reshape(self.X_train.shape[0], 1, self.X_train.shape[1], self.X_train.shape[2])

        # 原标签（不连续的，如 0,5,6,7,8,9,10）
        raw_labels = train_dataset[:, 0, 69]

        # 定义映射字典，将 [0,5,6,7,8,9,10] 转为 [0,1,2,3,4,5,6]
        label_map = {0: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6}
        # 对原标签进行映射
        mapped_labels = np.array([label_map[val] for val in raw_labels])
        self.y_train = mapped_labels

        print(f'X_train shape is {self.X_train.shape}')
        print(f'y_train shape is {self.y_train.shape}')


        print(f'数据增强之前各类， class 0 {len(data_0_3d)}, class 5 {len(data_1_3d)} ,class 6  {len(data_2_3d)},class 7  {len(data_3_3d)},'
              f'class 8  {len(data_4_3d)},class 9  {len(data_5_3d)},class 10  {len(data_6_3d)}')

        print( f'数据增强之后各类， class 0  {len(data_0_resample)}, class 5  {len(data_1_resample)} ,class 6  {len(data_2_resample)},class 7  {len(data_3_resample)},'
            f'class 8  {len(data_4_resample)},class 9  {len(data_5_resample)},class 10  {len(data_6_resample)}')




    def __len__(self):
        return len(self.y_train)

    def __getitem__(self, idx):
        return self.X_train[idx], self.y_train[idx]


class mitbih_test(Dataset):
    def __init__(self, filename='./train_result.csv', oneD=False):
        data_test = pd.read_csv(filename)
        data_test = normalize_features(data_test, skip_last_column=True)

        # 根据 fault_type_id 制作各个类别的数据
        data_0 = data_test[data_test['fault_type_id'] == 0]
        data_1 = data_test[data_test['fault_type_id'] == 5]
        data_2 = data_test[data_test['fault_type_id'] == 6]
        data_3 = data_test[data_test['fault_type_id'] == 7]
        data_4 = data_test[data_test['fault_type_id'] == 8]
        data_5 = data_test[data_test['fault_type_id'] == 9]
        data_6 = data_test[data_test['fault_type_id'] == 10]

        # 将每个类别的数据按照 chunk_size 分块为 3D 数据（不进行重采样）
        data_0_3d = chunk_to_3d(data_0.iloc[:, :70], chunk_size=100)
        data_1_3d = chunk_to_3d(data_1.iloc[:, :70], chunk_size=100)
        data_2_3d = chunk_to_3d(data_2.iloc[:, :70], chunk_size=100)
        data_3_3d = chunk_to_3d(data_3.iloc[:, :70], chunk_size=100)
        data_4_3d = chunk_to_3d(data_4.iloc[:, :70], chunk_size=100)
        data_5_3d = chunk_to_3d(data_5.iloc[:, :70], chunk_size=100)
        data_6_3d = chunk_to_3d(data_6.iloc[:, :70], chunk_size=100)

        # 直接将所有类别的数据拼接，构建测试集
        test_dataset = np.concatenate((data_0_3d, data_1_3d,
                                       data_2_3d, data_3_3d,
                                       data_4_3d, data_5_3d,
                                       data_6_3d))

        self.X_test = test_dataset[:, :, :-1]

        # 检查归一化后是否存在 NaN 值
        if np.isnan(self.X_test).any():
            print("Warning: 归一化后的数据中存在 NaN 值！")
        else:
            print("归一化后的数据中没有 NaN 值。")
        self.X_test = self.X_test.reshape(self.X_test.shape[0], 1, self.X_test.shape[1], self.X_test.shape[2])

        # 原标签（不连续的，如 0,5,6,7,8,9,10）
        raw_labels = test_dataset[:, 0, 69]

        # 定义映射字典，将 [0,5,6,7,8,9,10] 转为 [0,1,2,3,4,5,6]
        label_map = {0: 0, 5: 1, 6: 2, 7: 3, 8: 4, 9: 5, 10: 6}
        # 对原标签进行映射
        mapped_labels = np.array([label_map[val] for val in raw_labels])
        self.y_test = mapped_labels

        print(f'X_test shape is {self.X_test.shape}')
        print(f'y_test shape is {self.y_test.shape}')
        print(f'测试集各类， class 0 {len(data_0_3d)}, class 5 {len(data_1_3d)}, class 6 {len(data_2_3d)}, '
              f'class 7 {len(data_3_3d)}, class 8 {len(data_4_3d)}, class 9 {len(data_5_3d)}, class 10 {len(data_6_3d)}')

    def __len__(self):
        return len(self.y_test)

    def __getitem__(self, idx):
        return self.X_test[idx], self.y_test[idx]




if __name__ == "__main__":
    # 示例：用 chunk_size=100, n_chunk_samples=20 测试
    dataset_test = mitbih_train()
    dataset_test1 = mitbih_test()