# -*- coding = utf-8 -*-
# @Time : 2024/10/4 16:19
# @Author : bobobobn          定义了多个继承自 DataBase 类的数据集类，用于生成不同类型的自监督标签
# @File : ssv_data.py
# @Software: PyCharm





import torch
import torchvision.transforms as transforms
#from data.data_process_1D_uav import *
from data.data_process_2D_uav  import  *
from data.ccgan_ssv.synDataLoader_2D_ssv  import *


aug_method ='2D'
aug_model_path=r'D:\Users\Administrator\Desktop\Semi-Marc-SimSiam-main\Semi-Marc-SimSiam-main\data\ccgan_ssv\logs\2D_info_2025_03_31_09_14_29\Model\checkpoint'


class NonLabelSSVData(Dataset):
    def get_train(self, transforms=None):
        # Provide a default transform if none is passed

        # return mitbih_test(filename='./data/test_result.csv', n_samples=1000, oneD=False, transform=transforms)
        if aug_method == '2D':
            syn_ecg = mitbih_train(filename=r'D:\Users\Administrator\Desktop\Semi-Marc-SimSiam-main\Semi-Marc-SimSiam-main\data\train_result.csv', n_samples=20, oneD=True, transform=transforms)
            syn_ecg.y_train = np.repeat(syn_ecg.y_train, syn_ecg.X_train.shape[1])
            syn_ecg.X_train = syn_ecg.X_train.reshape(syn_ecg.X_train.shape[0] * syn_ecg.X_train.shape[1], syn_ecg.X_train.shape[2])

            return syn_ecg
        else:
            # return mixed_mitbih(real_samples=200, syn_samples=800, transform=transforms)
            return  mitbih_train(filename=r'D:\Users\Administrator\Desktop\Semi-Marc-SimSiam-main\Semi-Marc-SimSiam-main\data\train_result.csv', n_samples=20, oneD=True, transform=transforms)


    def get_ssv(self, transforms=None):

        #return mitbih_test(filename='./data/test_result.csv', n_samples=1000, oneD=False, transform=transforms)
        if aug_method == '2D':
            syn_ecg = mixed_mitbih(aug_model_path,real_samples=20, syn_samples=80, transform=transforms)
            #syn_ecg = syn_mitbih(n_samples=80, reshape=True, transform=transforms)
            syn_ecg.labels = np.repeat(syn_ecg.labels, syn_ecg.data.shape[1])
            syn_ecg.data = syn_ecg.data.reshape(syn_ecg.data.shape[0] * syn_ecg.data.shape[1], syn_ecg.data.shape[2])

            return syn_ecg
        else:
            return mixed_mitbih(aug_model_path,real_samples=200, syn_samples=800, transform=transforms)
            #return syn_mitbih(n_samples=80, reshape=True, transform=transforms)

    def get_test(self, transforms=None):

        if aug_method == '2D':
            syn_ecg = mitbih_test(filename=r'D:\Users\Administrator\Desktop\Semi-Marc-SimSiam-main\Semi-Marc-SimSiam-main\data\test_result.csv', n_samples=1, oneD=False, transform=transforms)
            syn_ecg.y_test = np.repeat(syn_ecg.y_test, syn_ecg.X_test.shape[1])
            syn_ecg.X_test= syn_ecg.X_test.reshape(syn_ecg.X_test.shape[0] * syn_ecg.X_test.shape[1], syn_ecg.X_test.shape[2])
            return syn_ecg
        else:
            # return mixed_mitbih(real_samples=200, syn_samples=800, transform=transforms)
            return  mitbih_test(filename=r'D:\Users\Administrator\Desktop\Semi-Marc-SimSiam-main\Semi-Marc-SimSiam-main\data\test_result.csv', n_samples=10, oneD=False, transform=transforms)



def return_file_name():
     return aug_method



def set_aug_method():
    if '1D_ccgan' in aug_model_path:
        aug_metho_ssv = '1D_ccgan'
    elif '2D_ccgan' in aug_model_path:
        aug_metho_ssv = '2D_ccgan'
    elif '1D_wgan' in aug_model_path:
        aug_metho_ssv = '1D_wgan'
    elif '2D_wgan' in aug_model_path:
        aug_metho_ssv = '2D_wgan'
    elif '1D_constast' in aug_model_path:
        aug_metho_ssv = '1D_constast'
    elif '2D_constast' in aug_model_path:
        aug_metho_ssv = '2D_constast'
    elif '1D_info' in aug_model_path:
        aug_metho_ssv = '1D_info'
    elif '2D_info' in aug_model_path:
        aug_metho_ssv = '2D_info'
    else:
        aug_metho_ssv = 'default'
    return aug_metho_ssv



def check_nan_inf(array, name="array"):
    """检查一个 NumPy 数组中是否存在 NaN 或 Inf，并打印结果。"""
    if np.isnan(array).any():
        print(f"{name} contains NaN values.")
    else:
        print(f"{name} does not contain NaN values.")

    if np.isinf(array).any():
        print(f"{name} contains Inf values.")
    else:
        print(f"{name} does not contain Inf values.")


def check_data_object(data_obj, name="data_obj"):
    """
    先获取对象有哪些属性，再对其中是 NumPy 数组的属性进行 NaN/Inf 检查。
    """
    # 过滤掉内置方法或属性（如 __dict__、__init__ 等）以及可调用的方法
    attributes = [
        attr for attr in dir(data_obj)
        if not attr.startswith("__")
           and not callable(getattr(data_obj, attr))
    ]

    print(f"{name} has attributes: {attributes}")

    # 依次获取每个属性的值
    for attr in attributes:
        val = getattr(data_obj, attr)

        # 如果该属性是一个 NumPy 数组，则检查是否有 NaN/Inf
        if isinstance(val, np.ndarray):
            check_nan_inf(val, f"{name}.{attr}")
        else:
            # 如果不是 NumPy 数组，可以视需要继续处理：
            # 例如，如果是 list/tuple，可能还要进一步遍历；此处先简单跳过
            pass


if __name__ == "__main__":
    nonLabelCWRUData = NonLabelSSVData()
    # 使用实例调用 get_train() 方法
    train_data = nonLabelCWRUData.get_train()

    # 打印返回的数据形状
    print("Training data shape:", train_data.X_train.shape)
    print("Training labels shape:", train_data.y_train.shape)

    test_data = nonLabelCWRUData.get_test()

    # 使用实例调用 get_ssv() 方法
    ssv_data = nonLabelCWRUData.get_ssv()

    # 打印返回的 SSV 数据形状
    print("SSV data shape:", ssv_data.data.shape)
    print("SSV labels shape:", ssv_data.labels.shape)



    data_ssv = nonLabelCWRUData.get_ssv()
    data_train = nonLabelCWRUData.get_train()
    data_test = nonLabelCWRUData.get_test()

    check_data_object(data_ssv, "data_ssv")
    check_data_object(data_train, "data_train")
    check_data_object(data_test, "data_test")

