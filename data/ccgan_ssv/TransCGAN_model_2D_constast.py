import torch
import torch.nn as nn
import torch.nn.functional as F
from torch import Tensor
import math
import numpy as np


# 生成器类
import torch
import torch.nn as nn
import torch.nn.functional as F


# 生成器类
class Generator(nn.Module):
    def __init__(self, latent_dim=90, seq_len=100, data_embed_dim=69, channels=1):
        super(Generator, self).__init__()
        self.latent_dim = latent_dim
        self.seq_len = seq_len
        self.data_embed_dim = data_embed_dim
        self.channels = channels

        # 全连接层
        self.fc1 = nn.Linear(latent_dim, 256)  # 更大尺寸的隐藏层
        self.fc2 = nn.Linear(256, 512)  # 更大尺寸的隐藏层
        self.fc3 = nn.Linear(512,  100 )  # 最后一个较大的全连接层

    def forward(self, z):
        print(f"Generator Input z: {z.shape}")
        x = F.relu(self.fc1(z))
        print(f" Generator After fc1: {x.shape}")
        x = F.relu(self.fc2(x))
        print(f" Generator After fc2: {x.shape}")
        x = F.relu(self.fc3(x))
        print(f" Generator After fc3: {x.shape}")

        # reshape 为最终的目标形状
        x = x.view(x.size(0), 1, 100, 69)
        print(f" After reshaping to [32, 1, 100, 69]: {x.shape}")

        return x



# 判别器类
class Discriminator(nn.Module):
    def __init__(self, channels=1, data_embed_dim=69, seq_len=100):
        super(Discriminator, self).__init__()

        # 卷积层实现
        self.conv1 = nn.Conv2d(channels, 32, kernel_size=4, stride=2, padding=1)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=4, stride=2, padding=1)
        self.conv3 = nn.Conv2d(64, 128, kernel_size=4, stride=2, padding=1)

        # 添加全连接层，用于满足卷积层输入要求
        self.fc1 = nn.Linear(128 * 12 * 8, 2048)  # 添加更多神经元
        self.fc2 = nn.Linear(2048, 1024)  # 添加第二个全连接层
        self.fc3 = nn.Linear(1024, 1)  # 输出一个标量

    def forward(self, x):
        print(f"Discriminator Input x: {x.shape}")

        # 卷积层
        x = F.leaky_relu(self.conv1(x), 0.2)
        print(f" Discriminator After conv1: {x.shape}")

        x = F.leaky_relu(self.conv2(x), 0.2)
        print(f"Discriminator After conv2: {x.shape}")

        x = F.leaky_relu(self.conv3(x), 0.2)
        print(f"Discriminator After conv3: {x.shape}")

        # 展平
        x = x.view(x.size(0), -1)
        print(f"Discriminator After flattening: {x.shape}")

        # 添加两个全连接层
        x = F.leaky_relu(self.fc1(x), 0.2)
        print(f"Discriminator After fc1: {x.shape}")

        x = F.leaky_relu(self.fc2(x), 0.2)
        print(f"Discriminator After fc2: {x.shape}")

        x = self.fc3(x)
        print(f"Discriminator After fc3 (output): {x.shape}")

        return x



