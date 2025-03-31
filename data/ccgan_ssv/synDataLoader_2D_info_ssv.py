#synthetic heartbeat signal dataloader
#generate synthetic signal from the pre-trained generator model

import numpy as np
import torch 
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from data.ccgan_ssv.TransCGAN_model_2D_ccgan import *
from data.data_process_2D_uav import mitbih_train

# Ignore warnings
import warnings
warnings.filterwarnings("ignore")

cls_dit = {'Non-Ectopic Beats':0, 'Superventrical Ectopic':1, 'Ventricular Beats':2,
                                                'Unknown':3, 'Fusion Beats':4}

class syn_mitbih(Dataset):
    def __init__(self, model_path=r'D:\Users\Administrator\Desktop\tts-cgan-main\tts-cgan-main\logs\rfly_ccgan_2025_03_19_08_59_39\Model\checkpoint' , n_samples = 2000, seq_len = 69, reshape = False, transform=None):
        self.transform = transform
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        gen_net = Generator(seq_len=69, channels=1, num_classes=7, latent_dim=100, data_embed_dim=100,
                    label_embed_dim=10 ,depth=3, num_heads=4,
                    forward_drop_rate=0.5, attn_drop_rate=0.5).to(device)
        GAN_ckp = torch.load(model_path)

        state_dict = GAN_ckp['gen_state_dict']

        gen_net.load_state_dict(state_dict, strict=False)

        gen_net.load_state_dict(state_dict, strict=False)
        
        # shape = n_samples, 1, 1, seq_len
        self.syn_0 = self.generate_synthetic_data(gen_net, 0, n_samples)
        self.syn_1 = self.generate_synthetic_data(gen_net, 1, n_samples)
        self.syn_2 = self.generate_synthetic_data(gen_net, 2, n_samples)
        self.syn_3 = self.generate_synthetic_data(gen_net, 3, n_samples)
        self.syn_4 = self.generate_synthetic_data(gen_net, 4, n_samples)
        self.syn_5 = self.generate_synthetic_data(gen_net, 5, n_samples)
        self.syn_6 = self.generate_synthetic_data(gen_net, 6, n_samples)
        

        self.data = np.concatenate((self.syn_0, self.syn_1, self.syn_2, self.syn_3, self.syn_4, self.syn_5, self.syn_6), axis = 0)
        self.labels = np.concatenate((np.array([0]*n_samples), np.array([1]*n_samples), np.array([2]*n_samples), np.array([3]*n_samples), np.array([4]*n_samples), np.array([5]*n_samples), np.array([6]*n_samples)))

        self.data = self.data.squeeze(1)
        self.labels = self.labels.reshape(-1, 1)

        print(f'data shape is {self.data.shape}')
        print(f'labels shape is {self.labels.shape}')
        print(f'The dataset including {n_samples} class 0, {n_samples} class 1, {n_samples} class 2, {n_samples} class 3, {n_samples} class 4')

    def generate_synthetic_data(self, gen_net, classlabel, n):
        batch_size = 32
        n_batches = n // batch_size  # 生成整批数据的次数
        synthetic_list = []
        device = next(gen_net.parameters()).device  # 获取模型所在设备
        for _ in range(n_batches):
            fake_noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (batch_size, 69, 90))).to(device)
            # 生成的标签固定为当前类别（classlabel）
            fake_label = torch.full((batch_size,), classlabel, device=device, dtype=torch.long)
            fake_sigs = gen_net(fake_noise, fake_label).to('cpu').detach().numpy()
            synthetic_list.append(fake_sigs)
        # 如果 n 不是 batch_size 的整数倍，可以额外生成剩余数量的数据
        remainder = n % batch_size
        if remainder > 0:
            fake_noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (remainder, 69, 90))).to(device)
            fake_label = torch.full((remainder,), classlabel, device=device, dtype=torch.long)
            fake_sigs = gen_net(fake_noise, fake_label).to('cpu').detach().numpy()
            synthetic_list.append(fake_sigs)
        return np.concatenate(synthetic_list, axis=0)

    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[idx]



class mixed_mitbih(Dataset):
    def __init__(self, model_path,real_samples = 20, syn_samples = 30, transform=None):
        self.transform = transform
        syn_ecg = syn_mitbih(model_path,n_samples=syn_samples, reshape=True)
        real_ecg = mitbih_train(n_samples=real_samples, oneD=True)

        self.data = np.concatenate((syn_ecg.data, real_ecg.X_train), axis = 0)
        self.labels = np.concatenate((syn_ecg.labels, real_ecg.y_train), axis = 0)
        
        print(f'data shape is {self.data.shape}')
        print(f'labels shape is {self.labels.shape}')
    
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        x = self.data[idx]
        if self.transform is not None:
            x = self.transform(x)
        return x, self.labels[idx]



if __name__ == "__main__":
    mixed_ecg = mixed_mitbih(real_samples=20, syn_samples=80)
    syn_ec0g = syn_mitbih(n_samples=80, reshape=True)