
# InfoGAN Generator
import torch
import torch.nn as nn
import torch.nn.functional as F


# InfoGAN Generator
class Generator(nn.Module):
    def __init__(self, noise_dim=100, cat_dim=9, continuous_dim=2, channels=3, feature_dim=64):
        """
        noise_dim: 随机噪声维度
        cat_dim: 离散隐变量（通常 one-hot）的维度
        continuous_dim: 连续隐变量的维度
        channels: 输出图像通道数（如 3 表示 RGB）
        feature_dim: 特征图基础通道数
        """
        super(Generator, self).__init__()
        self.noise_dim = noise_dim
        self.cat_dim = cat_dim
        self.continuous_dim = continuous_dim
        self.input_dim = noise_dim + cat_dim + continuous_dim  # 拼接后的向量长度

        # 全连接层，将输入隐向量映射到足够展开成小尺寸特征图的向量
        # 注意这里：若 noise 拼接后形状为 [B, seq_len, input_dim]，那么flatten后尺寸为 seq_len*input_dim
        self.fc = nn.Sequential(
            nn.Linear(self.input_dim * 69, feature_dim * 8 * 4 * 4),
            nn.BatchNorm1d(feature_dim * 8 * 4 * 4),
            nn.ReLU(False)
        )
        # 转置卷积还原图像（目前生成 32x32 图像）
        self.deconv = nn.Sequential(
            nn.ConvTranspose2d(feature_dim * 8, feature_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 4),
            nn.ReLU(False),
            nn.ConvTranspose2d(feature_dim * 4, feature_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.ReLU(False),
            nn.ConvTranspose2d(feature_dim * 2, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim),
            nn.ReLU(False),
            nn.ConvTranspose2d(feature_dim, channels, kernel_size=3, stride=1, padding=1),
            nn.Tanh()  # 输出范围 [-1, 1]
        )

    def forward(self, noise, cat_code, cont_code):
        """
        noise: [B, seq_len, noise_dim]  例如 [32, 69, 90]
        cat_code: [B]  (类别索引) 例如 [32]
        cont_code: [B] 或 [B, 1] 例如 [32] —— 连续隐变量
        """
        print("Generator forward:")
        print("noise shape:", noise.shape)

        # 处理离散隐变量：扩展成 [B, seq_len]，再 one-hot 成 [B, seq_len, cat_dim]
        cat_code = cat_code.unsqueeze(1).repeat(1, noise.size(1))
        print("cat_code after unsqueeze and repeat:", cat_code.shape)
        cat_code = F.one_hot(cat_code, num_classes=self.cat_dim).float()
        print("cat_code after one_hot:", cat_code.shape)

        # 处理连续隐变量：保证其形状为 [B, seq_len, continuous_dim]
        if cont_code.dim() == 1:
            cont_code = cont_code.unsqueeze(1)  # [B, 1]
        if cont_code.dim() == 2:
            cont_code = cont_code.unsqueeze(1).repeat(1, noise.size(1), 1)
        print("cont_code shape after processing:", cont_code.shape)

        # 拼接所有隐变量：得到 [B, seq_len, noise_dim+cat_dim+continuous_dim]
        x = torch.cat([noise, cat_code, cont_code], dim=2)
        print("After concatenation, x shape:", x.shape)

        # Flatten 序列部分：转换为 [B, seq_len * (noise_dim+cat_dim+continuous_dim)]
        B, seq_len, feat_dim = x.shape
        x_flat = x.reshape(B, -1).clone()
        print("x flattened shape:", x_flat.shape)

        # 全连接层映射
        x_fc = self.fc(x_flat)
        print("After fc, x_fc shape:", x_fc.shape)

        # 重塑成初始特征图 [B, feature_dim*8, 4, 4]
        x_reshaped = x_fc.reshape(B, -1, 4, 4)

        print("After reshape, x_reshaped shape:", x_reshaped.shape)

        # 反卷积生成图像（目前输出 32x32）
        x_deconv = self.deconv(x_reshaped)
        print("After deconv, x_deconv shape:", x_deconv.shape)

        # 通过插值将输出图像调整到目标尺寸 [B, channels, 100, 69]
        x_out = F.interpolate(x_deconv, size=(100, 69), mode='bilinear', align_corners=False)
        print("After interpolation, x_out shape:", x_out.shape)

        return x_out



import torch
import torch.nn as nn

class Discriminator(nn.Module):
    def __init__(self, channels=1, cat_dim=9, continuous_dim=2, feature_dim=64, img_size=(100, 69)):
        """
        channels: 输入数据的通道数，例如 1（单通道）
        cat_dim: 离散隐变量的维度
        continuous_dim: 连续隐变量的维度
        feature_dim: 卷积层基础通道数
        img_size: 输入数据的尺寸 (高度, 宽度)
        """
        super(Discriminator, self).__init__()
        self.img_size = img_size

        self.conv = nn.Sequential(
            # 第一层卷积
            nn.Conv2d(channels, feature_dim, kernel_size=4, stride=2, padding=1),
            nn.LeakyReLU(0.1, inplace=False),
            # 第二层卷积
            nn.Conv2d(feature_dim, feature_dim * 2, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 2),
            nn.LeakyReLU(0.1, inplace=False),
            # 第三层卷积
            nn.Conv2d(feature_dim * 2, feature_dim * 4, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 4),
            nn.LeakyReLU(0.1, inplace=False),
            # 第四层卷积
            nn.Conv2d(feature_dim * 4, feature_dim * 8, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(feature_dim * 8),
            nn.LeakyReLU(0.1, inplace=False)
        )

        # 动态计算全连接层输入尺寸
        with torch.no_grad():
            dummy_input = torch.randn(1, channels, img_size[0], img_size[1])
            dummy_features = self.conv(dummy_input)
            flat_dim = dummy_features.view(1, -1).shape[1]
            print("Discriminator fc input dimension (flat_dim):", flat_dim)

        self.adv_layer = nn.Sequential(
            nn.Linear(flat_dim, 1)
        )
        self.q_cat = nn.Sequential(
            nn.Linear(flat_dim, cat_dim)
        )
        self.q_cont = nn.Sequential(
            nn.Linear(flat_dim, continuous_dim)
        )

    def forward(self, x):
        print("Discriminator forward:")
        print("Input x shape:", x.shape)
        batch_size = x.size(0)
        features = self.conv(x)
        print("After conv, features shape:", features.shape)
        features_flat = features.reshape(batch_size, -1).detach().clone()
        print("After flatten, features shape:", features_flat.shape)
        validity = self.adv_layer(features_flat)
        print("Validity shape:", validity.shape)
        q_cat_logits = self.q_cat(features_flat)
        print("q_cat_logits shape:", q_cat_logits.shape)
        q_cont = self.q_cont(features_flat)
        print("q_cont shape:", q_cont.shape)
        return validity, (q_cat_logits, q_cont)


