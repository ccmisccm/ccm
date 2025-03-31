from torch import nn
import torch

class Marc(nn.Module):            ###决策面的调整
    def __init__(self, base_model, num_classes):
        super().__init__()
        self.omega = nn.Parameter(torch.ones(1, num_classes), requires_grad=True)
        self.beta = nn.Parameter(torch.zeros(1, num_classes), requires_grad=True)
        ### 获取全连接层最后一层权重、计算L2范数，封装为不可训练部分
        ##计算L2范数的原因： 得到每个权重的大小吗，然后用来调整权重
        self.w_norm = nn.Parameter(
            torch.norm(base_model.fc[-1].weight, dim=1), requires_grad=False
        )
        self.model = base_model
        for name, param in self.model.named_parameters():
            param.requires_grad = False

    def forward(self, x):
        """
        Forward pass of the encoder with pooling.

        :param x: Input tensor of shape (batch_size, input_channels, sequence_length)
        :return: Encoded features
        """
        logit_before = self.model(x)
        logit_after = self.omega * logit_before + self.beta * self.w_norm  # Apply margin calibration
        return logit_after