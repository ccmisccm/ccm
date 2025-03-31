import torch
import torch.nn.functional as F

def supcon_fake(out1, out2, others, temperature, distributed=False):         ###  有监督对比学，创建掩码矩阵，可以选出正负样本，再分别比较相似性
    N = out1.size(0)

    _out = [out1, out2, others]
    outputs = torch.cat(_out, dim=0)
    sim_matrix = outputs @ outputs.t()             ##点积相似度，得到相似度矩阵
    sim_matrix = sim_matrix / temperature        ##用于控制相似度分布的平滑程度
    sim_matrix.fill_diagonal_(-5e4)

    mask = torch.zeros_like(sim_matrix)            ##创建掩码矩阵
    mask[2*N:,2*N:] = 1
    mask.fill_diagonal_(0)

    sim_matrix = sim_matrix[2*N:]
    mask = mask[2*N:]
    mask = mask / mask.sum(1, keepdim=True)

    lsm = F.log_softmax(sim_matrix, dim=1)
    lsm = lsm * mask
    d_loss = -lsm.sum(1).mean()
    return d_loss




def nt_xent(out1, out2, temperature=0.1, distributed=False, normalize=False):       #### 无监督学习，直接比较两个输入的相似性
    """Compute NT_xent loss"""
    assert out1.size(0) == out2.size(0)
    if normalize:
        out1 = F.normalize(out1)
        out2 = F.normalize(out2)
    N = out1.size(0)

    _out = [out1, out2]
    outputs = torch.cat(_out, dim=0)

    sim_matrix = outputs @ outputs.t()
    sim_matrix = sim_matrix / temperature

    sim_matrix.fill_diagonal_(-5e4)
    sim_matrix = F.log_softmax(sim_matrix, dim=1)
    loss = -torch.sum(sim_matrix[:N, N:].diag() + sim_matrix[N:, :N].diag()) / (2*N)

    return loss