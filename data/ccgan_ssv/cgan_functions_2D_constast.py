# Conditional GAN training  

import logging
import operator
import os
from copy import deepcopy
import torch.nn.functional as F
import numpy as np
import torch
import torch.nn as nn
from imageio import imsave
from utils import make_grid, save_image
from tqdm import tqdm
import cv2
import torch.optim as optim
from  ContraD  import *
logger = logging.getLogger(__name__)


def cur_stages(iter, args):
        """
        Return current stage.
        :param epoch: current epoch.
        :return: current stage
        """
        # if search_iter < self.grow_step1:
        #     return 0
        # elif self.grow_step1 <= search_iter < self.grow_step2:
        #     return 1
        # else:
        #     return 2
        # for idx, grow_step in enumerate(args.grow_steps):
        #     if iter < grow_step:
        #         return idx
        # return len(args.grow_steps)
        idx = 0
        for i in range(len(args.grow_steps)):
            if iter >= args.grow_steps[i]:
                idx = i+1
        return idx


def gradient_penalty(y, x, args):
    """Compute gradient penalty: (L2_norm(dy/dx) - 1)**2."""
    weight = torch.ones(y.size()).cuda(args.gpu, non_blocking=True)
    dydx = torch.autograd.grad(outputs=y,
                               inputs=x,
                               grad_outputs=weight,
                               retain_graph=True,
                               create_graph=True,
                               only_inputs=True)[0]

    dydx = dydx.view(dydx.size(0), -1)
    dydx_l2norm = torch.sqrt(torch.sum(dydx**2, dim=1))
    return torch.mean((dydx_l2norm-1)**2)    
    
def compute_gradient_penalty(D, real_samples, fake_samples, phi):
    """Calculates the gradient penalty loss for WGAN GP"""
    # Random weight term for interpolation between real and fake samples
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1, 1, 1))).to(real_samples.get_device())
    # Get random interpolation between real and fake samples
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)
    d_interpolates = D(interpolates)
    fake = torch.ones([real_samples.shape[0], 1], requires_grad=False).to(real_samples.get_device())
    # Get gradient w.r.t. interpolates
    gradients = torch.autograd.grad(
        outputs=d_interpolates,
        inputs=interpolates,
        grad_outputs=fake,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]
    gradients = gradients.reshape(gradients.size(0), -1)
    gradient_penalty = ((gradients.norm(2, dim=1) - phi) ** 2).mean()
    return gradient_penalty



def get_is(args, gen_net: nn.Module, num_img):
    """
    Get inception score.
    :param args:
    :param gen_net:
    :param num_img:
    :return: Inception score
    """

    # eval mode
    gen_net = gen_net.eval()

    eval_iter = num_img // args.eval_batch_size
    img_list = list()
    for _ in range(eval_iter):
        z = torch.cuda.FloatTensor(np.random.normal(0, 1, (args.eval_batch_size, args.latent_dim)))

        # Generate a batch of images
        gen_imgs = gen_net(z).mul_(127.5).add_(127.5).clamp_(0.0, 255.0).permute(0, 2, 3, 1).to('cpu',
                                                                                                torch.uint8).numpy()
        img_list.extend(list(gen_imgs))

    # get inception score
    logger.info('calculate Inception score...')
    mean, std = get_inception_score(img_list)

    return mean        
        
def save_samples(args, fixed_z, fid_stat, epoch, gen_net: nn.Module, writer_dict, clean_dir=True):

    # eval mode
    gen_net.eval()
    with torch.no_grad():
        # generate images
        batch_size = fixed_z.size(0)
        sample_imgs = []
        for i in range(fixed_z.size(0)):
            sample_img = gen_net(fixed_z[i:(i+1)], epoch)
            sample_imgs.append(sample_img)
        sample_imgs = torch.cat(sample_imgs, dim=0)
        os.makedirs(f"./samples/{args.exp_name}", exist_ok=True)
        save_image(sample_imgs, f'./samples/{args.exp_name}/sampled_images_{epoch}.png', nrow=10, normalize=True, scale_each=True)
    return 0


def get_topk_arch_hidden(args, controller, gen_net, prev_archs, prev_hiddens):
    """
    ~
    :param args:
    :param controller:
    :param gen_net:
    :param prev_archs: previous architecture
    :param prev_hiddens: previous hidden vector
    :return: a list of topk archs and hiddens.
    """
    logger.info(f'=> get top{args.topk} archs out of {args.num_candidate} candidate archs...')
    assert args.num_candidate >= args.topk
    controller.eval()
    cur_stage = controller.cur_stage
    archs, _, _, hiddens = controller.sample(args.num_candidate, with_hidden=True, prev_archs=prev_archs,
                                             prev_hiddens=prev_hiddens)
    hxs, cxs = hiddens
    arch_idx_perf_table = {}
    for arch_idx in range(len(archs)):
        logger.info(f'arch: {archs[arch_idx]}')
        gen_net.set_arch(archs[arch_idx], cur_stage)
        is_score = get_is(args, gen_net, args.rl_num_eval_img)
        logger.info(f'get Inception score of {is_score}')
        arch_idx_perf_table[arch_idx] = is_score
    topk_arch_idx_perf = sorted(arch_idx_perf_table.items(), key=operator.itemgetter(1))[::-1][:args.topk]
    topk_archs = []
    topk_hxs = []
    topk_cxs = []
    logger.info(f'top{args.topk} archs:')
    for arch_idx_perf in topk_arch_idx_perf:
        logger.info(arch_idx_perf)
        arch_idx = arch_idx_perf[0]
        topk_archs.append(archs[arch_idx])
        topk_hxs.append(hxs[arch_idx].detach().requires_grad_(False))
        topk_cxs.append(cxs[arch_idx].detach().requires_grad_(False))

    return topk_archs, (topk_hxs, topk_cxs)


class LinearLrDecay(object):
    def __init__(self, optimizer, start_lr, end_lr, decay_start_step, decay_end_step):

        assert start_lr > end_lr
        self.optimizer = optimizer
        self.delta = (start_lr - end_lr) / (decay_end_step - decay_start_step)
        self.decay_start_step = decay_start_step
        self.decay_end_step = decay_end_step
        self.start_lr = start_lr
        self.end_lr = end_lr

    def step(self, current_step):
        if current_step <= self.decay_start_step:
            lr = self.start_lr
        elif current_step >= self.decay_end_step:
            lr = self.end_lr
        else:
            lr = self.start_lr - self.delta * (current_step - self.decay_start_step)
            for param_group in self.optimizer.param_groups:
                param_group['lr'] = lr
        return lr

def load_params(model, new_param, args, mode="gpu"):
    if mode == "cpu":
        for p, new_p in zip(model.parameters(), new_param):
            cpu_p = deepcopy(new_p)
#             p.data.copy_(cpu_p.cuda().to(f"cuda:{args.gpu}"))
            p.data.copy_(cpu_p.cuda().to("cpu"))
            del cpu_p
    
    else:
        for p, new_p in zip(model.parameters(), new_param):
            p.data.copy_(new_p)

def copy_params(model, mode='cpu'):
    if mode == 'gpu':
        flatten = []
        for p in model.parameters():
            cpu_p = deepcopy(p).cpu()
            flatten.append(cpu_p.data)
    else:
        flatten = deepcopy(list(p.data for p in model.parameters()))
    return flatten




class ContrastiveLoss(nn.Module):
    def __init__(self, temperature=0.1):
        super(ContrastiveLoss, self).__init__()
        self.temperature = temperature

    def forward(self, x1, x2):
        # 归一化嵌入
        x1 = F.normalize(x1, dim=-1, p=2)
        x2 = F.normalize(x2, dim=-1, p=2)

        # 计算相似度（余弦相似度）
        similarity = torch.matmul(x1, x2.T) / self.temperature

        # 使用 NT-Xent 损失
        loss = F.cross_entropy(similarity, torch.arange(x1.size(0)).to(x1.device))
        return loss


def RandomMask(sample, args_transformation):        ##数据增强1：对输入数据进行随机掩码
    dataset_for_transform = sample.clone()
    # tr = transformation(dataset_for_transform.iloc[idx, :], sample1)
    tr = transformation(dataset_for_transform, sample)  # dataset_for_transform.iloc[:, idx]
    tr.random_mask(args_transformation['mask_percentage'], args_transformation['apply_mask_prob'])
    return tr.cell_profile


def InstanceCrossover(sample, args_transformation):       ##数据增强23：对输入数据进行实例交叉
    # tr = transformation(self.dataset_for_transform.iloc[:, index], sample)
    dataset_for_transform = sample.detach().cpu().clone()
    tr = transformation(dataset_for_transform, sample)
    # # inner swap
    # tr.random_swap(self.args_transformation['swap_percentage'], self.args_transformation['apply_swap_prob'])
    tr.instance_crossover(args_transformation['cross_percentage'], args_transformation['apply_cross_prob'])
    return tr.cell_profile

class transformation():
    def __init__(self,
                 dataset,
                 cell_profile):
        self.dataset = dataset
        # self.sct_profile = sct_profile
        # self.cell_profile = torch.empty_like(cell_profile).copy_(cell_profile)
        # c=self.cell_profile
        # self.cell_profile=deepcopy(c)
        self.cell_profile = cell_profile.clone()

        # self.cell_profile = cell_profile.clone()#修改deepcopy，使用deepcopy复制，使叶子节点可以参与复制
        # self.gene_num = len(self.cell_profile)
        self.cell_num = self.cell_profile.shape[0]
        self.gene_num = self.cell_profile.shape[1]
        # self.cell_num = len(self.dataset)
        # self.cell_num = self.sct_profile.shape[1]

    def build_mask(self, masked_percentage: float):                   ##构建掩码数组
        mask = np.concatenate([np.ones(int(self.gene_num * masked_percentage), dtype=bool),
                               np.zeros(self.gene_num - int(self.gene_num * masked_percentage), dtype=bool)])
        # print(self.gene_num)
        # print(np.ones(int(self.gene_num * masked_percentage)))
        # print(np.zeros(self.gene_num - int(self.gene_num * masked_percentage)))
        np.random.shuffle(mask)
        return mask

    def random_mask(self,                                ##进行掩码操作
                    mask_percentage: float = 0.7,
                    apply_mask_prob: float = 0.5):
        # s = np.random.uniform(0, 1)
        # if s < apply_mask_prob:
        for i in range(self.cell_profile.shape[0]):
            mask = self.build_mask(mask_percentage)
            self.cell_profile[i][mask] = 0

                # o=self.cell_profile
                # mask = self.build_mask(mask_percentage)
                # b = self.cell_profile.shape[0]  # 得到batch size大小
                # mask = np.tile(mask, (b, 1))  # 复制mask矩阵，得到batch size*gene number的矩阵，对第一维复制b次，对第二维复制1次
                # self.cell_profile[mask] = 0

    def random_swap(self,                                ##进行交换操作
                    swap_percentage: float = 0.1,
                    apply_swap_prob: float = 0.5):

        ##### for debug
        #     from copy import deepcopy
        #     before_swap = deepcopy(cell_profile)
        # s = np.random.uniform(0, 1)
        # if s < apply_swap_prob:
            # create the number of pairs for swapping
        swap_instances = int(self.gene_num * swap_percentage / 2)  # 一个细胞中需要交换基因表达值的次数
        swap_pair = np.random.randint(self.gene_num,
                                      size=(
                                          swap_instances, 2))  # 生成随机值从0到gene_num，大小为(swap_instances, 2),生成需要交换基因表达值的对

        # do the inner crossover with p
        # s=self.cell_profile[:,swap_pair[:, 0]]
        # q=self.cell_profile[:,swap_pair[:, 1]]
        self.cell_profile[:, swap_pair[:, 0]], self.cell_profile[:, swap_pair[:, 1]] = \
            self.cell_profile[:, swap_pair[:, 1]], self.cell_profile[:, swap_pair[:, 0]]  # 将一个细胞中需要交换基因表达值的基因进行交换
        # d = self.cell_profile


            # return d

    def instance_crossover(self, cross_percentage: float = 0.25, apply_cross_prob: float = 0.4):
        # 获取当前设备
        device = self.cell_profile.device  # 获取 self.cell_profile 的设备

        for i in range(self.cell_num):
            # 随机选择一个细胞
            cross_idx = np.random.randint(self.cell_num)
            cross_instance = self.dataset[cross_idx]  # 得到随机选择细胞的基因表达值

            # 确保 cross_instance 在正确的设备上
            cross_instance = cross_instance.to(device)

            # 确定需要交换的基因位置True
            mask = self.build_mask(cross_percentage)

            # 进行实例交叉时，确保 tmp 和 cross_instance 都在相同的设备上
            tmp = cross_instance[mask].clone().to(device)
            # 交换
            cross_instance[mask], self.cell_profile[i][mask] = self.cell_profile[i][mask], tmp


args_transformation = {
    # mask
    'mask_percentage': 0.7,
    'apply_mask_prob': 0.5,
    'noise_percentage': 0.8,
    'sigma': 0.5,
    'apply_noise_prob': 0.5,
    'swap_percentage': 0.1,
    'apply_swap_prob': 0.5,
    'cross_percentage': 0.25,
    'apply_cross_prob': 0.5,
}


# 对比学习中的损失
def train(args, gen_net: nn.Module, dis_net: nn.Module, gen_optimizer, dis_optimizer, gen_avg_param, train_loader,
          epoch, writer_dict, fixed_z, schedulers=None):
    writer = writer_dict['writer']
    gen_step = 0
    cls_criterion = nn.CrossEntropyLoss()
    lambda_cls = 1
    lambda_gp = 10

    # train mode
    gen_net.train()
    dis_net.train()

    for iter_idx, (real_imgs, real_img_labels) in enumerate(tqdm(train_loader)):
        global_steps = writer_dict['train_global_steps']

        # Adversarial ground truths
        real_imgs = real_imgs.type(torch.cuda.FloatTensor).cuda(args.gpu, non_blocking=True)
        real_img_labels = real_img_labels.type(torch.LongTensor)
        real_img_labels = real_img_labels.cuda(args.gpu, non_blocking=True)

        # Sample noise as generator input
        # noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (real_imgs.shape[0], args.latent_dim))).cuda(args.gpu, non_blocking=True)
        # fake_img_labels = torch.randint(0, 5, (real_imgs.shape[0],)).cuda(args.gpu, non_blocking=True)

        noise = torch.cuda.FloatTensor(np.random.normal(0, 1, (32, 69, 90))).cuda(args.gpu, non_blocking=True)
        fake_img_labels = torch.randint(0, 5, (32,)).cuda(args.gpu, non_blocking=True)

        # ---------------------
        #  Train Discriminator
        # ---------------------
        print('noise.shape', noise.shape)
        print('fake_img_labels.shape', fake_img_labels.shape)

        dis_net.zero_grad()
        r_out_adv = dis_net(real_imgs)
        fake_imgs = gen_net(noise)
        fake_imgs1 = gen_net(noise)

        assert fake_imgs.size() == real_imgs.size(), f"fake_imgs.size(): {fake_imgs.size()} real_imgs.size(): {real_imgs.size()}"

        f_out_adv = dis_net(fake_imgs)
        f_out_adv1 = dis_net(fake_imgs1)
        f_out_adv1=RandomMask(f_out_adv1,args_transformation)
        f_out_adv_compare=InstanceCrossover(f_out_adv1,args_transformation)
        #f_out_adv_compare = dis_net(fake_imgs1)
        d_loss_simclr= nt_xent(f_out_adv,f_out_adv1)
        d_loss_sup=supcon_fake(f_out_adv,f_out_adv1,f_out_adv_compare,0.1)



        # Compute loss for gradient penalty.
        alpha = torch.rand(real_imgs.size(0), 1, 1, 1).cuda(args.gpu, non_blocking=True)  # bh, C, H, W
        x_hat = (alpha * real_imgs.data + (1 - alpha) * fake_imgs.data).requires_grad_(True)
        out_src = dis_net(x_hat)
        d_loss_gp = gradient_penalty(out_src, x_hat, args)


        d_real_loss = -torch.mean(r_out_adv)
        d_fake_loss = torch.mean(f_out_adv)
        d_adv_loss = d_real_loss + d_fake_loss

        d_loss = d_adv_loss + lambda_gp * d_loss_gp+d_loss_simclr+d_loss_sup
        d_loss.backward()

        torch.nn.utils.clip_grad_norm_(dis_net.parameters(), 5.)
        dis_optimizer.step()

        writer.add_scalar('d_loss', d_loss.item(), global_steps) if args.rank == 0 else 0

        # -----------------
        #  Train Generator
        # -----------------

        gen_net.zero_grad()

        gen_imgs = gen_net(noise)
        g_out_adv = dis_net(gen_imgs)

        g_adv_loss = -torch.mean(g_out_adv)
        g_loss = g_adv_loss
        g_loss.backward()

        torch.nn.utils.clip_grad_norm_(gen_net.parameters(), 5.)
        gen_optimizer.step()

        # adjust learning rate
        if schedulers:
            gen_scheduler, dis_scheduler = schedulers
            g_lr = gen_scheduler.step(global_steps)
            d_lr = dis_scheduler.step(global_steps)
            writer.add_scalar('LR/g_lr', g_lr, global_steps)
            writer.add_scalar('LR/d_lr', d_lr, global_steps)

        # moving average weight
        ema_nimg = args.ema_kimg * 1000
        cur_nimg = args.dis_batch_size * args.world_size * global_steps
        if args.ema_warmup != 0:
            ema_nimg = min(ema_nimg, cur_nimg * args.ema_warmup)
            ema_beta = 0.5 ** (float(args.dis_batch_size * args.world_size) / max(ema_nimg, 1e-8))
        else:
            ema_beta = args.ema

        # moving average weight
        for p, avg_p in zip(gen_net.parameters(), gen_avg_param):
            cpu_p = deepcopy(p)
            avg_p.mul_(ema_beta).add_(1. - ema_beta, cpu_p.cpu().data)
            del cpu_p

        writer.add_scalar('g_loss', g_loss.item(), global_steps) if args.rank == 0 else 0
        gen_step += 1

        # verbose
        if gen_step and iter_idx % args.print_freq == 0 and args.rank == 0:
            tqdm.write(
                "[Epoch %d/%d] [Batch %d/%d] [D loss: %f] [G loss: %f] [ema: %f] " %
                (epoch, args.max_epoch, iter_idx % len(train_loader), len(train_loader), d_loss.item(), g_loss.item(),
                 ema_beta))

        del gen_imgs
        del real_imgs
        del fake_imgs
        del f_out_adv
        del r_out_adv

        del d_loss_sup
        del d_loss_simclr
        del g_adv_loss
        del g_loss
        del d_adv_loss
        del d_loss

        writer_dict['train_global_steps'] = global_steps + 1

