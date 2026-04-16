from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import scipy.io as io
import os
import random
import time
import socket

from torch.optim.lr_scheduler import StepLR, MultiStepLR
from torch.utils.data import DataLoader
from dual import DCT
from data import get_patch_training_set, get_test_set
from torch.autograd import Variable
from psnr import MPSNR
from datetime import datetime
from torch.utils.tensorboard import SummaryWriter
import torch.multiprocessing as mp
import torch.distributed as dist
import torch.distributed.autograd as dist_autograd
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed.optim import DistributedOptimizer
from torch.distributed.rpc import RRef
from torch.utils.data.distributed import DistributedSampler



# Training settings
parser = argparse.ArgumentParser(description='PyTorch Super Res Example')
parser.add_argument('--upscale_factor', type=int, default=8, help="super resolution upscale factor")
parser.add_argument('--batchSize', type=int, default=8, help='training batch size')
parser.add_argument('--patch_size', type=int, default=64, help='training patch size')
parser.add_argument('--testBatchSize', type=int, default=1, help='testing batch size')
parser.add_argument('--ChDim', type=int, default=31, help='output channel number (HSI光谱波段数)')
parser.add_argument('--alpha', type=float, default=0.1, help='alpha')
parser.add_argument('--nEpochs', type=int, default=0, help='number of epochs to train for (断点续训起始epoch)')
parser.add_argument('--lr', type=float, default=0.0001, help='Learning Rate. Default=0.0001')
parser.add_argument('--threads', type=int, default=4, help='number of threads for data loader to use')
parser.add_argument('--seed', type=int, default=123, help='random seed to use. Default=123')
parser.add_argument('--save_folder', default='TrainedNet/', help='Directory to keep training outputs.')
parser.add_argument('--outputpath', type=str, default='result/', help='Path to output img')
parser.add_argument('--mode', default=1, type=int, help='Train(1) or Test(其他).')
parser.add_argument('--local_rank', default=1, type=int, help='None')
parser.add_argument('--use_distribute', type=int, default=0, help='是否使用分布式训练DDP: 1=是 0=否')
opt = parser.parse_args()

print(opt)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

set_random_seed(opt.seed)

# 分布式训练初始化
use_dist = opt.use_distribute
if use_dist:
    dist.init_process_group(backend="nccl", init_method='env://')

print('===> Loading datasets')
train_set = get_patch_training_set(opt.upscale_factor, opt.patch_size)
if use_dist:
    sampler = DistributedSampler(train_set)
test_set = get_test_set()

if use_dist:
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, sampler=sampler, pin_memory=True)
else:
    training_data_loader = DataLoader(dataset=train_set, num_workers=opt.threads, batch_size=opt.batchSize, shuffle=True, pin_memory=True)
testing_data_loader = DataLoader(dataset=test_set, num_workers=opt.threads, batch_size=opt.testBatchSize, shuffle=False, pin_memory=True)

print('===> Building model')
print("===> distribute model")


if use_dist:
    local_rank = torch.distributed.get_rank()
    torch.cuda.set_device(local_rank)
    device = torch.device("cuda", local_rank)
else:
    local_rank = 0
    device = 'cuda:0'

# [论文公式] 创建 DCTransformer 模型: DCT(n_colors=S, upscale_factor=l)
model = DCT(opt.ChDim, opt.upscale_factor).to(device)  # ChDim=31(Chikusei), upscale_factor=8

if use_dist:
    model = torch.nn.parallel.DistributedDataParallel(model, find_unused_parameters=True, device_ids=[local_rank], output_device=local_rank)
print('# network parameters: {}'.format(sum(param.numel() for param in model.parameters())))
optimizer = optim.Adam(model.parameters(), lr=opt.lr)
scheduler = MultiStepLR(optimizer, milestones=[100, 150, 175, 190, 195], gamma=0.5)  # [3.5节] 学习率衰减



if opt.nEpochs != 0:  # 断点续训
    dist.barrier()
    map_location = {'cuda:%d' % 0: 'cuda:%d' % local_rank}
    load_dict = torch.load(opt.save_folder+"_epoch_{}.pth".format(opt.nEpochs), map_location=map_location)
    opt.lr = load_dict['lr']
    epoch = load_dict['epoch']
    model.load_state_dict(load_dict['param'])
    optimizer.load_state_dict(load_dict['adam'])

criterion = nn.L1Loss(reduction='none')  # [论文公式(8)] L1 Loss


current_time = datetime.now().strftime('%b%d_%H-%M-%S')
CURRENT_DATETIME_HOSTNAME = '/' + current_time + '_' + socket.gethostname()
tb_logger = SummaryWriter(log_dir='./tb_logger/' + 'unfolding2' + CURRENT_DATETIME_HOSTNAME)
current_step = 0

def mkdir(path):
    folder = os.path.exists(path)
    if not folder:
        os.makedirs(path)
        print("---  new folder...  ---")
        print("---  " + path + "  ---")
    else:
        print("---  There exsits folder " + path + "  ---")
        
mkdir(opt.save_folder)
mkdir(opt.outputpath)

def train(epoch, optimizer, scheduler):
    """[训练函数] 单个 epoch 的训练流程

    数据流: (Y, Z, X) → model(Y,Z) → HX̂ → L1Loss(HX̂, X) → 反向传播

    Args:
        Y (Tensor): HR-MSI [B, 3, W, H] - 高空间分辨率多光谱输入
        Z (Tensor): LR-HSI [B, S, w, h] - 低空间分辨率高光谱输入 (上采样前)
        X (Tensor): GT HR-HSI [B, S, W, H] - 真值高光谱图像 (监督标签)
    Returns:
        avg_loss (float): 本 epoch 平均 loss
    """
    epoch_loss = 0
    global current_step
    model.train()
    for iteration, batch in enumerate(training_data_loader, 1):
        Y, Z, X = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
        # Y=HR-MSI[B,3,W,H], Z=LR-HSI[B,S,w,h], X=GT HR-HSI[B,S,W,H]

        optimizer.zero_grad()
        Y = Variable(Y).float()   # [B, 3, W, H]
        Z = Variable(Z).float()   # [B, S, w, h]
        X = Variable(X).float()   # [B, S, W, H]

        HX = model(Y, Z)         # 前向推理: DCTransformer → 预测HR-HSI [B, S, W, H]

        loss = criterion(HX, X).mean()   # [公式8] L1 Loss

        epoch_loss += loss.item()
        if local_rank == 0:
            tb_logger.add_scalar('total_loss', loss.item(), current_step)  # TensorBoard记录
        current_step += 1

        loss.backward()            # 反向传播
        optimizer.step()           # 参数更新 Adam

        if iteration % 100 == 0:
            print("===> Epoch[{}]({}/{}): Loss: {:.4f}".format(epoch, iteration, len(training_data_loader), loss.item()))
    print("===> Epoch {} Complete: Avg. Loss: {:.4f}".format(epoch, epoch_loss / len(training_data_loader)))
    return epoch_loss / len(training_data_loader)


def test():
    """[测试/验证函数] 在测试集上评估模型性能

    流程: 加载模型 → 逐样本推理 → 计算MPSNR → 保存结果为 .mat 文件

    Returns:
        avg_psnr (float): 测试集平均多光谱 PSNR (dB)
    """
    avg_psnr = 0
    avg_time = 0
    model.eval()
    with torch.no_grad():
        for batch in testing_data_loader:
            Y, Z, X = batch[0].cuda(), batch[1].cuda(), batch[2].cuda()
            Y = Variable(Y).float()   # [B, 3, W, H] HR-MSI
            Z = Variable(Z).float()   # [B, S, w, h] LR-HSI
            X = Variable(X).float()   # [B, S, W, H] GT HR-HSI

            start_time = time.time()
            HX = model(Y, Z)         # 推理得到预测HR-HSI
            end_time = time.time()

            # 转换为numpy计算PSNR: [B,S,W,H] → [W,H,S]
            X = torch.squeeze(X).permute(1, 2, 0).cpu().numpy()
            HX = torch.squeeze(HX).permute(1, 2, 0).cpu().numpy()
            psnr = MPSNR(HX, X)     # 多光谱PSNR(dB)，逐波段计算后取平均

            im_name = batch[3][0]   # 文件名
            print(im_name)
            print(end_time - start_time)
            avg_time += end_time - start_time
            (path, filename) = os.path.split(im_name)
            io.savemat(opt.outputpath + filename, {'HX': HX})  # 保存预测结果为mat文件
            avg_psnr += psnr
    print("===> Avg. PSNR: {:.4f} dB".format(avg_psnr / len(testing_data_loader)))
    print("===> Avg. time: {:.4f} s".format(avg_time / len(testing_data_loader)))
    return avg_psnr / len(testing_data_loader)


def checkpoint(epoch):
    """保存训练断点: 模型参数、优化器状态、学习率、当前epoch"""

    model_out_path = opt.save_folder+"_epoch_{}.pth".format(epoch)
    if epoch % 1 == 0 and local_rank == 0:
        save_dict = dict(
            lr = optimizer.state_dict()['param_groups'][0]['lr'],
            param = model.state_dict(),
            adam = optimizer.state_dict(),
            epoch = epoch
        )
        torch.save(save_dict, model_out_path)

        print("Checkpoint saved to {}".format(model_out_path))


# ===== 主训练循环 =====
if opt.mode == 1:
    for epoch in range(opt.nEpochs + 1, 201):     # 默认从第0个checkpoint继续训练到200epoch
        avg_loss = train(epoch, optimizer, scheduler)  # 训练一个epoch
        checkpoint(epoch)                              # 保存断点
        avg_psnr = test()                              # 验证集测试
        torch.cuda.empty_cache()                        # 清理显存
        if local_rank == 0:
            tb_logger.add_scalar('psnr', avg_psnr, epoch)  # TensorBoard记录PSNR
        scheduler.step()                                # 学习率衰减 (100,150,175,190,195 epoch处 ×0.5)

else:
    test()  # 仅做测试，不训练
