# -*- coding:utf-8 -*-
"""
加utils.random_seed
"""
from utils.random_seed import *
setup_seed(seed=3407)
import os
import sys
import json
import torch
import torch.nn as nn
import torch.optim as optim
# from torchvision import transforms, datasets
from tqdm import tqdm
from torch.utils.tensorboard import SummaryWriter
from utils.model.model import My_model
from utils.model.TGCN import TGCN,TGCN_CNN
from utils.model.DMSTGCN import DMSTGCN
from utils.model.STID import *
from torch.utils.data import DataLoader
from utils.Data_Load_TGCN import *
import argparse
import datetime
import numpy as np
import warnings
warnings.filterwarnings("ignore")
from  utils.score import *
# from torchcontrib.optim import SWA

train_data = np.load('./New_dataset/train_cluster.npy')
val_data = np.load('./New_dataset/val_cluster.npy')
# adj = np.load('./Data/adj_mat_1.npy')

mean, std = train_data.mean(axis=(0,1)), train_data.std(axis=(0,1))
score_mean, score_std = torch.from_numpy(mean).cuda(), torch.from_numpy(std).cuda()
train_data[:,:,:12] = (train_data[:,:,:12] - mean[:12])/std[:12]
val_data[:,:,:12] = (val_data[:,:,:12] - mean[:12])/std[:12]

train_data[:,:,14:50] = (train_data[:,:,14:50] - mean[14:50])/std[14:50]
val_data[:,:,14:50] = (val_data[:,:,14:50] - mean[14:50])/std[14:50]

# train_loader = DataLoader(LeakDataSet(train_data), batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
# val_loader = DataLoader(LeakDataSet(val_data), batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
train_loader = DataLoader(MyDataSet(train_data, input_dim=50,shift=True), batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
val_loader = DataLoader(MyDataSet(val_data, input_dim=50,shift=True), batch_size=512, shuffle=True, num_workers=0, pin_memory=True)

"""
模型和超参数加载
"""
model_param = {
        "num_nodes": 325,
        'input_len': 36,
        'input_dim': 48,
        'embed_dim': 48,
        'output_len': 64,
        'num_layer': 4,
        "if_node": True,
        'node_dim': 32,
        "if_T_i_D": True,
        "if_D_i_W": True,
        'temp_dim_tid': 32,
        'temp_dim_diw': 32,
        'discard_week': 6,
        'temp_dim_pid': 0,
        'congest_dim': 0,
        "if_P_i_D": False,
        "if_C_T_N": False,
        'if_cluster': False,
        'cluster_dim': 0,
    }
# model_f = STID_WITH_FUSION(model_param).cuda()
model_f = STID_NODE_40(model_param).cuda()
# model_f = TGCN(adj).cuda()
# model_s = DynamicTGCN_Dual_Leak(input_dim=24, hidden_dim=128).cuda()
# a = nn.Parameter(torch.tensor(0.5))
optimizer_f = torch.optim.AdamW(model_f.parameters(),lr=1e-3, weight_decay=1e-4)
# optimizer_s = torch.optim.AdamW(model_s.parameters(), lr=1e-3, weight_decay=1e-4)
scheduler_f = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_f, T_max=100, eta_min=1e-4)
# scheduler_s = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer=optimizer_s, T_max=50, eta_min=1e-4)
loss_function = nn.MSELoss()
# loss_function = RMSE_MAE_Loss()
# opt = torchcontrib.optim.SWA(optimizer_f, swa_start=10, swa_freq=5, swa_lr=0.05)


"""
定义日志路径
"""
writer = SummaryWriter("runs/ablation" + '', flush_secs=60)
writer_name = 'My_model_without_cluster'
model_weight_path = os.path.join('model_pth',writer_name+'.pth')

"""
训练
"""
name = "STID_Node40"
for epoch in range(300):
    sum_train_loss = 0.0
    train_loss_dict = {'flow_loss': [], 'speed_loss': [], 'Score': []}
    model_f.train()
    # model_s.train()
    for step, (x_sequence, y_flow, y_speed) in enumerate(train_loader):
        optimizer_f.zero_grad()
        # opt.zero_grad()
        x_sequence, y_flow, y_speed = x_sequence.float().cuda(), y_flow.float().cuda(), y_speed.float().cuda()
        out_flow, out_speed = model_f(x_sequence)
        # out_speed = model_s(x_sequence, aux)
        flow_loss = loss_function(out_flow, y_flow)
        speed_loss = loss_function(out_speed, y_speed)
        score, flow_score, speed_score = get_score(out_flow, out_speed, y_flow, y_speed, score_mean, score_std)
        loss = flow_loss + speed_loss
        loss.backward()

        optimizer_f.step()
        # opt.step()
        scheduler_f.step()
        # optimizer_s.step()
        # scheduler_s.step()
        sum_train_loss = sum_train_loss + loss.item()
        train_loss_dict['flow_loss'].append(flow_loss.item())
        train_loss_dict['speed_loss'].append(speed_loss.item())
        train_loss_dict['Score'].append(score.item())
    epoch_times = step + 1
    # opt.swap_swa_sgd()
    # writer.add_scalars('Train_Score', {f"{name}": round(float(np.mean(train_loss_dict['score'])), 2)}, epoch + 1)
    # writer.add_scalars('Train_Flow_Score', {f"{name}": round(float(np.mean(train_loss_dict['flow_score'])), 2)}, epoch + 1)
    # writer.add_scalars('Train_Speed_Score', {f"{name}": round(float(np.mean(train_loss_dict['speed_score'])), 2)}, epoch + 1)

    writer.add_scalars('训练指标Flow_loss', {writer_name: round(float(np.mean(train_loss_dict['flow_loss'])), 3)},
                       epoch + 1)
    writer.add_scalars('训练指标Speed_loss', {writer_name: round(float(np.mean(train_loss_dict['speed_loss'])), 3)},
                       epoch + 1)
    writer.add_scalars('训练指标Score', {writer_name: round(float(np.mean(train_loss_dict['Score'])), 3)}, epoch + 1)
    """
    进入评估阶段
    """
    if epoch % 1 == 0:
        model_f.eval()
        # model_s.eval()
        sum_val_loss = 0.0
        val_loss_dict = {'flow_loss': [], 'speed_loss': [], 'Score': []}
        with torch.no_grad():
            for step, (x_sequence, y_flow, y_speed) in enumerate(val_loader):
                optimizer_f.zero_grad()
                # opt.zero_grad()

                x_sequence, y_flow, y_speed = x_sequence.float().cuda(), y_flow.float().cuda(), y_speed.float().cuda()

                out_flow, out_speed = model_f(x_sequence)
                # out_speed = model_s(x_sequence, aux)
                flow_loss = loss_function(out_flow, y_flow)
                speed_loss = loss_function(out_speed, y_speed)
                score, flow_score, speed_score = get_score(out_flow, out_speed, y_flow, y_speed, score_mean, score_std)
                loss = flow_loss + speed_loss
                sum_val_loss = sum_val_loss + loss.item()
                val_loss_dict['flow_loss'].append(flow_loss.item())
                val_loss_dict['speed_loss'].append(speed_loss.item())
                val_loss_dict['Score'].append(score.item())
            epoch_times = step + 1
            print('[Train epoch %d] Train_SUM_Loss: %.3f Val_SUM_Loss: %.3f' % (
            epoch + 1, sum_train_loss / epoch_times, sum_val_loss / epoch_times))
            writer.add_scalars('验证指标Flow_loss', {writer_name: round(float(np.mean(val_loss_dict['flow_loss'])), 3)},
                               epoch + 1)
            writer.add_scalars('验证指标Speed_loss', {writer_name: round(float(np.mean(val_loss_dict['speed_loss'])), 3)},
                               epoch + 1)
            writer.add_scalars('验证指标Score', {writer_name: round(float(np.mean(val_loss_dict['Score'])), 3)}, epoch + 1)

    if (epoch + 1) % 300 == 0:
        # continue
        torch.save(model_f.state_dict(), model_weight_path)