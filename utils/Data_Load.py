# -*- coding:utf-8 -*-
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
import imageio as io
import cv2
import torch
from scipy.ndimage import filters
import numpy
import scipy.io as sio
import matplotlib.pyplot as plt
import json
import numpy as np
from tqdm import tqdm
from PIL import Image
import datetime
import multiprocessing
import warnings
warnings.filterwarnings("ignore")

class MyDataSet(Dataset):
    def __init__(self,Data_file_dict:dict,Data_inf,train_flag=True):
        self.Data_file_dict = Data_file_dict
        self.Data_inf = Data_inf
        self.train_flag = train_flag

    def __getitem__(self, index):
        data_inf = self.Data_inf[index]
        if self.train_flag:
            x_list = [os.path.join(self.Data_file_dict['train_x'],x_name) for x_name in data_inf['x']]
            y_list = [os.path.join(self.Data_file_dict['train_y'], x_name) for x_name in data_inf['y']]
        else:
            x_list = [os.path.join(self.Data_file_dict['val_x'], x_name) for x_name in data_inf['x']]
            y_list = [os.path.join(self.Data_file_dict['val_y'], x_name) for x_name in data_inf['y']]

        x_sequence_list = []
        y_sequence_list = []
        for x_path in x_list:
            data = np.load(x_path)
            x_sequence_list.append(data)
        for y_path in y_list:
            data = np.load(y_path)
            y_sequence_list.append(data)
        x_sequence = np.stack(x_sequence_list)
        y_sequence = np.stack(y_sequence_list)
        x_sequence = torch.from_numpy(x_sequence)
        y_sequence = torch.from_numpy(y_sequence)

        y_flow,y_speed,y_occ = y_sequence.split(1,dim=1)
        y_flow = y_flow.squeeze(dim=1)
        y_speed = y_speed.squeeze(dim=1)
        return x_sequence,y_flow,y_speed

    def __len__(self):
        return len(self.Data_inf)

class MySet(Dataset):
    def __init__(self, data, seq_len=36, pre_len=12):
        x, y = list(), list()
        for i in range(len(data) - seq_len - pre_len):
            x.append(np.array(data[i : i + seq_len]))
            y.append(np.array(data[i + seq_len : i + seq_len + pre_len]))
        x, y = np.array(x), np.array(y)
        y_flow, y_speed = y[:, :, :, :4], y[:, :, :, 4:8]
        self.x = torch.tensor(x, dtype=torch.float32)
        self.y_flow = torch.tensor(y_flow, dtype=torch.float32)
        self.y_speed = torch.tensor(y_speed, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        return self.x[idx], self.y_flow[idx], self.y_speed[idx]

if __name__ == '__main__':
    Data_file_dict = {'train_x':'../Dataset/train_x','train_y':'../Dataset/train_y',
                      'val_x':'../Dataset/val_x','val_y':'../Dataset/val_y',}
    Data_inf_path = '../Dataset/'
    train_inf = [json.loads(line) for line in open(Data_inf_path + 'train.json')][0]
    val_inf = [json.loads(line) for line in open(Data_inf_path + 'val.json')][0]

    train_loader = DataLoader(
        MyDataSet(Data_file_dict, train_inf,train_flag=True),
        batch_size=2, shuffle=True,
        num_workers=0,
        pin_memory=True)

    for epochs in range(1):
        for data in tqdm(train_loader, total=len(train_loader)):
            pass
