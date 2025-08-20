# -*- coding:utf-8 -*-
from torch.utils.data import Dataset
from torch.utils.data import DataLoader
from torchvision import transforms, datasets
import os
# import imageio as io
# import cv2
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
    def __init__(self, data, seq_len=36, pre_len=12, input_dim=16, test_flag=False):
        self.test_flag = test_flag
        if test_flag:
            self.x = torch.tensor(data, dtype=torch.float32).reshape(-1, seq_len, 10, input_dim)
        else:
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
        if self.test_flag:
            return self.x[idx]
        else:
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
