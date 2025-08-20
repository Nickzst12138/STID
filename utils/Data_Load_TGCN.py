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
    def __init__(self, data, seq_len=36, pre_len=12, input_dim=16, test_flag=False, shift=False):
        self.test_flag = test_flag
        """
        idx = [0, 16, 4, 20, 8, 24,
               1, 17, 5, 21, 9, 25,
               2, 18, 6, 22, 10, 26,
               3, 19, 7, 23, 11, 27,
               12, 13, 14, 15]
        """

        idx = [0, 14, 26, 38, 4, 18, 30, 42, 8, 22, 34, 46,
               1, 15, 27, 39, 5, 19, 31, 43, 9, 23, 35, 47,
               2, 16, 28, 40, 6, 20, 32, 44, 10, 24, 36, 48,
               3, 17, 29, 41, 7, 21, 33, 45, 11, 25, 37, 49,
               12, 13]
        # idx = [0, 14, 26, 38, 4, 18, 30, 42, 8, 22, 34, 46,
        #        1, 15, 27, 39, 5, 19, 31, 43, 9, 23, 35, 47,
        #        2, 16, 28, 40, 6, 20, 32, 44, 10, 24, 36, 48,
        #        3, 17, 29, 41, 7, 21, 33, 45, 11, 25, 37, 49,
        #        12, 13, 50, 51, 52]

        if test_flag:
            self.x = torch.tensor(data, dtype=torch.float32).reshape(-1, seq_len, 10, input_dim)
            self.x = self.x[:, :, :, idx] if shift else self.x
        else:
            x, y = list(), list()
            for i in range(len(data) - seq_len - pre_len):
                x.append(np.array(data[i: i + seq_len]))
                y.append(np.array(data[i + seq_len: i + seq_len + pre_len]))
            x, y = np.array(x), np.array(y)
            y_flow, y_speed = y[:, :, :, :4], y[:, :, :, 4:8]
            self.x = torch.tensor(x, dtype=torch.float32)

            self.x = self.x[:, :, :, idx] if shift else self.x
            self.y_flow = torch.tensor(y_flow, dtype=torch.float32)
            self.y_speed = torch.tensor(y_speed, dtype=torch.float32)

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.test_flag:
            return self.x[idx]
        else:
            return self.x[idx], self.y_flow[idx], self.y_speed[idx]


class Dataset_with_his(Dataset):
    def __init__(self, data, seq_len=36, pre_len=12, input_dim=16, discard_week=1, split_ratio=0.8, test_flag=True):
        """
        Args:
            data:总数据信息
            seq_len: 在预测前使用的近邻的预测步长
            pre_len: 要预测的数据步长
            input_dim: 数据的维度 4*[flow+speed+occ]+其他
            discard_times: 将多少天的数据丢弃，作为历史数据使用
            his_times:加上前几天的y数据
        """

        self.test_flag = test_flag
        self.discard_week = discard_week
        self.his_data_len = self.discard_week * 288 * 7
        self.test_len = 36 * 21
        self.data = data
        assert discard_week * 288 * 7 <= len(self.data) - self.test_len
        self.train_val_inf = self.data[:len(self.data)-self.test_len]
        self.test_inf = self.data[-self.test_len:]
        self.train_last_week_inf = self.train_val_inf[-self.his_data_len:]
        self.last_weeks = None

        if self.test_flag:
            test_morning_begin, test_morning_end = 96, 108
            test_noon_begin, test_noon_end = 150, 162
            test_after_begin, test_after_end = 204, 216

            weeks_inf = []
            for i in range(7 * self.discard_week):
                weeks_inf.append(np.array(self.train_last_week_inf[i * 288 + test_morning_begin: i * 288 + test_morning_end]))
                weeks_inf.append(np.array(self.train_last_week_inf[i * 288 + test_noon_begin: i * 288 + test_noon_end]))
                weeks_inf.append(np.array(self.train_last_week_inf[i * 288 + test_after_begin: i * 288 + test_after_end]))

            x = list()
            weeks_list = [[] for _ in range(self.discard_week)]
            for i in range(self.discard_week):
                for j in range(21):
                    if i == 0:
                        x_begin = j * seq_len
                        x_end = x_begin + seq_len
                        x.append(np.array(self.test_inf[x_begin: x_end]))
                    weeks_list[i].append(weeks_inf[i*21+j])

            x = np.array(x)
            self.x = torch.tensor(x, dtype=torch.float32)
            self.last_weeks = [torch.tensor(w, dtype=torch.float32) for w in weeks_list]

        else:
            x, y = list(), list()
            weeks_list = [[] for _ in range(self.discard_week)]

            WEEK_COUNT = 288 * 7
            for i in range(len(self.train_val_inf) - seq_len - pre_len - self.his_data_len):
                x_begin = self.his_data_len + i
                x_end = x_begin + seq_len
                y_end = x_end + pre_len
                x.append(np.array(self.train_val_inf[x_begin: x_end]))
                y.append(np.array(self.train_val_inf[x_end: y_end]))
                for j in range(self.discard_week):
                    t_begin = i + WEEK_COUNT * j + seq_len
                    t_end = t_begin + pre_len
                    weeks_list[j].append(np.array(self.train_val_inf[t_begin: t_end]))

            x, y = np.array(x), np.array(y)
            y_flow, y_speed = y[:, :, :, :4], y[:, :, :, 4:8]
            self.x = torch.tensor(x, dtype=torch.float32)
            self.y_flow = torch.tensor(y_flow, dtype=torch.float32)
            self.y_speed = torch.tensor(y_speed, dtype=torch.float32)
            self.last_weeks = [torch.tensor(w, dtype=torch.float32) for w in weeks_list]
            pass

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.test_flag:
            return self.x[idx], [w[idx] for w in self.last_weeks]
        else:
            return self.x[idx], self.y_flow[idx], self.y_speed[idx], [w[idx] for w in self.last_weeks]


class LeakDataSet(Dataset):
    def __init__(self, data, seq_len=36, pre_len=12, input_dim=15, interval=6, test_flag=False, discard_week=1):
        self.test_flag = test_flag
        self.last_weeks = None
        self.discard_week = discard_week
        self.his_data_len = self.discard_week * 288 * 7
        self.test_len = 36 * 21
        self.data = data
        assert discard_week * 288 * 7 <= len(self.data) - self.test_len
        self.train_val_inf = data[:-self.test_len]
        self.train_last_week_inf = self.train_val_inf[-self.his_data_len:]
        self.test_inf = self.data[-self.test_len:]

        if test_flag:
            test_morning_begin, test_morning_end = 96, 108
            test_noon_begin, test_noon_end = 150, 162
            test_after_begin, test_after_end = 204, 216

            weeks_inf = []
            for i in range(7 * self.discard_week):
                weeks_inf.append(np.array(self.train_last_week_inf[i * 288 + test_morning_begin: i * 288 + test_morning_end]))
                weeks_inf.append(np.array(self.train_last_week_inf[i * 288 + test_noon_begin: i * 288 + test_noon_end]))
                weeks_inf.append(np.array(self.train_last_week_inf[i * 288 + test_after_begin: i * 288 + test_after_end]))

            weeks_list = [[] for _ in range(self.discard_week)]
            for i in range(self.discard_week):
                for j in range(21):
                    weeks_list[i].append(weeks_inf[i*21+j])

            self.x = torch.tensor(self.test_inf, dtype=torch.float32).reshape(-1, seq_len, 10, input_dim)  # 21,36,10,input_dim
            self.aux = self.x[1:]  # 20,36,10,input_dim
            self.x = self.x[:-1]
            self.last_weeks = [torch.tensor(w[:-1], dtype=torch.float32) for w in weeks_list]  # discard_week, 20, 12, 10, input_dim

        else:
            WEEK_COUNT = 288 * 7

            x, y, fut = list(), list(), list()
            weeks_list = [[] for _ in range(self.discard_week)]

            for i in range(len(self.train_val_inf) - seq_len - pre_len - interval - seq_len - self.his_data_len):
                x.append(np.array(self.train_val_inf[i : i + seq_len]))
                y.append(np.array(self.train_val_inf[i + seq_len : i + seq_len + pre_len]))
                fut.append(np.array(data[i + seq_len + pre_len + interval: i + seq_len + pre_len + interval + seq_len]))
                for j in range(self.discard_week):
                    t_begin = i + WEEK_COUNT * j + seq_len
                    t_end = t_begin + pre_len
                    weeks_list[j].append(np.array(self.train_val_inf[t_begin: t_end]))

            x, y, fut = np.array(x), np.array(y), np.array(fut)
            y_flow, y_speed = y[:, :, :, :4], y[:, :, :, 4:8]
            self.x = torch.tensor(x, dtype=torch.float32)
            self.aux = torch.tensor(fut, dtype=torch.float32)
            self.y_flow = torch.tensor(y_flow, dtype=torch.float32)
            self.y_speed = torch.tensor(y_speed, dtype=torch.float32)
            self.last_weeks = [torch.tensor(w, dtype=torch.float32) for w in weeks_list]

    def __len__(self):
        return len(self.x)

    def __getitem__(self, idx):
        if self.test_flag:
            return self.x[idx], self.aux[idx], [w[idx] for w in self.last_weeks]
        else:
            return self.x[idx], self.y_flow[idx], self.y_speed[idx], self.aux[idx], [w[idx] for w in self.last_weeks]


if __name__ == '__main__':
    # train_data = np.load('../Dataset/train_2_8.npy')
    # val_data = np.load('../Dataset/val_2_8.npy')

    data_inf = np.load('../Dataset/all_inf.npy')

    # train_loader = DataLoader(MyDataSet(train_data), batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
    # train_loader = DataLoader(Dataset_with_his(data_inf, test_flag=False, discard_week=3), batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
    # test_loader = DataLoader(Dataset_with_his(data_inf, test_flag=True, discard_week=6), batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
    # val_loader = DataLoader(ifMyDataSet(val_data), batch_size=512, shuffle=True, num_workers=0, pin_memory=True)

    train_loader = DataLoader(Dataset_with_his(data_inf, discard_week=6), batch_size=512, shuffle=True, num_workers=0, pin_memory=True)
    # test_loader = DataLoader(LeakDataSet(data_inf, discard_week=5, test_flag=True), batch_size=512, shuffle=True, num_workers=0, pin_memory=True)

    for index, (data) in enumerate(train_loader):
        print(data)
        pass
