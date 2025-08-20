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

class MyDataSet_test(Dataset):
    def __init__(self,Data_file_dict:dict,Data_inf,train_flag=True,test_flag = False):
        self.Data_file_dict = Data_file_dict
        self.Data_inf = Data_inf
        self.train_flag = train_flag
        self.test_flag = test_flag

    def __getitem__(self, index):
        data_inf = self.Data_inf[index]
        if self.train_flag:
            x_list = [os.path.join(self.Data_file_dict['train_x'],x_name) for x_name in data_inf['x']]
            y_list = [os.path.join(self.Data_file_dict['train_y'], x_name) for x_name in data_inf['y']]

        if self.test_flag:
            x_list = [os.path.join(self.Data_file_dict['test_x'], x_name) for x_name in data_inf['x']]
            time_inf = data_inf['sample_index']
            week_inf = data_inf['Week_index']
            x_sequence_list = []
            for x_path in x_list:
                data = np.load(x_path)
                x_sequence_list.append(data)
            x_sequence = np.stack(x_sequence_list)
            x_sequence = torch.from_numpy(x_sequence)
            return x_sequence,time_inf,week_inf

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


if __name__ == '__main__':
    Data_file_dict = {'test_x':'../Dataset/test_x'}
    Data_inf_path = '../Dataset/'
    test_inf = [json.loads(line) for line in open(Data_inf_path + 'test.json')][0]

    train_loader = DataLoader(
        MyDataSet_test(Data_file_dict, test_inf,train_flag=False,test_flag=True),
        batch_size=1, shuffle=True,
        num_workers=0,
        pin_memory=True)

    Test_Data_list = []
    for epochs in range(1):
        for (x_sequence,time_inf,week_inf) in tqdm(train_loader, total=len(train_loader)):
            out_flow = torch.randn(1, 12, 4, 10)
            out_speed = torch.randn(1, 12, 12, 4, 10)
            Test_Data_inf = {'sample_index':time_inf,'week_inf':week_inf,'out_flow':out_flow,'out_speed':out_flow}
            Test_Data_list.append(Test_Data_inf)
            pass
        pass
