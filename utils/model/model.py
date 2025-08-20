# -*- coding:utf-8 -*-
import os
import torch.nn as nn
import torch
from ST import *


class My_model(nn.Module):
    def __init__(self):
        super(My_model, self).__init__()
        self.FT_conv = SpatioTemporalConv(in_channels=3,out_channels=15)
        self.temporal_encoder_1 = nn.Sequential(
            nn.Conv3d(in_channels=15, out_channels=15, kernel_size=(5, 1, 1),stride=(2, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(15),
            nn.ReLU()
        )
        self.temporal_encoder_2 = nn.Sequential(
            nn.Conv3d(in_channels=15, out_channels=32, kernel_size=(5, 1, 1), stride=(2, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(32),
            nn.ReLU()
        )
        self.temporal_encoder_3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(6, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.temporal_encoder_3 = nn.Sequential(
            nn.Conv3d(in_channels=32, out_channels=64, kernel_size=(6, 1, 1), stride=(1, 1, 1), padding=(0, 0, 0)),
            nn.BatchNorm3d(64),
            nn.ReLU()
        )
        self.Spatio_encoder_line_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=( 2, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.Spatio_encoder_line_2 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=64, kernel_size=(3, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(64),
            nn.ReLU()
        )
        self.Spatio_encoder_road_1 = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=(1, 3), stride=(1, 2), padding=(0, 0)),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.ST_1_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=128,out_channels=64,kernel_size=(2,2),stride=(1,2),padding=(0,0)),
            # nn.BatchNorm2d(self.config.hidden_dim//8, affine=True),
            nn.ReLU()
        )
        self.ST_2_decoder = nn.Sequential(
            nn.ConvTranspose2d(in_channels=64, out_channels=32, kernel_size=(2, 3), stride=(2, 1), padding=(0, 0)),
            # nn.BatchNorm2d(self.config.hidden_dim//8, affine=True),
            nn.ReLU()
        )
        self.Inter_encoder = nn.Sequential(
            nn.Conv2d(in_channels=64, out_channels=32, kernel_size=1, stride=1),
            # nn.BatchNorm2d(self.config.hidden_dim//8, affine=True),
            nn.ReLU()
        )

        self.encoder_to_speed = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=12, kernel_size=1, stride=1),
            # nn.BatchNorm2d(self.config.hidden_dim//8, affine=True),
            nn.ReLU()
        )
        self.encoder_to_flow = nn.Sequential(
            nn.Conv2d(in_channels=32, out_channels=12, kernel_size=1, stride=1),
            # nn.BatchNorm2d(self.config.hidden_dim//8, affine=True),
            nn.ReLU()
        )
    def forward(self, x):
        """
        x : (2,3,36,4,10) 2是batch
        流程：
        编码：
            1.FT_conv == >(2,3,36,4,10),这里对时间维度进行卷积，并且对空间进行3*3卷积，通道数由3变到15
            2.fusion_temporal_encoder_1==>(2,15,16,4,10) 对时间维度进行第一次卷积
            3.fusion_temporal_encoder_2==>(2,32,6,4,10) 对时间维度进行第二次卷积
            4.fusion_temporal_encoder_3==>(2,64,1,4,10) 对时间维度进行第三次卷积
            上面三次卷积都没有涉及到空间特征卷积，用的是1*1卷积

            保留中间特征，后面融合使用：
            Inter_x = self.Inter_encoder(x) ==> (2,32,4,10)
            接下来是空间卷积：
            5.Spatio_encoder_line_1 ==> (2,64,3,10) 这个是对相邻车道信息进行卷积（第一次）
            6.Spatio_encoder_line_2 ==> (2,64,3,10) 这个是对相邻车道信息进行卷积（第二次）
            7.Spatio_encoder_road_1 ==> (2,128,1,4) 这个是对车道方向信息进行卷积（第一次）
        解码：
            8.ST_1_decoder:==> （2，64，2，8）
            9.ST_2_decoder:==> （2，32，4，10）
        到y的大小：
            10.encoder_to_flow ==>（12，4，10）
            11.encoder_to_speed ==>（12，4，10）
        """
        x = x.permute(0, 2, 1, 3, 4)

        x = self.FT_conv(x)
        x = self.temporal_encoder_1(x)
        x = self.temporal_encoder_2(x)
        x = self.temporal_encoder_3(x)
        x = x.squeeze(dim=2)
        Inter_x = self.Inter_encoder(x)
        x = self.Spatio_encoder_line_1(x)
        x = self.Spatio_encoder_line_2(x)
        x = self.Spatio_encoder_road_1(x)

        x = self.ST_1_decoder(x)
        x = self.ST_2_decoder(x)

        x = torch.add(x,Inter_x)
        out_flow = self.encoder_to_flow(x)
        out_speed = self.encoder_to_speed(x)
        return out_flow,out_speed


if __name__ == '__main__':
    x = torch.randn(2,3,36,4,10)  #分别表示 batch、channel、F_number,H,W（训练中要调位置）
    label_flow,label_speed = torch.randn(2,12,4,10),torch.randn(2,12,4,10)
    model = My_model()
    out_flow,out_speed = model(x)
    criterion = nn.MSELoss()
    loss = criterion(out_flow, label_flow)
    pass