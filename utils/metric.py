import numpy as np
import torch
import torch.nn.functional as F
import pandas as pd
import datetime
import torch.nn as nn
import torch

class RMSE_MAE_Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        self.mae = nn.SmoothL1Loss()

    def forward(self, yhat, y):
        RMSE_loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        MAE_loss = self.mae(yhat,y)
        loss = RMSE_loss + MAE_loss
        return loss

def calculate_score(f, s, y_f, y_s, std=None, mean=None):
    out_flow = f
    out_speed = s
    y_flow = y_f
    y_speed = y_s
    std = torch.from_numpy(std).cuda()
    mean = torch.from_numpy(mean).cuda()
    if std is not None and mean is not None:
        out_flow = out_flow * std[:4] + mean[:4]
        out_speed = out_speed * std[4:8] + mean[4:8]
        y_flow = y_flow * std[:4] + mean[:4]
        y_speed = y_speed * std[4:8] + mean[4:8]
    B, T, D, L = out_flow.size()  # batch, time, detector, lane
    sum_flow = calculate_flow(out_flow)  # flow_{1..l}
    sum_speed = calculate_speed(out_flow, out_speed, sum_flow)  # flow_{1..l} * speed_{1..l}
    y_sum_flow = calculate_flow(y_flow)  # flow_{1..l}
    y_sum_speed = calculate_speed(y_flow, y_speed, y_sum_flow)  # flow_{1..l} * speed_{1..l}
    p = [[sum_flow, sum_speed], [y_sum_flow, y_sum_speed]]  # [flow_{1..l}, flow_{1..l} * speed_{1..l} / flow_{1..l}]
    score = 0.0

    for b in range(B):  # b: 1..B batch
        for i in range(len(p)):  # p: 1..P metrics
            for d in range(D):  # d: 1..D detector loop
                score += (mae(p[0][i][b, :, d], p[1][i][b, :, d]) + rmse(p[0][i][b, :, d], p[1][i][b, :, d])) / torch.mean(p[1][i][b, :, d])
    return score / B

def metrics_flow_score(output_flow,label_flow,std=None, mean=None):
    if std is not None and mean is not None:
        output_flow = output_flow * std[:4] + mean[:4]
        label_flow = label_flow * std[:4] + mean[:4]

    node_output_flow,node_label_flow = torch.sum(output_flow,dim=-1),torch.sum(label_flow,dim=-1)
    mae = torch.mean(torch.abs(node_output_flow - node_label_flow))
    mse = torch.mean((node_output_flow - node_label_flow) ** 2)
    rmse = torch.sqrt(mse)
    flow_score = mae + rmse
    return flow_score

def metrics_speed_score(output_speed,label_speed,std=None, mean=None):
    if std is not None and mean is not None:
        output_speed = output_speed * std[4:8] + mean[4:8]
        label_speed = label_speed * std[4:8] + mean[4:8]
    node_output_speed,node_label_speed = torch.sum(output_speed,dim=-1),torch.sum(label_speed,dim=-1)
    mae = torch.mean(torch.abs(node_output_speed - node_label_speed))
    mse = torch.mean((node_output_speed - node_label_speed) ** 2)
    rmse = torch.sqrt(mse)
    flow_score = mae + rmse
    return flow_score

def metrics_node_flow_score(output_flow,label_flow,std=None, mean=None):
    if std is not None and mean is not None:
        output_flow = output_flow * torch.mean(std[:4]) + torch.mean(mean[:4])
        label_flow = label_flow * torch.mean(std[:4]) + torch.mean(mean[:4])
    bs,_,_,_ = output_flow.shape
    output_flow = output_flow.reshape(bs,-1)
    label_flow = label_flow.reshape(bs,-1)
    mae = torch.mean(torch.abs(output_flow - label_flow))
    mse = torch.mean((output_flow - label_flow) ** 2)
    rmse = torch.sqrt(mse)
    flow_score = mae + rmse
    return flow_score

def metrics_node_speed_score(output_speed,label_speed,std=None, mean=None):
    if std is not None and mean is not None:
        output_speed = output_speed * torch.mean(std[4:8]) + torch.mean(mean[4:8])
        label_speed = label_speed * torch.mean(std[4:8]) + torch.mean(mean[4:8])
    node_output_speed,node_label_speed = torch.sum(output_speed,dim=-1),torch.sum(label_speed,dim=-1)
    mae = torch.mean(torch.abs(node_output_speed - node_label_speed))
    mse = torch.mean((node_output_speed - node_label_speed) ** 2)
    rmse = torch.sqrt(mse)
    flow_score = mae + rmse
    return flow_score

def calculate_flow(data):
    B, T, D, L = data.size()  # batch, time, detector, lane
    return torch.sum(data, dim=3)


def calculate_speed(flow, speed, sum_flow):
    B, T, D, L = speed.size()  # batch, time, detector, lane
    flow_mul_speed = torch.sum(flow * speed, dim=3)
    return flow_mul_speed / (sum_flow + 1e-10)


def mae(y_, y):
    return F.l1_loss(y_.float(), y.float()).item()


def rmse(y_, y):
    return torch.sqrt(F.mse_loss(y_.float(), y.float())).item()


def calculate_flow_speed(data):
    ret = pd.DataFrame({})

    flow = calculate_flow(data['out_flow']).squeeze()  # 计算流量
    speed = calculate_speed(data['out_flow'], data['out_speed'], flow).squeeze()  # 计算速度
    result = torch.empty(flow.shape[0], flow.shape[1] + speed.shape[1])
    result[:, ::2] = flow
    result[:, 1::2] = speed  # 拼接, 10个车道的流量速度

    start = ["08:00", "12:30", "17:00"]
    end = ["08:55", "13:25", "17:55"]
    phase = -1
    if "morning" in str(data['sample_index'][0]):
        phase = 0
    elif "nooning" in str(data['sample_index'][0]):
        phase = 1
    elif "afternoon" in str(data['sample_index'][0]):
        phase = 2
    assert phase >= 0

    start_time = datetime.datetime.strptime(start[phase], "%H:%M")
    end_time = datetime.datetime.strptime(end[phase], "%H:%M")
    interval = datetime.timedelta(minutes=5)

    time_list = []

    current_time = start_time
    while current_time <= end_time:
        time_list.append(current_time.strftime("%H:%M:%S"))
        current_time += interval

    ret['Time'] = time_list
    ret['Week'] = int(data['week_inf'][0])
    columns = []
    LOOP_NUM = 10
    for i in range(1, LOOP_NUM + 1):
        columns.append('Loop{} Flow (Veh/h)'.format(str(i)))
        columns.append('Loop{} Speed (km/h)'.format(str(i)))
    df = pd.DataFrame(result.numpy(), columns=columns)

    return pd.concat([ret, df], axis=1)

def get_score(flow_pred, speed_pred, flow_true, speed_true, mean=None, std=None):
    periods = flow_pred.shape[0]



    if std is not None and mean is not None:
        flow_true = flow_true * std[:4] + mean[:4]
        flow_pred = flow_pred * std[:4] + mean[:4]
        speed_pred = speed_pred * std[4:8] + mean[4:8]
        speed_true = speed_true * std[4:8] + mean[4:8]



    # n * 12 * 10
    final_flow_true = torch.sum(flow_true, dim=3)  # 流量预测值，sum(flow)
    final_flow_pred = torch.sum(flow_pred, dim=3)  # 流量预测值，sum(flow)

    ttl_true = torch.einsum('abcd,abcd->abc', flow_true, speed_true)
    ttl_pred = torch.einsum('abcd,abcd->abc', flow_pred, speed_pred)

    final_speed_true = ttl_true / final_flow_true
    final_speed_pred = ttl_pred / final_flow_pred

    has_nan = torch.isnan(final_flow_true)
    ndices = torch.nonzero(has_nan)

    # flow metrics n * 10
    flow_squared_diff = torch.square(final_flow_true - final_flow_pred)
    flow_mse = torch.mean(flow_squared_diff, dim=1)
    flow_rmse = torch.sqrt(torch.mean(flow_squared_diff, dim=1))
    flow_absolute_diff = torch.abs(final_flow_true - final_flow_pred)
    flow_mae = torch.mean(flow_absolute_diff, dim=1)
    flow_aver = torch.mean(final_flow_true, dim=1)
    flow_score = torch.sum((flow_mae + flow_rmse) / flow_aver) / periods * 21



    # speed metrics
    speed_squared_diff = torch.square(final_speed_true - final_speed_pred)
    speed_mse =  torch.mean(speed_squared_diff, dim=1)
    speed_rmse = torch.sqrt(torch.mean(speed_squared_diff, dim=1))
    speed_absolute_diff = torch.abs(final_speed_true - final_speed_pred)
    speed_mae = torch.mean(speed_absolute_diff, dim=1)
    speed_aver = torch.mean(final_speed_true, dim=1)


    speed_score = torch.sum((speed_mae + speed_rmse) / speed_aver) / periods * 21
    test = torch.sum((speed_mae + speed_rmse))
    score = flow_score + speed_score
    return score, flow_score, speed_score

if __name__ == '__main__':
    std = np.array([379.88216046, 502.45335188, 467.92877809, 454.46912642,
        13.44062786,  16.90991173,  18.70948768,  17.15657618,
         3.68793619,   5.61726124,   6.5255286 ,   6.04380136,
        83.13793759,   2.03442594])
    mean = np.array([563.74481096, 871.65727479, 823.68168885, 721.41552614,
       110.29007829,  99.93975919,  95.20205862,  89.59848312,
         3.90788291,   6.50432123,   6.98355903,   6.35010417,
       143.5       ,   3.        ])
    std = torch.from_numpy(std)
    mean = torch.from_numpy(mean)
    out_flow = torch.randn(2,12,10,1)
    label_flow = torch.randn(2, 12, 10, 1)
    metrics_node_flow_score(out_flow,label_flow,mean=mean,std=std)
