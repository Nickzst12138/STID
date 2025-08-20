import torch
import torch.nn.functional as F
import pandas as pd
import datetime


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


if __name__ == '__main__':
    pass
