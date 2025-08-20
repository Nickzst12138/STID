import torch
import torch.nn as nn
import torch.nn.functional as F

class GCNLayer(nn.Module):
    def __init__(self, in_features, out_features):
        super(GCNLayer, self).__init__()
        self.linear = nn.Linear(in_features, out_features)

    def forward(self, x, adj):
        x = torch.matmul(adj, x)  # 执行图卷积操作
        x = self.linear(x)  # 线性变换
        x = F.relu(x)  # 使用ReLU激活函数
        return x

class Speed_GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,time_step):
        super(Speed_GCN, self).__init__()
        self.time_step = time_step
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, out_features)

    def forward(self, x, adj):
        speed_adj = []
        for i in range(self.time_step):
            time_speed_feature = x[:,i,:,:]
            x1 = self.layer1(time_speed_feature, adj)
            x2 = self.layer2(x1, adj)
            speed_adj.append(x2)
        speed_adj = torch.stack(speed_adj,dim=1)
        return speed_adj

class occ_GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,time_step):
        super(occ_GCN, self).__init__()
        self.time_step = time_step
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, out_features)

    def forward(self, x, adj):
        speed_adj = []
        for i in range(self.time_step):
            time_speed_feature = x[:,i,:,:]
            x1 = self.layer1(time_speed_feature, adj)
            x2 = self.layer2(x1, adj)
            speed_adj.append(x2)
        speed_adj = torch.stack(speed_adj,dim=1)
        return speed_adj

class flow_GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,time_step):
        super(flow_GCN, self).__init__()
        self.time_step = time_step
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, out_features)

    def forward(self, x, adj):
        speed_adj = []
        for i in range(self.time_step):
            time_speed_feature = x[:,i,:,:]
            x1 = self.layer1(time_speed_feature, adj)
            x2 = self.layer2(x1, adj)
            speed_adj.append(x2)
        speed_adj = torch.stack(speed_adj,dim=1)
        return speed_adj

class All_GCN(nn.Module):
    def __init__(self, in_features, hidden_features, out_features,time_step):
        super(All_GCN, self).__init__()
        self.time_step = time_step
        self.layer1 = GCNLayer(in_features, hidden_features)
        self.layer2 = GCNLayer(hidden_features, out_features)

    def forward(self, x, adj):
        speed_adj = []
        for i in range(self.time_step):
            time_speed_feature = x[:,i,:,:]
            x1 = self.layer1(time_speed_feature, adj)
            x2 = self.layer2(x1, adj)
            speed_adj.append(x2)
        speed_adj = torch.stack(speed_adj,dim=1)
        return speed_adj