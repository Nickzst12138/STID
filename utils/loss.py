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

class node_flow_mae_rmse_Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        self.mae = nn.SmoothL1Loss()

    def forward(self, yhat, y):
        yhat, y = torch.sum(yhat,dim=-1),torch.sum(y,dim=-1)
        RMSE_loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        MAE_loss = self.mae(yhat,y)
        loss = RMSE_loss + MAE_loss
        return loss

class node_speed_mae_rmse_Loss(nn.Module):
    def __init__(self, eps=1e-6):
        super().__init__()
        self.mse = nn.MSELoss()
        self.eps = eps
        self.mae = nn.SmoothL1Loss()

    def forward(self, yhat, y):
        yhat, y = torch.mean(yhat,dim=-1),torch.mean(y,dim=-1)
        RMSE_loss = torch.sqrt(self.mse(yhat, y) + self.eps)
        MAE_loss = self.mae(yhat,y)
        loss = RMSE_loss + MAE_loss
        return loss

if __name__ == '__main__':
    label_flow, label_speed = torch.randn(1, 12, 10, 1), torch.randn(1, 12, 10, 1)
    loss = node_speed_mae_rmse_Loss()
    output = loss(label_flow,label_speed)