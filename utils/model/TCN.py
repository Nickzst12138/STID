import torch
import torch.nn as nn
import numpy as np
import sys
from ST import SpatioTemporalConv

from torch.utils.data import DataLoader
from torch.nn.utils import weight_norm

class Chomp1d(nn.Module):
    def __init__(self, chomp_size):
        super(Chomp1d, self).__init__()
        self.chomp_size = chomp_size

    def forward(self, x):
        return x[:, :, :-self.chomp_size].contiguous()


class TemporalBlock(nn.Module):
    def __init__(self, n_inputs, n_outputs, kernel_size, stride, dilation, padding, dropout=0.2):
        super(TemporalBlock, self).__init__()
        self.conv1 = weight_norm(nn.Conv1d(n_inputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp1 = Chomp1d(padding)
        self.relu1 = nn.ReLU()
        self.dropout1 = nn.Dropout(dropout)

        self.conv2 = weight_norm(nn.Conv1d(n_outputs, n_outputs, kernel_size,
                                           stride=stride, padding=padding, dilation=dilation))
        self.chomp2 = Chomp1d(padding)
        self.relu2 = nn.ReLU()
        self.dropout2 = nn.Dropout(dropout)

        self.net = nn.Sequential(self.conv1, self.chomp1, self.relu1, self.dropout1,
                                 self.conv2, self.chomp2, self.relu2, self.dropout2)
        self.downsample = nn.Conv1d(n_inputs, n_outputs, 1) if n_inputs != n_outputs else None
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.conv1.weight.data.normal_(0, 0.01)
        self.conv2.weight.data.normal_(0, 0.01)
        if self.downsample is not None:
            self.downsample.weight.data.normal_(0, 0.01)

    def forward(self, x):
        out = self.net(x)
        res = x if self.downsample is None else self.downsample(x)
        return self.relu(out + res)


class TemporalConvNet(nn.Module):
    def __init__(self, num_nodes, input_dims, num_channels, kernel_size=3, dropout=0.2):
        """
        Args:
            num_nodes: number of nodes in graph
            input_dims: input feature dimensions
            num_channels: list of channel sizes for each layer
            kernel_size: convolution kernel size
            dropout: dropout rate
        """
        super(TemporalConvNet, self).__init__()
        self.num_nodes = num_nodes
        self.input_dims = input_dims

        layers = []
        num_levels = len(num_channels)

        for i in range(num_levels):
            dilation_size = 2 ** i
            in_channels = input_dims if i == 0 else num_channels[i - 1]
            out_channels = num_channels[i]

            layers += [TemporalBlock(in_channels, out_channels, kernel_size,
                                     stride=1, dilation=dilation_size,
                                     padding=(kernel_size - 1) * dilation_size,
                                     dropout=dropout)]

        # self.network = nn.Sequential(*layers)
        self.temporal_layers = nn.Sequential(*layers)
        self.flow_cnn = nn.Conv2d(in_channels=36,out_channels=12,kernel_size=1)
        self.speed_cnn = nn.Conv2d(in_channels=36, out_channels=12, kernel_size=1)


    def forward(self, x):
        tf_feature = x[:, :, :, :12]
        batch_size, T, N, D = tf_feature.shape
        x = tf_feature.permute(0, 2, 1, 3)
        x = x.reshape(batch_size * N, T, D).permute(0, 2, 1)
        out = self.temporal_layers(x)
        out = out.permute(0, 2, 1)
        out = out.reshape(batch_size, N, T, -1)
        out = out.permute(0, 2, 1, 3)
        f_output = self.flow_cnn(out)
        s_output = self.speed_cnn(out)


        return f_output,s_output




if __name__ == '__main__':
    sys.path.append("..")
    from Data_Load_TGCN import MyDataSet

    val_data = np.load('val.npy')
    val_loader = DataLoader(MyDataSet(val_data), batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    adj = np.load('adj_mat.npy')
    # model = TemporalConvNet(adj)
    model = TemporalConvNet(num_nodes=40,
                            input_dims=12,
                            num_channels=[64, 32, 4])
    for index,(x_sequence, y_flow,y_speed) in enumerate(val_loader):
        out_put = model(x_sequence)
