import torch
import torch.nn as nn
import numpy as np
from GCN import *
from einops.layers.torch import Rearrange


class RevIN(nn.Module):
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """Reversible Instance Normalization for Accurate Time-Series Forecasting
           against Distribution Shift, ICLR2021.

    Parameters
    ----------
    num_features: int, the number of features or channels.
    eps: float, a value added for numerical stability, default 1e-5.
    affine: bool, if True(default), RevIN has learnable affine parameters.
        """
        super().__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode:str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else:
            raise NotImplementedError('Only modes norm and denorm are supported.')
        return x

    def _init_params(self):
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps*self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x

def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    return normalized_laplacian

class MyMultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MyMultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.act = nn.PReLU()
        self.drop = nn.Dropout(p=0.3)

    def forward(self, in_data):
        # in_data: [B, D, N]
        out = self.act(self.fc1(in_data))
        out = self.fc2(self.drop(out))
        return out + in_data

class STID_WITH_FUSION(nn.Module):
    def __init__(self,model_param):
        super(STID_WITH_FUSION, self).__init__()
        model_kwargs = model_param
        # attributes
        self.num_nodes = model_kwargs['num_nodes']
        self.node_dim = model_kwargs['node_dim']
        self.input_len = model_kwargs['input_len']
        self.input_dim = model_kwargs['input_dim']
        self.embed_dim = model_kwargs['embed_dim']
        self.output_len = model_kwargs['output_len']
        self.num_layer = model_kwargs['num_layer']
        self.temp_dim_tid = model_kwargs['temp_dim_tid']
        self.temp_dim_diw = model_kwargs['temp_dim_diw']

        self.if_time_in_day = model_kwargs['if_T_i_D']
        self.if_day_in_week = model_kwargs['if_D_i_W']
        self.if_spatial = model_kwargs['if_node']

        # # spatial embeddings (nn.init.xavier_uniform_(self.node_emb))
        self.node_emb = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.node_dim)
        # temporal embeddings
        self.time_in_day_emb = nn.Embedding(num_embeddings=288, embedding_dim=self.temp_dim_tid)
        self.day_in_week_emb = nn.Embedding(num_embeddings=7, embedding_dim=self.temp_dim_diw)
        # embedding layer
        self.time_sires_emb_layer = nn.Linear(in_features=self.input_dim * self.input_len, out_features=self.embed_dim,
                                              bias=True)
        # encoding
        # self.hidden_dim = self.embed_dim + self.node_dim * int(self.if_spatial) + \
        #                   self.temp_dim_tid * int(self.if_time_in_day) + \
        #                   self.temp_dim_diw * int(self.if_day_in_week)
        self.hidden_dim = 192
        self.encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        # regression
        self.regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)
        self.fc_f = nn.Linear(self.output_len, 48)
        self.fc_s = nn.Linear(self.output_len, 48)

        self.flow_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim,bias=True)
        self.speed_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)
        self.occ_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)

        self.flow_conv = nn.Sequential(
                        nn.Conv2d(in_channels=36,out_channels=36,kernel_size=(3,3),stride=(1,1),padding=(1,1)),
                        nn.BatchNorm2d(36),
                        nn.ReLU()
        )
        self.speed_conv = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(36),
            nn.ReLU()
        )
        self.occ_conv = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(36),
            nn.ReLU()
        )

        self.all_feature_conv = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(36),
            nn.ReLU()
        )

    def forward(self, his_data):
        """Feed forward of STID.
        Args:
            his_data (torch.Tensor): history data with shape [B, L, N, C]
        Returns:
            torch.Tensor: prediction wit shape [B, L, N, C]
        """
        # prepare
        time_in_day_emb = None

        input_data = his_data[:,:,:,:-2]

        """
        2.第二次尝试，对整体先提特征
        input_data = self.all_feature_conv(input_data)
        """
        """
        第三次尝试，改变loss权重
        """

        flow_feature = input_data[:,:,:,0:4]
        speed_feature = input_data[:, :, :, 4:8]
        Occ_feature = input_data[:, :, :, 8:]

        batch_size, _, num_nodes, _ = input_data.shape


        """
        1.第一次尝试，加了CNN提特征
        flow_feature = self.flow_conv(flow_feature)
        speed_feature = self.speed_conv(speed_feature)
        Occ_feature = self.occ_conv(Occ_feature)
        """


        flow_feature = flow_feature.transpose(1, 2).contiguous()
        flow_feature = flow_feature.view(batch_size, num_nodes, -1)
        flow_time_series_emb = self.flow_time_series_emb(flow_feature)

        speed_feature = speed_feature.transpose(1, 2).contiguous()
        speed_feature = speed_feature.view(batch_size, num_nodes, -1)
        speed_time_series_emb = self.speed_time_series_emb(speed_feature)

        Occ_feature = Occ_feature.transpose(1, 2).contiguous()
        Occ_feature = Occ_feature.view(batch_size, num_nodes, -1)
        occ_time_series_emb = self.occ_time_series_emb(Occ_feature)

        if self.if_time_in_day:
            t_i_d_data = his_data[:, 1, :, -2].long()
            # (b, node_nums(time_in_day dim-len)) -> T^{TiD}(Nd x D)-> (b, node_nums(time_in_day dim-len), emb)
            time_in_day_emb = self.time_in_day_emb(t_i_d_data)

        day_in_week_emb = None
        if self.if_day_in_week:
            d_i_w_data = his_data[:, 1, :, -1].long()
            # (b, node_nums(day_in_week dim-len)) -> T^{DiW}(Nw x D)-> (b, node_nums(day_in_week dim-len), emb)
            day_in_week_emb = self.day_in_week_emb(d_i_w_data)

        # ts embedding

        # (b, L, num_nodes, channel) -> (b, num_nodes, L, channel) ->
        # (b, num_nodes, L, channel) -> (b, num_nodes, L * channel) (data || time-in-day || day-of-week)
        input_data = input_data.transpose(1, 2).contiguous()
        input_data = input_data.view(batch_size, num_nodes, -1)
        # (b, num_nodes, L * channel) -> FC_emb (L*channel, D) -> (b, num_nodes, emb)
        time_series_emb = self.time_sires_emb_layer(input_data)

        # node emb
        # (b, node_nums)
        nodes_indx = torch.Tensor([list(range(num_nodes)) for _ in range(batch_size)]).long().type_as(d_i_w_data)
        node_emb = []
        if self.if_spatial:
            node_emb.append(
                # (b, node_nums) -> E(N x D) -> (b, node_nums, emb)
                self.node_emb(nodes_indx)
            )

        # time embedding
        tem_emb = []
        if self.if_time_in_day:
            tem_emb.append(time_in_day_emb)
        if self.if_day_in_week:
            tem_emb.append(day_in_week_emb)

        hidden = torch.cat([flow_time_series_emb]+[speed_time_series_emb]+[occ_time_series_emb]+node_emb+tem_emb,dim=2)

        # concat (b, num_node, 32*4)
        # hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=2)
        # (b, num_node, 32*4) -> (b, num_node, 32*4)
        hidden = self.encoder(hidden)
        # (b, num_node, 32*4) -> (b, num_node, out_len)
        pred = self.regression_layer(hidden)
        # print('pred - pred.shape:', pred.shape)

        f_output = self.fc_f(pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        s_output = self.fc_s(pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)

        # return pred.transpose(1, 2).contiguous()
        return f_output,s_output



if __name__ == '__main__':
    model_param = {
        "num_nodes": 325,
        'input_len': 36,
        'input_dim': 12,
        'embed_dim': 32,
        'output_len': 12,
        'num_layer': 3,
        "if_node": True,
        'node_dim': 32,
        "if_T_i_D": True,
        "if_D_i_W": True,
        'temp_dim_tid': 32,
        'temp_dim_diw': 32,
        'device': 'CPU',
    }
    import sys
    sys.path.append("..")
    from Data_Load_TGCN import *
    from score import *
    from torch.utils.data import DataLoader

    val_data = np.load('tw2_train_2_8.npy')
    # val_data = np.load('train.npy')
    val_loader = DataLoader(MyDataSet(val_data), batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    adj = np.load('adj_mat.npy')
    model = STID_WITH_FUSION(model_param).cuda()
    for index,(x_sequence, y_flow,y_speed) in enumerate(val_loader):
        x_sequence, y_flow, y_speed = x_sequence.cuda(), y_flow.cuda(), y_speed.cuda()
        x_sequence = x_sequence.float()
        f_output, s_output = model(x_sequence)
        # score = get_score(f_output, s_output, y_flow, y_speed, mean=None, std=None)
        pass
    # for index,(x_sequence, y_flow,y_speed,aux_x_sequence) in enumerate(val_loader):
    #     out_put = model(x_sequence,aux_x_sequence)
    #     pass
