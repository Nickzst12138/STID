import torch
import torch.nn as nn
import numpy as np
from GCN import *
from einops.layers.torch import Rearrange


# from My_model.utils.Data_Load_TGCN import Dataset_with_his


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

    def forward(self, x, mode: str):
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
            x = x / (self.affine_weight + self.eps * self.eps)
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


class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, input_dim: int = 64, bias: float = 0.0):
        super(TGCNGraphConvolution, self).__init__()
        self._num_gru_units = num_gru_units
        self._output_dim = output_dim
        self._bias_init_value = bias
        self._input_dim = input_dim
        self.register_buffer("laplacian", calculate_laplacian_with_self_loop(torch.FloatTensor(adj)))
        self.weights = nn.Parameter(torch.FloatTensor(self._num_gru_units + self._input_dim, self._output_dim))
        self.biases = nn.Parameter(torch.FloatTensor(self._output_dim))
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.weights)
        nn.init.constant_(self.biases, self._bias_init_value)

    def forward(self, inputs, hidden_state):
        batch_size, num_nodes, input_dim = inputs.shape
        # inputs (batch_size, num_nodes) -> (batch_size, num_nodes, 1)
        # inputs = inputs.reshape((batch_size, num_nodes, 1))
        # hidden_state (batch_size, num_nodes, num_gru_units)
        hidden_state = hidden_state.reshape((batch_size, num_nodes, self._num_gru_units))
        # [x, h] (batch_size, num_nodes, num_gru_units + 1)
        concatenation = torch.cat((inputs, hidden_state), dim=2)
        # [x, h] (num_nodes, num_gru_units + 1, batch_size)
        concatenation = concatenation.transpose(0, 1).transpose(1, 2)
        # [x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        concatenation = concatenation.reshape((num_nodes, (self._num_gru_units + input_dim) * batch_size))
        # A[x, h] (num_nodes, (num_gru_units + 1) * batch_size)
        a_times_concat = self.laplacian @ concatenation
        # A[x, h] (num_nodes, num_gru_units + 1, batch_size)
        a_times_concat = a_times_concat.reshape((num_nodes, self._num_gru_units + input_dim, batch_size))
        # A[x, h] (batch_size, num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.transpose(0, 2).transpose(1, 2)
        # A[x, h] (batch_size * num_nodes, num_gru_units + 1)
        a_times_concat = a_times_concat.reshape((batch_size * num_nodes, self._num_gru_units + input_dim))
        # A[x, h]W + b (batch_size * num_nodes, output_dim)
        outputs = a_times_concat @ self.weights + self.biases
        # A[x, h]W + b (batch_size, num_nodes, output_dim)
        outputs = outputs.reshape((batch_size, num_nodes, self._output_dim))
        # A[x, h]W + b (batch_size, num_nodes * output_dim)
        outputs = outputs.reshape((batch_size, num_nodes * self._output_dim))
        return outputs

    @property
    def hyperparameters(self):
        return {
            "num_gru_units": self._num_gru_units,
            "output_dim": self._output_dim,
            "bias_init_value": self._bias_init_value,
        }


class TGCNCell(nn.Module):
    def __init__(self, adj, input_dim: int, hidden_dim: int):
        super(TGCNCell, self).__init__()
        self._input_dim = input_dim
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.graph_conv1 = TGCNGraphConvolution(self.adj, self._hidden_dim, self._hidden_dim * 2, bias=1.0)
        self.graph_conv2 = TGCNGraphConvolution(self.adj, self._hidden_dim, self._hidden_dim)

    def forward(self, inputs, hidden_state):
        # [r, u] = sigmoid(A[x, h]W + b)
        # [r, u] (batch_size, num_nodes * (2 * num_gru_units))
        concatenation = torch.sigmoid(self.graph_conv1(inputs, hidden_state))
        # r (batch_size, num_nodes, num_gru_units)
        # u (batch_size, num_nodes, num_gru_units)
        r, u = torch.chunk(concatenation, chunks=2, dim=1)
        # c = tanh(A[x, (r * h)W + b])
        # c (batch_size, num_nodes * num_gru_units)
        c = torch.tanh(self.graph_conv2(inputs, r * hidden_state))
        # h := u * h + (1 - u) * c
        # h (batch_size, num_nodes * num_gru_units)
        new_hidden_state = u * hidden_state + (1.0 - u) * c
        return new_hidden_state, new_hidden_state

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGCN(nn.Module):
    def __init__(self, adj, hidden_dim: int = 64, **kwargs):
        super(TGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(16, self._hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self._hidden_dim // 2, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
        )
        self.fc_f = nn.Linear(self._hidden_dim, 48)
        self.fc_s = nn.Linear(self._hidden_dim, 48)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes, input_dim = inputs.shape
        inputs = self.mlp(inputs)
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        f_output = self.fc_f(output).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        s_output = self.fc_s(output).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        # output = self.fc(output).reshape((batch_size, num_nodes, 2, 12, 4))
        # output = output.transpose(1, 2).transpose(0, 1).transpose(2, 3)  # 2, bs, 12, 10, 4
        return f_output, s_output

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class TGCN_WITH_CNN(nn.Module):
    def __init__(self, adj, hidden_dim: int = 64, **kwargs):
        super(TGCN_WITH_CNN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(16, self._hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self._hidden_dim // 2, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
        )
        self.fc_f = nn.Linear(self._hidden_dim, 48)
        self.fc_s = nn.Linear(self._hidden_dim, 48)

        self.time_emb = nn.Embedding(num_embeddings=288, embedding_dim=32)
        self.week_emb = nn.Embedding(num_embeddings=49, embedding_dim=32)

    def forward(self, inputs):
        """
        time_feature：取出时间维度特征，原本是两维[sin,cos]
        week_feature：取出星期维度特征，原本是两维[sin,cos]
        1.对时间、星期进行词嵌入
        """
        batch_size, seq_len, num_nodes, input_dim = inputs.shape

        # 1 日期、星期 词嵌入
        time_feature = inputs[..., -4:-3].long()
        week_feature = inputs[..., -1:].long()
        # time_emd = self.time_emb(time_feature)
        week_emd = self.week_emb(week_feature)

        inputs = self.mlp(inputs)
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        f_output = self.fc_f(output).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        s_output = self.fc_s(output).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        # output = self.fc(output).reshape((batch_size, num_nodes, 2, 12, 4))
        # output = output.transpose(1, 2).transpose(0, 1).transpose(2, 3)  # 2, bs, 12, 10, 4
        return f_output, s_output

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}


class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout=0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)


class MixerBlock(nn.Module):

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout=0.):
        super().__init__()

        self.token_mix = nn.Sequential(
            nn.LayerNorm(dim),
            Rearrange('b n d -> b d n'),
            FeedForward(num_patch, token_dim, dropout),
            Rearrange('b d n -> b n d')
        )

        self.channel_mix = nn.Sequential(
            nn.LayerNorm(dim),
            FeedForward(dim, channel_dim, dropout),
        )

    def forward(self, x):
        # x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x


class MyMultiLayerPerceptron(nn.Module):
    def __init__(self, input_dim, hidden_dim):
        super(MyMultiLayerPerceptron, self).__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim, bias=True)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim, bias=True)
        self.layernorm = nn.LayerNorm(hidden_dim)
        self.act = nn.PReLU()
        self.drop = nn.Dropout(p=0.3)

    def forward(self, in_data):
        # in_data: [B, D, N]
        # in_data = self.layernorm(in_data)
        out = self.act(self.fc1(in_data))
        out = self.fc2(self.drop(out))
        return out + in_data


class STID_WITH_FUSION(nn.Module):
    def __init__(self, model_param):
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
        self.periods_emb = nn.Embedding(num_embeddings=4, embedding_dim=self.temp_dim_diw)
        # embedding layer
        self.time_sires_emb_layer = nn.Linear(in_features=self.input_dim * self.input_len, out_features=self.embed_dim,
                                              bias=True)
        # encoding
        # self.hidden_dim = self.embed_dim + self.node_dim * int(self.if_spatial) + \
        #                   self.temp_dim_tid * int(self.if_time_in_day) + \
        #                   self.temp_dim_diw * int(self.if_day_in_week)
        self.hidden_dim = self.embed_dim * (6 + 5 + 3)
        self.encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        # regression
        self.regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)
        self.fc_f = nn.Linear(self.output_len, 48)
        self.fc_s = nn.Linear(self.output_len, 48)

        self.flow_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)
        self.speed_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)
        self.occ_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)

        self.last_week_flow_time_series_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.last_week_speed_time_series_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.last_week_occ_time_series_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)

        self.last_last_week_flow_time_series_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.last_last_week_speed_time_series_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.last_last_week_occ_time_series_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)

        self.flow_conv = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
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

        self.flow_layer_norm = nn.LayerNorm(normalized_shape=4)
        self.speed_layer_norm = nn.LayerNorm(normalized_shape=4)
        self.occ_layer_norm = nn.LayerNorm(normalized_shape=4)

        self.time_in_day_fc = nn.Linear(in_features=self.input_len, out_features=10)
        self.day_in_week_fc = nn.Linear(in_features=self.input_len, out_features=10)

        self.last_week_time_in_day_fc = nn.Linear(in_features=12, out_features=10)
        self.last_week_day_in_week_fc = nn.Linear(in_features=12, out_features=10)

        self.last_last_week_time_in_day_fc = nn.Linear(in_features=12, out_features=10)
        self.last_last_week_day_in_week_fc = nn.Linear(in_features=12, out_features=10)

        self.last_time_in_day_emb = nn.Embedding(num_embeddings=288, embedding_dim=self.temp_dim_tid)
        self.last_day_in_week_emb = nn.Embedding(num_embeddings=7, embedding_dim=self.temp_dim_diw)

    def forward(self, his_data, last_week_y, last_last_week_y):

        """
        前3小时数据
        """
        time_in_day_emb = None
        input_data = his_data[:, :, :, :-3]
        flow_feature = input_data[:, :, :, 0:4]
        speed_feature = input_data[:, :, :, 4:8]
        Occ_feature = input_data[:, :, :, 8:]

        batch_size, _, num_nodes, _ = input_data.shape

        flow_feature = flow_feature.transpose(1, 2).contiguous()
        flow_feature = flow_feature.view(batch_size, num_nodes, -1)
        flow_time_series_emb = self.flow_time_series_emb(flow_feature)
        speed_feature = speed_feature.transpose(1, 2).contiguous()
        speed_feature = speed_feature.view(batch_size, num_nodes, -1)
        speed_time_series_emb = self.speed_time_series_emb(speed_feature)
        Occ_feature = Occ_feature.transpose(1, 2).contiguous()
        Occ_feature = Occ_feature.view(batch_size, num_nodes, -1)
        occ_time_series_emb = self.occ_time_series_emb(Occ_feature)

        """
        上周数据
        """
        last_week_input_data = last_week_y[:, :, :, :-3]
        last_week_flow_feature = last_week_input_data[:, :, :, 0:4]
        last_week_speed_feature = last_week_input_data[:, :, :, 4:8]
        last_week_Occ_feature = last_week_input_data[:, :, :, 8:]

        last_week_flow_feature = last_week_flow_feature.transpose(1, 2).contiguous()
        last_week_flow_feature = last_week_flow_feature.view(batch_size, num_nodes, -1)
        last_week_flow_feature_emb = self.last_week_flow_time_series_emb(last_week_flow_feature)
        last_week_speed_feature = last_week_speed_feature.transpose(1, 2).contiguous()
        last_week_speed_feature = last_week_speed_feature.view(batch_size, num_nodes, -1)
        last_week_speed_feature_emb = self.last_week_speed_time_series_emb(last_week_speed_feature)
        last_week_Occ_feature = last_week_Occ_feature.transpose(1, 2).contiguous()
        last_week_Occ_feature = last_week_Occ_feature.view(batch_size, num_nodes, -1)
        last_week_Occ_feature_emb = self.last_week_occ_time_series_emb(last_week_Occ_feature)

        """
        上上周数据
        """
        last_last_week_input_data = last_last_week_y[:, :, :, :-3]
        last_last_week_flow_feature = last_last_week_input_data[:, :, :, 0:4]
        last_last_week_speed_feature = last_last_week_input_data[:, :, :, 4:8]
        last_last_week_Occ_feature = last_last_week_input_data[:, :, :, 8:]

        last_last_week_flow_feature = last_last_week_flow_feature.transpose(1, 2).contiguous()
        last_last_week_flow_feature = last_last_week_flow_feature.view(batch_size, num_nodes, -1)
        last_last_week_flow_feature_emb = self.last_last_week_flow_time_series_emb(last_last_week_flow_feature)

        last_last_week_speed_feature = last_last_week_speed_feature.transpose(1, 2).contiguous()
        last_last_week_speed_feature = last_last_week_speed_feature.view(batch_size, num_nodes, -1)
        last_last_week_speed_feature_emb = self.last_last_week_speed_time_series_emb(last_last_week_speed_feature)

        last_last_week_Occ_feature = last_last_week_Occ_feature.transpose(1, 2).contiguous()
        last_last_week_Occ_feature = last_last_week_Occ_feature.view(batch_size, num_nodes, -1)
        last_last_week_Occ_feature_emb = self.last_last_week_occ_time_series_emb(last_last_week_Occ_feature)

        if self.if_time_in_day:
            t_i_d_data = his_data[:, :, 1, 12].long()
            last_week_time_data = last_week_y[:, :, 1, 12].long()
            time_in_day_emb = self.time_in_day_emb(t_i_d_data)
            last_week_time_in_day_emb = self.last_time_in_day_emb(last_week_time_data)
            time_in_day_emb = self.time_in_day_fc(time_in_day_emb.transpose(1, 2).contiguous()).transpose(1, 2)
            last_week_time_in_day_emb = self.last_week_time_in_day_fc(
                last_week_time_in_day_emb.transpose(1, 2).contiguous()).transpose(1, 2)

        day_in_week_emb = None
        if self.if_day_in_week:
            d_i_w_data = his_data[:, :, 1, 13].long()
            last_week_day_data = last_week_y[:, :, 1, 13].long()
            day_in_week_emb = self.day_in_week_emb(d_i_w_data)
            last_week_day_data_emb = self.last_day_in_week_emb(last_week_day_data)
            day_in_week_emb = self.day_in_week_fc(day_in_week_emb.transpose(1, 2).contiguous()).transpose(1, 2)
            last_week_day_in_week_emb = self.last_week_day_in_week_fc(
                last_week_day_data_emb.transpose(1, 2).contiguous()).transpose(1, 2)

        nodes_indx = torch.Tensor([list(range(num_nodes)) for _ in range(batch_size)]).long().type_as(d_i_w_data)
        node_emb = []
        if self.if_spatial:
            node_emb.append(self.node_emb(nodes_indx))

        tem_emb = []
        last_week_tem_emb = []
        if self.if_time_in_day:
            tem_emb.append(time_in_day_emb)
            last_week_tem_emb.append(last_week_time_in_day_emb)
        if self.if_day_in_week:
            tem_emb.append(day_in_week_emb)
            last_week_tem_emb.append(last_week_day_in_week_emb)
        hidden = torch.cat(
            [flow_time_series_emb] + [speed_time_series_emb] + [occ_time_series_emb] + node_emb + tem_emb, dim=2)
        last_week_hidden = torch.cat([last_week_flow_feature_emb] + [last_week_speed_feature_emb] + [
            last_week_Occ_feature_emb] + last_week_tem_emb, dim=2)
        last_last_week_hidden = torch.cat(
            [last_last_week_flow_feature_emb] + [last_last_week_speed_feature_emb] + [last_last_week_Occ_feature_emb],
            dim=2)

        hidden = torch.cat([hidden, last_week_hidden, last_last_week_hidden], dim=2)

        hidden = self.encoder(hidden)
        pred = self.regression_layer(hidden)

        f_output = self.fc_f(pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        s_output = self.fc_s(pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)

        # return pred.transpose(1, 2).contiguous()
        return f_output, s_output


class My_STID(nn.Module):
    def __init__(self, model_param):
        super(My_STID, self).__init__()
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

        self.flow_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)
        self.speed_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)
        self.occ_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)

        """
        分别对应早中晚
        """
        self.early_flow_time_series_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.early_speed_time_series_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.early_occ_time_series_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)

        self.mid_flow_time_series_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.mid_speed_time_series_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.mid_occ_time_series_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)

        self.later_flow_time_series_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.later_speed_time_series_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.later_occ_time_series_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)

        # 早中晚时间、星期提取
        self.early_time_in_day_emb = nn.Embedding(num_embeddings=288, embedding_dim=self.temp_dim_tid)
        self.mid_time_in_day_emb = nn.Embedding(num_embeddings=288, embedding_dim=self.temp_dim_tid)
        self.later_time_in_day_emb = nn.Embedding(num_embeddings=288, embedding_dim=self.temp_dim_tid)
        self.early_day_in_week_emb = nn.Embedding(num_embeddings=7, embedding_dim=self.temp_dim_diw)
        self.mid_day_in_week_emb = nn.Embedding(num_embeddings=7, embedding_dim=self.temp_dim_diw)
        self.later_day_in_week_emb = nn.Embedding(num_embeddings=7, embedding_dim=self.temp_dim_diw)

        self.early_time_in_day_fc = nn.Linear(in_features=12, out_features=10, bias=True)
        self.early_day_in_week_fc = nn.Linear(in_features=12, out_features=10, bias=True)
        self.mid_time_in_day_fc = nn.Linear(in_features=12, out_features=10, bias=True)
        self.mid_day_in_week_fc = nn.Linear(in_features=12, out_features=10, bias=True)
        self.later_time_in_day_fc = nn.Linear(in_features=12, out_features=10, bias=True)
        self.later_day_in_week_fc = nn.Linear(in_features=12, out_features=10, bias=True)

        # 多层感知机
        self.early_encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])
        self.mid_encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])
        self.later_encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)])
        # 回归
        self.early_regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)
        self.mid_regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)
        self.later_regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)
        # 线性层
        self.early_fc_f = nn.Linear(self.output_len, 48)
        self.early_fc_s = nn.Linear(self.output_len, 48)
        self.mid_fc_f = nn.Linear(self.output_len, 48)
        self.mid_fc_s = nn.Linear(self.output_len, 48)
        self.later_fc_f = nn.Linear(self.output_len, 48)
        self.later_fc_s = nn.Linear(self.output_len, 48)

        self.all_time_fc_f = nn.Linear(12, 4, bias=True)
        self.all_time_fc_s = nn.Linear(12, 4, bias=True)

        """
        加入time in day的时间特征、节点特征
        """
        self.time_in_day_fc = nn.Linear(in_features=36, out_features=10, bias=True)
        self.day_in_week_fc = nn.Linear(in_features=36, out_features=10, bias=True)
        self.node_fc = nn.Linear(in_features=40, out_features=10, bias=True)
        """
        加入可学习的系数和偏置
        """
        self.early_s_coefficient = nn.Parameter(torch.tensor(0.33))
        self.mid_s_coefficient = nn.Parameter(torch.tensor(0.33))
        self.later_s_coefficient = nn.Parameter(torch.tensor(0.33))
        self.early_f_coefficient = nn.Parameter(torch.tensor(0.33))
        self.mid_f_coefficient = nn.Parameter(torch.tensor(0.33))
        self.later_f_coefficient = nn.Parameter(torch.tensor(0.33))
        # bias = nn.Parameter(torch.tensor(0.5))

    def forward(self, his_data):
        his_data = his_data[:, :, :, :15]
        input_data = his_data[:, :, :, :-3]
        """
        提取三个指标
        """
        batch_size, _, num_nodes, _ = input_data.shape
        flow_feature = input_data[:, :, :, 0:4]
        speed_feature = input_data[:, :, :, 4:8]
        Occ_feature = input_data[:, :, :, 8:]

        early_flow_feature, mid_flow_feature, later_flow_feature = flow_feature[:, 0:12, :, :], flow_feature[:, 12:24,
                                                                                                :, :], flow_feature[:,
                                                                                                       24:, :, :]
        early_speed_feature, mid_speed_feature, later_speed_feature = speed_feature[:, 0:12, :, :], speed_feature[:,
                                                                                                    12:24, :,
                                                                                                    :], speed_feature[:,
                                                                                                        24:, :, :]
        early_occ_feature, mid_occ_feature, later_occ_feature = Occ_feature[:, 0:12, :, :], Occ_feature[:, 12:24, :,
                                                                                            :], Occ_feature[:, 24:, :,
                                                                                                :]

        """
        早期数据的融合 1.早 2.中 3.晚
        """
        # 1
        early_flow_feature = early_flow_feature.transpose(1, 2).contiguous()
        early_flow_feature = early_flow_feature.view(batch_size, num_nodes, -1)
        early_flow_time_series_emb = self.early_flow_time_series_emb(early_flow_feature)
        early_speed_feature = early_speed_feature.transpose(1, 2).contiguous()
        early_speed_feature = early_speed_feature.view(batch_size, num_nodes, -1)
        early_speed_time_series_emb = self.early_speed_time_series_emb(early_speed_feature)
        early_occ_feature = early_occ_feature.transpose(1, 2).contiguous()
        early_occ_feature = early_occ_feature.view(batch_size, num_nodes, -1)
        early_occ_time_series_emb = self.early_occ_time_series_emb(early_occ_feature)
        # 2
        mid_flow_feature = mid_flow_feature.transpose(1, 2).contiguous()
        mid_flow_feature = mid_flow_feature.view(batch_size, num_nodes, -1)
        mid_flow_time_series_emb = self.mid_flow_time_series_emb(mid_flow_feature)
        mid_speed_feature = mid_speed_feature.transpose(1, 2).contiguous()
        mid_speed_feature = mid_speed_feature.view(batch_size, num_nodes, -1)
        mid_speed_time_series_emb = self.mid_speed_time_series_emb(mid_speed_feature)
        mid_occ_feature = mid_occ_feature.transpose(1, 2).contiguous()
        mid_occ_feature = mid_occ_feature.view(batch_size, num_nodes, -1)
        mid_occ_time_series_emb = self.mid_occ_time_series_emb(mid_occ_feature)
        # 3
        later_flow_feature = later_flow_feature.transpose(1, 2).contiguous()
        later_flow_feature = later_flow_feature.view(batch_size, num_nodes, -1)
        later_flow_time_series_emb = self.later_flow_time_series_emb(later_flow_feature)
        later_speed_feature = later_speed_feature.transpose(1, 2).contiguous()
        later_speed_feature = later_speed_feature.view(batch_size, num_nodes, -1)
        later_speed_time_series_emb = self.later_speed_time_series_emb(later_speed_feature)
        later_occ_feature = later_occ_feature.transpose(1, 2).contiguous()
        later_occ_feature = later_occ_feature.view(batch_size, num_nodes, -1)
        later_occ_time_series_emb = self.later_occ_time_series_emb(later_occ_feature)

        """
        时间、星期提取
        """
        All_time = his_data[:, :, 1, -3].long()
        All_week = his_data[:, :, 1, -2].long()
        early_time, mid_time, later_time = All_time[:, :12], All_time[:, 12:24], All_time[:, 24:]
        early_week, mid_week, later_week = All_week[:, :12], All_week[:, 12:24], All_week[:, 24:]
        # 时间
        early_time_in_day_emb = self.early_time_in_day_emb(early_time)
        mid_time_in_day_emb = self.mid_time_in_day_emb(mid_time)
        later_time_in_day_emb = self.later_time_in_day_emb(later_time)
        early_time_in_day_emb = self.early_time_in_day_fc(early_time_in_day_emb.transpose(1, 2).contiguous()).transpose(
            1, 2)
        mid_time_in_day_emb = self.mid_time_in_day_fc(mid_time_in_day_emb.transpose(1, 2).contiguous()).transpose(1, 2)
        later_time_in_day_emb = self.later_time_in_day_fc(later_time_in_day_emb.transpose(1, 2).contiguous()).transpose(
            1, 2)
        # 星期
        early_day_in_week_emb = self.early_day_in_week_emb(early_week)
        mid_day_in_week_emb = self.mid_day_in_week_emb(mid_week)
        later_day_in_week_emb = self.later_day_in_week_emb(later_week)
        early_day_in_week_emb = self.early_day_in_week_fc(early_day_in_week_emb.transpose(1, 2).contiguous()).transpose(
            1, 2)
        mid_day_in_week_emb = self.mid_day_in_week_fc(mid_day_in_week_emb.transpose(1, 2).contiguous()).transpose(1, 2)
        later_day_in_week_emb = self.later_day_in_week_fc(later_day_in_week_emb.transpose(1, 2).contiguous()).transpose(
            1, 2)
        """
        地点提取
        """
        nodes_indx = torch.Tensor([list(range(40)) for _ in range(batch_size)]).long().type_as(early_week)
        node_emb = self.node_emb(nodes_indx)
        node_emb = self.node_fc(node_emb.transpose(1, 2).contiguous()).transpose(1, 2)

        """
        早中晚三种隐藏状态
        """
        early_hidden = torch.cat(
            [early_flow_time_series_emb] + [early_speed_time_series_emb] + [early_occ_time_series_emb] + [node_emb] + [
                early_time_in_day_emb] + [early_day_in_week_emb], dim=2)
        mid_hidden = torch.cat(
            [mid_flow_time_series_emb] + [mid_speed_time_series_emb] + [mid_occ_time_series_emb] + [node_emb] + [
                mid_time_in_day_emb] + [mid_day_in_week_emb], dim=2)
        later_hidden = torch.cat(
            [later_flow_time_series_emb] + [later_speed_time_series_emb] + [later_occ_time_series_emb] + [node_emb] + [
                later_time_in_day_emb] + [later_day_in_week_emb], dim=2)

        early_hidden = self.encoder(early_hidden)
        mid_hidden = self.encoder(mid_hidden)
        later_hidden = self.encoder(later_hidden)

        early_pred = self.early_regression_layer(early_hidden)
        mid_pred = self.mid_regression_layer(mid_hidden)
        later_pred = self.later_regression_layer(later_hidden)

        early_f_out = self.early_fc_f(early_pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        early_s_out = self.early_fc_s(early_pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        mid_f_out = self.mid_fc_f(mid_pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        mid_s_out = self.mid_fc_s(mid_pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        later_f_out = self.later_fc_f(later_pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        later_s_out = self.later_fc_s(later_pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)

        All_f_out = torch.cat([early_f_out] + [mid_f_out] + [later_f_out], dim=-1)
        All_s_out = torch.cat([early_s_out] + [mid_s_out] + [later_s_out], dim=-1)

        f_output = self.all_time_fc_f(All_f_out)
        s_output = self.all_time_fc_s(All_s_out)

        # f_output = self.all_time_fc_f(pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        # s_output = self.all_time_fc_s(pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)

        return f_output, s_output


class STID_with_hisdata(nn.Module):
    def __init__(self, model_param):
        super(STID_with_hisdata, self).__init__()
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
        self.discard_week = model_kwargs['discard_week']

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

        self.flow_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)
        self.speed_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)
        self.occ_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)

        """
        加入time in day的时间特征、节点特征
        """
        self.time_in_day_fc = nn.Linear(in_features=36, out_features=10, bias=True)
        self.day_in_week_fc = nn.Linear(in_features=36, out_features=10, bias=True)
        self.node_fc = nn.Linear(in_features=40, out_features=10, bias=True)

        self.his_week_extract = His_week_extract(discard_week=self.discard_week, num_layer=self.num_layer)
        self.all_time_fc_f = nn.Linear(8, 4, bias=True)
        self.all_time_fc_s = nn.Linear(8, 4, bias=True)

    def forward(self, data, his_week_y):
        input_data = data[:, :, :, :15]
        feature_data = input_data[:, :, :, :-3]
        batch_size, _, num_nodes, _ = input_data.shape
        """
        x特征的提取
        1.分开提取
        2.各自emb
        3.时间、星期和node的提取
        4.mlp感知机 + 回归
        """
        # 1
        flow_feature = feature_data[:, :, :, 0:4]
        speed_feature = feature_data[:, :, :, 4:8]
        Occ_feature = feature_data[:, :, :, 8:]
        # 2
        flow_feature = flow_feature.transpose(1, 2).contiguous()
        flow_feature = flow_feature.view(batch_size, num_nodes, -1)
        flow_time_series_emb = self.flow_time_series_emb(flow_feature)
        speed_feature = speed_feature.transpose(1, 2).contiguous()
        speed_feature = speed_feature.view(batch_size, num_nodes, -1)
        speed_time_series_emb = self.speed_time_series_emb(speed_feature)
        Occ_feature = Occ_feature.transpose(1, 2).contiguous()
        Occ_feature = Occ_feature.view(batch_size, num_nodes, -1)
        occ_time_series_emb = self.occ_time_series_emb(Occ_feature)
        # 3
        t_i_d_data = input_data[:, :, 1, -3].long()
        time_in_day_emb = self.time_in_day_emb(t_i_d_data)
        time_in_day_emb = self.time_in_day_fc(time_in_day_emb.transpose(1, 2).contiguous()).transpose(1, 2)
        d_i_w_data = input_data[:, :, 1, -2].long()
        day_in_week_emb = self.day_in_week_emb(d_i_w_data)
        day_in_week_emb = self.day_in_week_fc(day_in_week_emb.transpose(1, 2).contiguous()).transpose(1, 2)
        nodes_indx = torch.Tensor([list(range(40)) for _ in range(batch_size)]).long().type_as(d_i_w_data)
        node_emb = self.node_emb(nodes_indx)
        node_emb = self.node_fc(node_emb.transpose(1, 2).contiguous()).transpose(1, 2)
        hidden = torch.cat([flow_time_series_emb] + [speed_time_series_emb] + [occ_time_series_emb] + [node_emb] + [
            time_in_day_emb] + [day_in_week_emb], dim=2)
        # 4
        hidden = self.encoder(hidden)
        pred = self.regression_layer(hidden)
        f_output = self.fc_f(pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        s_output = self.fc_s(pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        """
        历史y特征提取
        """
        his_flow, his_speed = self.his_week_extract(his_week_y)

        pre_flow = torch.cat([f_output, his_flow], dim=-1)
        pre_speed = torch.cat([his_speed, his_speed], dim=-1)

        pre_flow = self.all_time_fc_f(pre_flow)
        pre_speed = self.all_time_fc_s(pre_speed)

        return pre_flow, pre_speed


class His_week_extract(nn.Module):
    def __init__(self, discard_week, num_layer):
        super(His_week_extract, self).__init__()
        self.discard_week = discard_week
        self.num_layer = num_layer
        self.embed_dim = 32
        self.output_len = 12
        self.flow_time_series_emb = nn.Linear(in_features=self.discard_week * 12 * 4, out_features=self.embed_dim,
                                              bias=True)
        self.speed_time_series_emb = nn.Linear(in_features=self.discard_week * 12 * 4, out_features=self.embed_dim,
                                               bias=True)
        self.occ_time_series_emb = nn.Linear(in_features=self.discard_week * 12 * 4, out_features=self.embed_dim,
                                             bias=True)
        self.time_in_day_emb = nn.Embedding(num_embeddings=288, embedding_dim=self.embed_dim)
        self.day_in_week_emb = nn.Embedding(num_embeddings=7, embedding_dim=self.embed_dim)
        """
        加入time in day的时间特征、节点特征
        """
        self.time_in_day_fc = nn.Linear(in_features=self.discard_week * 12, out_features=10, bias=True)
        self.day_in_week_fc = nn.Linear(in_features=self.discard_week * 12, out_features=10, bias=True)

        self.node_emb = nn.Embedding(num_embeddings=40, embedding_dim=self.embed_dim)
        self.node_fc = nn.Linear(in_features=40, out_features=10, bias=True)
        """
        mlp和回归
        """
        self.hidden_dim = self.embed_dim * 6
        self.encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        # regression
        self.regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)
        self.fc_f = nn.Linear(self.output_len, 48)
        self.fc_s = nn.Linear(self.output_len, 48)

    def forward(self, his_week_y):
        his_week_y = torch.cat(his_week_y, dim=1)
        his_week_y = his_week_y[:, :, :, :15]
        feature_data = his_week_y[:, :, :, :-3]
        batch_size, time_len, num_nodes, _ = his_week_y.shape
        """
        x特征的提取
        1.分开提取
        2.各自emb
        3.时间、星期和node的提取
        4.mlp感知机 + 回归
                """
        # 1
        flow_feature = feature_data[:, :, :, 0:4]
        speed_feature = feature_data[:, :, :, 4:8]
        Occ_feature = feature_data[:, :, :, 8:]
        # 2
        flow_feature = flow_feature.transpose(1, 2).contiguous()
        flow_feature = flow_feature.view(batch_size, num_nodes, -1)
        flow_time_series_emb = self.flow_time_series_emb(flow_feature)
        speed_feature = speed_feature.transpose(1, 2).contiguous()
        speed_feature = speed_feature.view(batch_size, num_nodes, -1)
        speed_time_series_emb = self.speed_time_series_emb(speed_feature)
        Occ_feature = Occ_feature.transpose(1, 2).contiguous()
        Occ_feature = Occ_feature.view(batch_size, num_nodes, -1)
        occ_time_series_emb = self.occ_time_series_emb(Occ_feature)
        # 3
        t_i_d_data = his_week_y[:, :, 1, -3].long()
        time_in_day_emb = self.time_in_day_emb(t_i_d_data)
        time_in_day_emb = self.time_in_day_fc(time_in_day_emb.transpose(1, 2).contiguous()).transpose(1, 2)
        d_i_w_data = his_week_y[:, :, 1, -2].long()
        day_in_week_emb = self.day_in_week_emb(d_i_w_data)
        day_in_week_emb = self.day_in_week_fc(day_in_week_emb.transpose(1, 2).contiguous()).transpose(1, 2)
        nodes_indx = torch.Tensor([list(range(40)) for _ in range(batch_size)]).long().type_as(d_i_w_data)
        node_emb = self.node_emb(nodes_indx)
        node_emb = self.node_fc(node_emb.transpose(1, 2).contiguous()).transpose(1, 2)
        hidden = torch.cat([flow_time_series_emb] + [speed_time_series_emb] + [occ_time_series_emb] + [node_emb] + [
            time_in_day_emb] + [day_in_week_emb], dim=2)
        # 4
        hidden = self.encoder(hidden)
        pred = self.regression_layer(hidden)
        f_output = self.fc_f(pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        s_output = self.fc_s(pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)

        return f_output, s_output


class STID_NODE_40(nn.Module):
    def __init__(self, model_param):
        super(STID_NODE_40, self).__init__()
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
        self.temp_dim_pid = model_kwargs['temp_dim_pid']

        self.if_time_in_day = model_kwargs['if_T_i_D']
        self.if_day_in_week = model_kwargs['if_D_i_W']
        self.if_period_in_day = model_kwargs['if_P_i_D']
        self.if_spatial = model_kwargs['if_node']

        # # spatial embeddings (nn.init.xavier_uniform_(self.node_emb))
        self.node_emb = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.node_dim)
        # temporal embeddings
        self.time_in_day_emb = nn.Embedding(num_embeddings=288, embedding_dim=self.temp_dim_tid)
        self.day_in_week_emb = nn.Embedding(num_embeddings=7, embedding_dim=self.temp_dim_diw)
        self.period_in_day_emb = nn.Embedding(num_embeddings=4, embedding_dim=self.temp_dim_pid)
        # self.time_in_day_fc = nn.Linear(self.input_len, 40)
        # self.day_in_week_fc = nn.Linear(self.input_len, 40)
        self.time_in_day_fc = nn.Linear(self.input_len * self.temp_dim_tid, self.temp_dim_tid)
        self.day_in_week_fc = nn.Linear(self.input_len * self.temp_dim_diw, self.temp_dim_diw)
        self.period_in_day_fc = nn.Linear(self.input_len, 40)

        self.hidden_dim = self.embed_dim * 3 + self.temp_dim_tid + self.temp_dim_diw + self.temp_dim_pid + self.node_dim
        self.encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        # regression
        self.fc_f = nn.Linear(self.hidden_dim, 12)
        self.fc_s = nn.Linear(self.hidden_dim, 12)

        self.flow_time_series_emb = nn.Linear(in_features=self.input_dim * 3, out_features=self.embed_dim, bias=True)
        self.speed_time_series_emb = nn.Linear(in_features=self.input_dim * 3, out_features=self.embed_dim, bias=True)
        self.occ_time_series_emb = nn.Linear(in_features=self.input_dim * 3, out_features=self.embed_dim, bias=True)

    def forward(self, his_data):
        # prepare
        input_data = his_data[:, :, :, :48]
        batch_size, seq_len, num_nodes, features = input_data.shape
        input_data = input_data.reshape(batch_size, seq_len, num_nodes * 4, features // 4)

        flow_feature = input_data[:, :, :, 0:4]
        speed_feature = input_data[:, :, :, 4:8]
        Occ_feature = input_data[:, :, :, 8:12]

        flow_feature = flow_feature.transpose(1, 2).contiguous()
        flow_feature = flow_feature.view(batch_size, num_nodes * 4, -1)
        flow_time_series_emb = self.flow_time_series_emb(flow_feature)

        speed_feature = speed_feature.transpose(1, 2).contiguous()
        speed_feature = speed_feature.view(batch_size, num_nodes * 4, -1)
        speed_time_series_emb = self.speed_time_series_emb(speed_feature)

        Occ_feature = Occ_feature.transpose(1, 2).contiguous()
        Occ_feature = Occ_feature.view(batch_size, num_nodes * 4, -1)
        occ_time_series_emb = self.occ_time_series_emb(Occ_feature)

        if self.if_time_in_day:
            t_i_d_data = his_data[:, :, 0, -2].long()
            # (b, node_nums(time_in_day dim-len)) -> T^{TiD}(Nd x D)-> (b, node_nums(time_in_day dim-len), emb)
            time_in_day_emb = self.time_in_day_emb(t_i_d_data)
            time_in_day_emb = time_in_day_emb.unsqueeze(1)
            time_in_day_emb = self.time_in_day_fc(time_in_day_emb.reshape(batch_size, 1, -1)).repeat(1, 40, 1)
            # time_in_day_emb = self.time_in_day_fc(time_in_day_emb.transpose(1, 2).contiguous()).transpose(1, 2)

        if self.if_day_in_week:
            d_i_w_data = his_data[:, :, 0, -1].long()
            # (b, node_nums(day_in_week dim-len)) -> T^{DiW}(Nw x D)-> (b, node_nums(day_in_week dim-len), emb)
            day_in_week_emb = self.day_in_week_emb(d_i_w_data)
            day_in_week_emb = day_in_week_emb.unsqueeze(1)
            day_in_week_emb = self.time_in_day_fc(day_in_week_emb.reshape(batch_size, 1, -1)).repeat(1, 40, 1)
            # day_in_week_emb = self.day_in_week_fc(day_in_week_emb.transpose(1, 2).contiguous()).transpose(1, 2)

        if self.if_period_in_day:
            p_i_d_data = his_data[:, :, 0, -2].long()
            # (b, node_nums(period_in_day dim-len)) -> T^{PiD}(Np x D)-> (b, node_nums(period_in_day dim-len), emb)
            period_in_day_emb = self.period_in_day_emb(p_i_d_data)
            period_in_day_emb = self.period_in_day_fc(period_in_day_emb.transpose(1, 2).contiguous()).transpose(1, 2)
        # ts embedding

        # (b, L, num_nodes, channel) -> (b, num_nodes, L, channel) ->
        # (b, num_nodes, L, channel) -> (b, num_nodes, L * channel) (data || time-in-day || day-of-week)
        # input_data = input_data.transpose(1, 2).contiguous()
        # input_data = input_data.view(batch_size, num_nodes, -1)
        # (b, num_nodes, L * channel) -> FC_emb (L*channel, D) -> (b, num_nodes, emb)
        # time_series_emb = self.time_sires_emb_layer(input_data)

        # node emb
        # (b, node_nums)
        nodes_indx = torch.Tensor([list(range(num_nodes * 4)) for _ in range(batch_size)]).long().type_as(d_i_w_data)
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
        if self.if_period_in_day:
            tem_emb.append(period_in_day_emb)

        hidden = torch.cat(
            [flow_time_series_emb] + [speed_time_series_emb] + [occ_time_series_emb] + node_emb + tem_emb, dim=2)

        # concat (b, num_node, 32*4)
        # hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=2)
        # (b, num_node, 32*4) -> (b, num_node, 32*4)
        hidden = self.encoder(hidden)
        # (b, num_node, 32*4) -> (b, num_node, out_len)
        # pred = self.regression_layer(hidden)
        # print('pred - pred.shape:', pred.shape)

        f_output = self.fc_f(hidden).transpose(1, 2).reshape((batch_size, 12, 10, 4))
        s_output = self.fc_s(hidden).transpose(1, 2).reshape((batch_size, 12, 10, 4))

        return f_output, s_output

class My_model(nn.Module):
    def __init__(self, model_param):
        super(My_model, self).__init__()
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
        self.temp_dim_pid = model_kwargs['temp_dim_pid']

        self.if_time_in_day = model_kwargs['if_T_i_D']
        self.if_day_in_week = model_kwargs['if_D_i_W']
        self.if_period_in_day = model_kwargs['if_P_i_D']
        self.if_spatial = model_kwargs['if_node']
        self.ablation_type = model_kwargs['ablation_type']

        # # spatial embeddings (nn.init.xavier_uniform_(self.node_emb))
        self.node_emb = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.node_dim)
        # temporal embeddings
        self.time_in_day_emb = nn.Embedding(num_embeddings=288, embedding_dim=self.temp_dim_tid)
        self.day_in_week_emb = nn.Embedding(num_embeddings=7, embedding_dim=self.temp_dim_diw)
        self.period_in_day_emb = nn.Embedding(num_embeddings=4, embedding_dim=self.temp_dim_pid)
        # self.time_in_day_fc = nn.Linear(self.input_len, 40)
        # self.day_in_week_fc = nn.Linear(self.input_len, 40)
        self.time_in_day_fc = nn.Linear(self.input_len * self.temp_dim_tid, self.temp_dim_tid)
        self.day_in_week_fc = nn.Linear(self.input_len * self.temp_dim_diw, self.temp_dim_diw)
        self.period_in_day_fc = nn.Linear(self.input_len, 40)
        if self.ablation_type == 'All':# All tf_inf tf_inf_node tf_cluster
            self.hidden_dim = self.embed_dim * 3 + self.temp_dim_tid + self.temp_dim_diw + self.temp_dim_pid + self.node_dim
        elif self.ablation_type == 'tf_inf':
            self.hidden_dim = self.embed_dim * 3
        elif self.ablation_type == 'tf_inf_node':
            self.hidden_dim = self.embed_dim * 3 + self.node_dim
        elif self.ablation_type == 'tf_cluster':
            self.hidden_dim = self.embed_dim * 3 + self.temp_dim_tid + self.temp_dim_diw + self.temp_dim_pid
        self.encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        # regression
        self.fc_f = nn.Linear(self.hidden_dim, 12)
        self.fc_s = nn.Linear(self.hidden_dim, 12)

        self.flow_time_series_emb = nn.Linear(in_features=self.input_dim * 3, out_features=self.embed_dim, bias=True)
        self.speed_time_series_emb = nn.Linear(in_features=self.input_dim * 3, out_features=self.embed_dim, bias=True)
        self.occ_time_series_emb = nn.Linear(in_features=self.input_dim * 3, out_features=self.embed_dim, bias=True)

    def forward(self, his_data):
        # prepare
        input_data = his_data[:, :, :, :48]
        batch_size, seq_len, num_nodes, features = input_data.shape
        input_data = input_data.reshape(batch_size, seq_len, num_nodes * 4, features // 4)

        flow_feature = input_data[:, :, :, 0:4]
        speed_feature = input_data[:, :, :, 4:8]
        Occ_feature = input_data[:, :, :, 8:12]

        flow_feature = flow_feature.transpose(1, 2).contiguous()
        flow_feature = flow_feature.view(batch_size, num_nodes * 4, -1)
        flow_time_series_emb = self.flow_time_series_emb(flow_feature)

        speed_feature = speed_feature.transpose(1, 2).contiguous()
        speed_feature = speed_feature.view(batch_size, num_nodes * 4, -1)
        speed_time_series_emb = self.speed_time_series_emb(speed_feature)

        Occ_feature = Occ_feature.transpose(1, 2).contiguous()
        Occ_feature = Occ_feature.view(batch_size, num_nodes * 4, -1)
        occ_time_series_emb = self.occ_time_series_emb(Occ_feature)

        if self.if_time_in_day:
            t_i_d_data = his_data[:, :, 0, -2].long()
            # (b, node_nums(time_in_day dim-len)) -> T^{TiD}(Nd x D)-> (b, node_nums(time_in_day dim-len), emb)
            time_in_day_emb = self.time_in_day_emb(t_i_d_data)
            time_in_day_emb = time_in_day_emb.unsqueeze(1)
            time_in_day_emb = self.time_in_day_fc(time_in_day_emb.reshape(batch_size, 1, -1)).repeat(1, 40, 1)
            # time_in_day_emb = self.time_in_day_fc(time_in_day_emb.transpose(1, 2).contiguous()).transpose(1, 2)

        if self.if_day_in_week:
            d_i_w_data = his_data[:, :, 0, -1].long()
            # (b, node_nums(day_in_week dim-len)) -> T^{DiW}(Nw x D)-> (b, node_nums(day_in_week dim-len), emb)
            day_in_week_emb = self.day_in_week_emb(d_i_w_data)
            day_in_week_emb = day_in_week_emb.unsqueeze(1)
            day_in_week_emb = self.time_in_day_fc(day_in_week_emb.reshape(batch_size, 1, -1)).repeat(1, 40, 1)
            # day_in_week_emb = self.day_in_week_fc(day_in_week_emb.transpose(1, 2).contiguous()).transpose(1, 2)

        if self.if_period_in_day:
            p_i_d_data = his_data[:, :, 0, -2].long()
            # (b, node_nums(period_in_day dim-len)) -> T^{PiD}(Np x D)-> (b, node_nums(period_in_day dim-len), emb)
            period_in_day_emb = self.period_in_day_emb(p_i_d_data)
            period_in_day_emb = self.period_in_day_fc(period_in_day_emb.transpose(1, 2).contiguous()).transpose(1, 2)

        nodes_indx = torch.Tensor([list(range(num_nodes * 4)) for _ in range(batch_size)]).long().type_as(d_i_w_data)
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
        if self.if_period_in_day:
            tem_emb.append(period_in_day_emb)
        # All tf_inf tf_inf_node tf_cluster
        if self.ablation_type == 'All':
            hidden = torch.cat([flow_time_series_emb] + [speed_time_series_emb] + [occ_time_series_emb] + node_emb + tem_emb, dim=2)
        elif self.ablation_type == 'tf_inf':
            hidden = torch.cat([flow_time_series_emb] + [speed_time_series_emb] + [occ_time_series_emb], dim=2)
        elif self.ablation_type == 'tf_inf_node':
            hidden = torch.cat([flow_time_series_emb] + [speed_time_series_emb] + [occ_time_series_emb] + node_emb, dim=2)
        elif self.ablation_type == 'tf_cluster':
            hidden = torch.cat([flow_time_series_emb] + [speed_time_series_emb] + [occ_time_series_emb] + tem_emb, dim=2)

        # concat (b, num_node, 32*4)
        # hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=2)
        # (b, num_node, 32*4) -> (b, num_node, 32*4)
        hidden = self.encoder(hidden)
        # (b, num_node, 32*4) -> (b, num_node, out_len)
        # pred = self.regression_layer(hidden)
        # print('pred - pred.shape:', pred.shape)

        f_output = self.fc_f(hidden).transpose(1, 2).reshape((batch_size, 12, 10, 4))
        s_output = self.fc_s(hidden).transpose(1, 2).reshape((batch_size, 12, 10, 4))

        return f_output, s_output


if __name__ == '__main__':
    model_param = {
        "num_nodes": 325,
        'input_len': 36,
        'input_dim': 48,
        'embed_dim': 48,
        'output_len': 64,
        'num_layer': 4,
        "if_node": True,
        'node_dim': 32,
        "if_T_i_D": True,
        "if_D_i_W": True,
        'temp_dim_tid': 32,
        'temp_dim_diw': 32,
        'discard_week': 6,
        'temp_dim_pid': 0,
        'congest_dim': 0,
        "if_P_i_D": False,
        "if_C_T_N": False,
        'if_cluster': False,
        'cluster_dim': 0,
        'ablation_type': 'tf_cluster' #All tf_inf tf_inf_node tf_cluster
    }
    import sys

    sys.path.append("..")
    from Data_Load_TGCN import *
    # from score import *
    from torch.utils.data import DataLoader

    # data_inf = np.load('val_2_8.npy')
    data_inf = np.load('val_cluster.npy')
    # train_loader = DataLoader(Dataset_with_his(data_inf, test_flag=False,discard_week=6), batch_size=512, shuffle=True, num_workers=0,pin_memory=True)
    train_loader = DataLoader(MyDataSet(data_inf, shift=True), batch_size=512, shuffle=True, num_workers=0,
                              pin_memory=True)
    # val_loader = DataLoader(MyDataSet(val_data, shift=True), batch_size=512, shuffle=True, num_workers=0,pin_memory=True)
    # adj = np.load('adj_mat.npy')
    model = My_model(model_param).cuda()
    for index, (x_sequence, y_flow, y_speed) in enumerate(train_loader):
        x_sequence, y_flow, y_speed = x_sequence.cuda(), y_flow.cuda(), y_speed.cuda()
        x_sequence = x_sequence.float()
        f_output, s_output = model(x_sequence)
        score = get_score(f_output, s_output, y_flow, y_speed, mean=None, std=None)
        pass
    # for index,(x_sequence, y_flow,y_speed,his_week_y) in enumerate(train_loader):
    #     his_week_y = [week_inf.cuda() for week_inf in his_week_y]
    #     out_put = model(x_sequence.cuda(),his_week_y)
    #     pass
