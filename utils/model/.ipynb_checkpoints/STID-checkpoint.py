import torch
import torch.nn as nn
import numpy as np
from GCN import *
from einops.layers.torch import Rearrange

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
            nn.Linear(16, self._hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self._hidden_dim//2, self._hidden_dim),
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
            nn.Linear(16, self._hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self._hidden_dim//2, self._hidden_dim),
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

        #1 日期、星期 词嵌入
        time_feature = inputs[...,-4:-3].long()
        week_feature = inputs[...,-1:].long()
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
    def __init__(self, dim, hidden_dim, dropout = 0.):
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

    def __init__(self, dim, num_patch, token_dim, channel_dim, dropout = 0.):
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

        x = x + self.token_mix(x)

        x = x + self.channel_mix(x)

        return x

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


class MYSTID(nn.Module):
    def __init__(self,model_param):
        super(MYSTID, self).__init__()
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
        self.hidden_dim = self.embed_dim + self.node_dim * int(self.if_spatial) + \
                          self.temp_dim_tid * int(self.if_time_in_day) + \
                          self.temp_dim_diw * int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        # regression
        self.regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)
        self.fc_f = nn.Linear(self.output_len, 48)
        self.fc_s = nn.Linear(self.output_len, 48)

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
        batch_size, _, num_nodes, _ = input_data.shape
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

        # concat (b, num_node, 32*4)
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=2)
        # (b, num_node, 32*4) -> (b, num_node, 32*4)
        hidden = self.encoder(hidden)
        # (b, num_node, 32*4) -> (b, num_node, out_len)
        pred = self.regression_layer(hidden)
        # print('pred - pred.shape:', pred.shape)

        f_output = self.fc_f(pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        s_output = self.fc_s(pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)

        # return pred.transpose(1, 2).contiguous()
        return f_output,s_output

class STID_speed(nn.Module):
    def __init__(self,model_param):
        super(STID_speed, self).__init__()
        model_kwargs = model_param
        # attributes
        self.num_nodes = model_kwargs['num_nodes']
        self.node_dim = model_kwargs['node_dim']
        self.input_len = model_kwargs['input_len']
        self.input_dim = model_kwargs['input_dim']
        self.embed_dim = model_kwargs['embed_dim']
        self.output_len = model_kwargs['output_len']
        model_kwargs['num_layer'] = 2
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
        self.hidden_dim = self.embed_dim + self.node_dim * int(self.if_spatial) + \
                          self.temp_dim_tid * int(self.if_time_in_day) + \
                          self.temp_dim_diw * int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        # regression
        self.regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)
        self.fc_f = nn.Linear(self.output_len, 48)
        self.fc_s = nn.Linear(self.output_len, 48)

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
        batch_size, _, num_nodes, _ = input_data.shape
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

        # concat (b, num_node, 32*4)
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=2)
        # (b, num_node, 32*4) -> (b, num_node, 32*4)
        hidden = self.encoder(hidden)
        # (b, num_node, 32*4) -> (b, num_node, out_len)
        pred = self.regression_layer(hidden)
        # print('pred - pred.shape:', pred.shape)

        s_output = self.fc_s(pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)

        # return pred.transpose(1, 2).contiguous()
        return s_output

class STID_flow(nn.Module):
    def __init__(self,model_param):
        super(STID_flow, self).__init__()
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
        self.hidden_dim = self.embed_dim + self.node_dim * int(self.if_spatial) + \
                          self.temp_dim_tid * int(self.if_time_in_day) + \
                          self.temp_dim_diw * int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        # regression
        self.regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)
        self.fc_f = nn.Linear(self.output_len, 48)
        self.fc_s = nn.Linear(self.output_len, 48)

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
        batch_size, _, num_nodes, _ = input_data.shape
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

        # concat (b, num_node, 32*4)
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=2)
        # (b, num_node, 32*4) -> (b, num_node, 32*4)
        hidden = self.encoder(hidden)
        # (b, num_node, 32*4) -> (b, num_node, out_len)
        pred = self.regression_layer(hidden)
        # print('pred - pred.shape:', pred.shape)

        f_output = self.fc_f(pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)

        # return pred.transpose(1, 2).contiguous()
        return f_output

class STID_node_flow(nn.Module):
    def __init__(self,model_param):
        super(STID_node_flow, self).__init__()
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
        self.hidden_dim = self.embed_dim + self.node_dim * int(self.if_spatial) + \
                          self.temp_dim_tid * int(self.if_time_in_day) + \
                          self.temp_dim_diw * int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        # regression
        self.regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)
        # self.fc_f = nn.Linear(self.output_len, 48)
        # self.fc_s = nn.Linear(self.output_len, 48)

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
        batch_size, _, num_nodes, _ = input_data.shape
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

        # concat (b, num_node, 32*4)
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=2)
        # (b, num_node, 32*4) -> (b, num_node, 32*4)
        hidden = self.encoder(hidden)
        # (b, num_node, 32*4) -> (b, num_node, out_len)
        pred = self.regression_layer(hidden)
        # print('pred - pred.shape:', pred.shape)

        f_output = pred.reshape((batch_size, num_nodes, 12, 1)).transpose(1, 2)

        # return pred.transpose(1, 2).contiguous()
        return f_output

class STID_node_speed(nn.Module):
    def __init__(self,model_param):
        super(STID_node_speed, self).__init__()
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
        self.hidden_dim = self.embed_dim + self.node_dim * int(self.if_spatial) + \
                          self.temp_dim_tid * int(self.if_time_in_day) + \
                          self.temp_dim_diw * int(self.if_day_in_week)
        self.encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        # regression
        self.regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)
        # self.fc_f = nn.Linear(self.output_len, 48)
        # self.fc_s = nn.Linear(self.output_len, 48)

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
        batch_size, _, num_nodes, _ = input_data.shape
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

        # concat (b, num_node, 32*4)
        hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=2)
        # (b, num_node, 32*4) -> (b, num_node, 32*4)
        hidden = self.encoder(hidden)
        # (b, num_node, 32*4) -> (b, num_node, out_len)
        pred = self.regression_layer(hidden)
        # print('pred - pred.shape:', pred.shape)

        f_output = pred.reshape((batch_size, num_nodes, 12, 1)).transpose(1, 2)

        # return pred.transpose(1, 2).contiguous()
        return f_output

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
        self.sum_regression_layer = nn.Linear(self.hidden_dim*2, self.output_len, bias=True)
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

    def forward(self, his_data,aux_data):
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

        aux_flow_feature = input_data[:, :, :, 0:4]
        aux_speed_feature = input_data[:, :, :, 4:8]
        aux_Occ_feature = input_data[:, :, :, 8:]

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

        aux_flow_feature = aux_flow_feature.transpose(1, 2).contiguous()
        aux_flow_feature = aux_flow_feature.view(batch_size, num_nodes, -1)
        aux_flow_time_series_emb = self.flow_time_series_emb(aux_flow_feature)

        aux_speed_feature = aux_speed_feature.transpose(1, 2).contiguous()
        aux_speed_feature = aux_speed_feature.view(batch_size, num_nodes, -1)
        aux_speed_time_series_emb = self.speed_time_series_emb(aux_speed_feature)

        aux_Occ_feature = aux_Occ_feature.transpose(1, 2).contiguous()
        aux_Occ_feature = aux_Occ_feature.view(batch_size, num_nodes, -1)
        aux_occ_time_series_emb = self.occ_time_series_emb(aux_Occ_feature)




        if self.if_time_in_day:
            t_i_d_data = his_data[:, 1, :, -2].long()
            aux_t_i_d_data = aux_data[:, 1, :, -2].long()
            # (b, node_nums(time_in_day dim-len)) -> T^{TiD}(Nd x D)-> (b, node_nums(time_in_day dim-len), emb)
            time_in_day_emb = self.time_in_day_emb(t_i_d_data)
            aux_time_in_day_emb = self.time_in_day_emb(aux_t_i_d_data)

        day_in_week_emb = None
        if self.if_day_in_week:
            d_i_w_data = his_data[:, 1, :, -1].long()
            aux_d_i_w_data = aux_data[:, 1, :, -1].long()
            # (b, node_nums(day_in_week dim-len)) -> T^{DiW}(Nw x D)-> (b, node_nums(day_in_week dim-len), emb)
            day_in_week_emb = self.day_in_week_emb(d_i_w_data)
            aux_day_in_week_emb = self.day_in_week_emb(aux_d_i_w_data)

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
        aux_tem_emb = []
        if self.if_time_in_day:
            tem_emb.append(time_in_day_emb)
            aux_tem_emb.append(aux_time_in_day_emb)
        if self.if_day_in_week:
            tem_emb.append(day_in_week_emb)
            aux_tem_emb.append(aux_day_in_week_emb)

        hidden = torch.cat([flow_time_series_emb]+[speed_time_series_emb]+[occ_time_series_emb]+node_emb+tem_emb,dim=2)
        aux_hidden = torch.cat([aux_flow_time_series_emb] + [aux_speed_time_series_emb] + [aux_occ_time_series_emb] + node_emb + aux_tem_emb, dim=2)

        # concat (b, num_node, 32*4)
        # hidden = torch.cat([time_series_emb] + node_emb + tem_emb, dim=2)
        # (b, num_node, 32*4) -> (b, num_node, 32*4)
        hidden = self.encoder(hidden)
        aux_hidden = self.encoder(aux_hidden)

        sum_hidden = torch.cat([hidden,aux_hidden],dim=2)

        # (b, num_node, 32*4) -> (b, num_node, out_len)
        # pred = self.regression_layer(hidden)
        pred = self.sum_regression_layer(sum_hidden)
        # print('pred - pred.shape:', pred.shape)

        f_output = self.fc_f(pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        s_output = self.fc_s(pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)

        # return pred.transpose(1, 2).contiguous()
        return f_output,s_output

class STID_WITH_GCN(nn.Module):
    def __init__(self,model_param,adj):
        super(STID_WITH_GCN, self).__init__()
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
        self.register_buffer("adj", torch.FloatTensor(adj))
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
        self.hidden_dim = self.embed_dim * 7
        self.encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        # regression
        self.regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)
        self.fc_f = nn.Linear(self.output_len, 48)
        self.fc_s = nn.Linear(self.output_len, 48)

        self.early_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)
        self.mid_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)
        self.last_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)


        self.flow_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim,bias=True)
        self.speed_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)
        self.occ_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)
        self.week_time_emd = nn.Linear(in_features=64, out_features=self.embed_dim, bias=True)

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

        self.speed_gcn = Speed_GCN(in_features=4,hidden_features=64,out_features=4,time_step=36)
        self.flow_gcn = flow_GCN(in_features=4, hidden_features=64, out_features=4, time_step=36)
        self.all_gcn = All_GCN(in_features=12, hidden_features=64, out_features=12, time_step=36)

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

        early_feature = input_data[:,0:12,:,:]
        mid_feature = input_data[:, 12:24, :, :]
        later_feature = input_data[:, 24:, :, :]
        """
        2.第二次尝试，对整体先提特征
        input_data = self.all_feature_conv(input_data)
        """
        """
        第三次尝试，改变loss权重
        """
        # input_data = self.all_gcn(input_data, self.adj)

        flow_feature = input_data[:,:,:,0:4]
        speed_feature = input_data[:, :, :, 4:8]
        Occ_feature = input_data[:, :, :, 8:]

        batch_size, _, num_nodes, _ = input_data.shape

        """
        第四次尝试，加入flow和speed的gcn
        speed_gcn = self.speed_gcn(speed_feature,self.adj)
        flow_gcn = self.flow_gcn(flow_feature, self.adj)
        speed_feature = speed_gcn
        flow_feature = flow_gcn
        """

        # early_feature = early_feature.transpose(1, 2).contiguous()
        # early_feature = early_feature.view(batch_size, num_nodes, -1)
        # early_time_series_emb = self.early_time_series_emb(early_feature)
        #
        # mid_feature = mid_feature.transpose(1, 2).contiguous()
        # mid_feature = mid_feature.view(batch_size, num_nodes, -1)
        # mid_time_series_emb = self.mid_time_series_emb(mid_feature)
        #
        # later_feature = later_feature.transpose(1, 2).contiguous()
        # later_feature = later_feature.view(batch_size, num_nodes, -1)
        # later_time_series_emb = self.last_time_series_emb(later_feature)

        """
        1.第一次尝试，加了CNN提特征
        flow_feature = self.flow_conv(flow_feature)
        speed_feature = self.speed_conv(speed_feature)
        Occ_feature = self.occ_conv(Occ_feature)
        """
        # flow_feature = self.flow_conv(flow_feature)
        # speed_feature = self.speed_conv(speed_feature)
        # Occ_feature = self.occ_conv(Occ_feature)



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

        # day_time_feature = torch.cat([time_in_day_emb,day_in_week_emb],dim=2)
        # day_time_feature = self.week_time_emd(day_time_feature)
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

        hidden = torch.cat([time_series_emb]+[flow_time_series_emb]+[speed_time_series_emb]+[occ_time_series_emb]+node_emb+tem_emb,dim=2)
        # hidden = torch.cat([time_series_emb]+[early_time_series_emb] + [mid_time_series_emb] + [later_time_series_emb]  + node_emb + tem_emb ,dim=2)
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

class STID_WITH_TEST(nn.Module):
    def __init__(self,model_param):
        super(STID_WITH_TEST, self).__init__()
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

        self.speed_node_emb = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.node_dim)
        self.flow_node_emb = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.node_dim)
        self.line_emb = nn.Embedding(num_embeddings=4, embedding_dim=self.node_dim)
        # temporal embeddings
        self.time_in_day_emb = nn.Embedding(num_embeddings=288, embedding_dim=self.temp_dim_tid)
        self.day_in_week_emb = nn.Embedding(num_embeddings=7, embedding_dim=self.temp_dim_diw)

        self.speed_time_in_day_emb = nn.Embedding(num_embeddings=288, embedding_dim=self.temp_dim_tid)
        self.speed_day_in_week_emb = nn.Embedding(num_embeddings=7, embedding_dim=self.temp_dim_diw)
        self.flow_time_in_day_emb = nn.Embedding(num_embeddings=288, embedding_dim=self.temp_dim_tid)
        self.flow_day_in_week_emb = nn.Embedding(num_embeddings=7, embedding_dim=self.temp_dim_diw)

        # embedding layer
        self.time_sires_emb_layer = nn.Linear(in_features=self.input_dim * self.input_len, out_features=self.embed_dim,
                                              bias=True)

        self.spaces_emb = nn.Linear(in_features=14, out_features=10,bias=True)
        # encoding
        # self.hidden_dim = self.embed_dim + self.node_dim * int(self.if_spatial) + \
        #                   self.temp_dim_tid * int(self.if_time_in_day) + \
        #                   self.temp_dim_diw * int(self.if_day_in_week)
        self.hidden_dim = self.embed_dim * 6
        self.encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        self.flow_encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )
        self.speed_encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        # regression
        self.regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)

        self.flow_regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)
        self.speed_regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)



        self.fc_f = nn.Linear(self.output_len, 48)
        self.fc_s = nn.Linear(self.output_len, 48)

        self.flow_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim,bias=True)
        self.speed_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)
        self.occ_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)

        self.early_flow_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.mid_flow_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.later_flow_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)

        self.early_speed_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.mid_speed_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.later_speed_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)

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

        # early_flow_feature = flow_feature[:,0:12,:,:]
        # mid_flow_feature = flow_feature[:, 12:24, :, :]
        # later_flow_feature = flow_feature[:, 24:, :, :]
        #
        # early_speed_feature = speed_feature[:, 0:12, :, :]
        # mid_speed_feature = speed_feature[:, 12:24, :, :]
        # later_speed_feature = speed_feature[:, 24:, :, :]

        """
        flow
        """
        # early_flow_feature = early_flow_feature.transpose(1, 2).contiguous()
        # early_flow_feature = early_flow_feature.view(batch_size, num_nodes, -1)
        # early_flow_emb = self.early_flow_emb(early_flow_feature)
        #
        # mid_flow_feature = mid_flow_feature.transpose(1, 2).contiguous()
        # mid_flow_feature = mid_flow_feature.view(batch_size, num_nodes, -1)
        # mid_flow_emb = self.mid_flow_emb(mid_flow_feature)
        #
        # later_flow_feature = later_flow_feature.transpose(1, 2).contiguous()
        # later_flow_feature = later_flow_feature.view(batch_size, num_nodes, -1)
        # later_flow_emb = self.later_flow_emb(later_flow_feature)
        #
        # flow_emb = [early_flow_emb] + [mid_flow_emb] + [later_flow_emb]
        # """
        # speed
        # """
        # early_speed_feature = early_speed_feature.transpose(1, 2).contiguous()
        # early_speed_feature = early_speed_feature.view(batch_size, num_nodes, -1)
        # early_speed_emb = self.early_speed_emb(early_speed_feature)
        #
        # mid_speed_feature = mid_speed_feature.transpose(1, 2).contiguous()
        # mid_speed_feature = mid_speed_feature.view(batch_size, num_nodes, -1)
        # mid_speed_emb = self.mid_speed_emb(mid_speed_feature)
        #
        # later_speed_feature = later_speed_feature.transpose(1, 2).contiguous()
        # later_speed_feature = later_speed_feature.view(batch_size, num_nodes, -1)
        # later_speed_emb = self.later_speed_emb(later_speed_feature)
        #
        # speed_emb = [early_speed_emb] + [mid_speed_emb] + [later_speed_emb]


        # flow_feature = flow_feature.transpose(1, 2).contiguous()
        # flow_feature = flow_feature.view(batch_size, num_nodes, -1)
        # flow_time_series_emb = self.flow_time_series_emb(flow_feature)
        #
        # speed_feature = speed_feature.transpose(1, 2).contiguous()
        # speed_feature = speed_feature.view(batch_size, num_nodes, -1)
        # speed_time_series_emb = self.speed_time_series_emb(speed_feature)

        # Occ_feature = Occ_feature.transpose(1, 2).contiguous()
        # Occ_feature = Occ_feature.view(batch_size, num_nodes, -1)
        # occ_time_series_emb = self.occ_time_series_emb(Occ_feature)

        flow_feature = flow_feature.transpose(1, 2).contiguous()
        flow_feature = flow_feature.view(batch_size, num_nodes, -1)
        flow_time_series_emb = self.flow_time_series_emb(flow_feature)

        speed_feature = speed_feature.transpose(1, 2).contiguous()
        speed_feature = speed_feature.view(batch_size, num_nodes, -1)
        speed_time_series_emb = self.speed_time_series_emb(speed_feature)

        Occ_feature = Occ_feature.transpose(1, 2).contiguous()
        Occ_feature = Occ_feature.view(batch_size, num_nodes, -1)
        occ_time_series_emb = self.occ_time_series_emb(Occ_feature)



        time_in_day_emb = None
        if self.if_time_in_day:
            t_i_d_data = his_data[:, 1, :, -2].long()
            # (b, node_nums(time_in_day dim-len)) -> T^{TiD}(Nd x D)-> (b, node_nums(time_in_day dim-len), emb)
            flow_time_in_day_emb = self.flow_time_in_day_emb(t_i_d_data)
            speed_time_in_day_emb = self.speed_time_in_day_emb(t_i_d_data)

            time_in_day_emb = self.time_in_day_emb(t_i_d_data)


        day_in_week_emb = None
        if self.if_day_in_week:
            d_i_w_data = his_data[:, 1, :, -1].long()
            # (b, node_nums(day_in_week dim-len)) -> T^{DiW}(Nw x D)-> (b, node_nums(day_in_week dim-len), emb)
            flow_day_in_week_emb = self.flow_day_in_week_emb(d_i_w_data)
            speed_day_in_week_emb = self.speed_day_in_week_emb(d_i_w_data)

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


        flow_node_emb = self.flow_node_emb(nodes_indx)
        speed_node_emb = self.speed_node_emb(nodes_indx)

        flow_hidden = torch.cat([flow_time_series_emb]+[speed_time_series_emb]+[occ_time_series_emb]+[flow_node_emb]+[time_in_day_emb]+[day_in_week_emb],dim=2)
        speed_hidden = torch.cat([flow_time_series_emb]+[speed_time_series_emb]+[occ_time_series_emb] +  [speed_node_emb] + [time_in_day_emb]+[day_in_week_emb], dim=2)

        flow_hidden = self.flow_encoder(flow_hidden)
        speed_hidden = self.flow_encoder(speed_hidden)

        flow_pred = self.flow_regression_layer(flow_hidden)
        speed_pred = self.speed_regression_layer(speed_hidden)



        f_output = self.fc_f(flow_pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        s_output = self.fc_s(speed_pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)

        # return pred.transpose(1, 2).contiguous()
        return f_output,s_output

class My_STID(nn.Module):
    def __init__(self,model_param):
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

        self.speed_node_emb = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.node_dim)
        self.flow_node_emb = nn.Embedding(num_embeddings=self.num_nodes, embedding_dim=self.node_dim)
        self.line_emb = nn.Embedding(num_embeddings=4, embedding_dim=self.node_dim)
        # temporal embeddings
        self.time_in_day_emb = nn.Embedding(num_embeddings=288, embedding_dim=self.temp_dim_tid)
        self.day_in_week_emb = nn.Embedding(num_embeddings=7, embedding_dim=self.temp_dim_diw)

        self.speed_time_in_day_emb = nn.Embedding(num_embeddings=288, embedding_dim=self.temp_dim_tid)
        self.speed_day_in_week_emb = nn.Embedding(num_embeddings=7, embedding_dim=self.temp_dim_diw)
        self.flow_time_in_day_emb = nn.Embedding(num_embeddings=288, embedding_dim=self.temp_dim_tid)
        self.flow_day_in_week_emb = nn.Embedding(num_embeddings=7, embedding_dim=self.temp_dim_diw)

        # embedding layer
        self.time_sires_emb_layer = nn.Linear(in_features=self.input_dim * self.input_len, out_features=self.embed_dim,
                                              bias=True)

        self.spaces_emb = nn.Linear(in_features=14, out_features=10,bias=True)

        self.hidden_dim = self.embed_dim * 6
        self.encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )

        self.flow_encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )
        self.speed_encoder = nn.Sequential(
            *[MyMultiLayerPerceptron(self.hidden_dim, self.hidden_dim) for _ in range(self.num_layer)]
        )



        # regression
        self.regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)

        self.flow_regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)
        self.speed_regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)

        self.flow_layer = nn.Linear(32, self.output_len, bias=True)
        self.speed_layer = nn.Linear(32, self.output_len, bias=True)


        self.fc_f = nn.Linear(self.output_len, 48)
        self.fc_s = nn.Linear(self.output_len, 48)

        self.flow_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim,bias=True)
        self.speed_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)
        self.occ_time_series_emb = nn.Linear(in_features=144, out_features=self.embed_dim, bias=True)

        self.early_flow_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.mid_flow_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.later_flow_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)

        self.early_speed_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.mid_speed_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)
        self.later_speed_emb = nn.Linear(in_features=48, out_features=self.embed_dim, bias=True)

        self.flow_regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)
        self.speed_regression_layer = nn.Linear(self.hidden_dim, self.output_len, bias=True)

        self.flow_conv = nn.Sequential(
                        nn.Conv2d(in_channels=10,out_channels=10,kernel_size=(3,5),stride=(1,1),padding=(1,2)),
                        nn.BatchNorm2d(10),
                        nn.GELU(),
                        nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(6, 1), stride=(1, 1), padding=(0, 0)),
                        nn.GELU(),
                        nn.Dropout(0.15),
        )
        self.speed_conv = nn.Sequential(
            nn.Conv2d(in_channels=10,out_channels=10,kernel_size=(3,5),stride=(1,1),padding=(1,2)),
            nn.BatchNorm2d(10),
            nn.GELU(),
            nn.Conv2d(in_channels=10, out_channels=10, kernel_size=(6, 1), stride=(1, 1), padding=(0, 0)),
            nn.GELU(),
            nn.Dropout(0.15),
        )
        self.occ_conv = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(36),
            nn.GELU()
        )

        self.all_feature_conv = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=36, kernel_size=(3, 3), stride=(1, 1), padding=(1, 1)),
            nn.BatchNorm2d(36),
            nn.ReLU()
        )

        self.flow_mlp = nn.Sequential(
            *[MixerBlock(dim=10,num_patch =self.hidden_dim, token_dim=256,channel_dim= 32,dropout=0.15) for _ in range(self.num_layer)]
        )
        self.speed_mlp = nn.Sequential(
            *[MixerBlock(dim=10,num_patch =self.hidden_dim, token_dim=256,channel_dim= 32,dropout=0.15) for _ in range(self.num_layer)]
        )


        self.layer_norm = nn.LayerNorm(10)

    def forward(self, his_data):



        # input_data = his_data[:,:,:,:-2]
        #
        # flow_feature = input_data[:,:,:,0:4]
        # speed_feature = input_data[:, :, :, 4:8]
        # Occ_feature = input_data[:, :, :, 8:]
        #
        # batch_size, _, num_nodes, _ = input_data.shape
        #
        # flow_feature = flow_feature.transpose(1, 2).contiguous()
        # flow_feature = flow_feature.view(batch_size, num_nodes, -1)
        # flow_time_series_emb = self.flow_time_series_emb(flow_feature)
        #
        # speed_feature = speed_feature.transpose(1, 2).contiguous()
        # speed_feature = speed_feature.view(batch_size, num_nodes, -1)
        # speed_time_series_emb = self.speed_time_series_emb(speed_feature)
        #
        # Occ_feature = Occ_feature.transpose(1, 2).contiguous()
        # Occ_feature = Occ_feature.view(batch_size, num_nodes, -1)
        # occ_time_series_emb = self.occ_time_series_emb(Occ_feature)
        #
        #
        #
        # time_in_day_emb = None
        # if self.if_time_in_day:
        #     t_i_d_data = his_data[:, 1, :, -2].long()
        #     time_in_day_emb = self.time_in_day_emb(t_i_d_data)
        #
        #
        # day_in_week_emb = None
        # if self.if_day_in_week:
        #     d_i_w_data = his_data[:, 1, :, -1].long()
        #     day_in_week_emb = self.day_in_week_emb(d_i_w_data)
        #
        # nodes_indx = torch.Tensor([list(range(num_nodes)) for _ in range(batch_size)]).long().type_as(d_i_w_data)
        # flow_node_emb = self.flow_node_emb(nodes_indx)
        # speed_node_emb = self.speed_node_emb(nodes_indx)
        #
        # # flow_hidden = torch.cat([flow_time_series_emb]+[speed_time_series_emb]+[occ_time_series_emb]+[flow_node_emb]+[time_in_day_emb]+[day_in_week_emb],dim=2)
        # # speed_hidden = torch.cat([flow_time_series_emb]+[speed_time_series_emb]+[occ_time_series_emb] +  [speed_node_emb] + [time_in_day_emb]+[day_in_week_emb], dim=2)
        #
        # flow_all_feature = torch.stack([flow_time_series_emb]+[speed_time_series_emb]+[occ_time_series_emb]+[flow_node_emb]+[time_in_day_emb]+[day_in_week_emb],dim=2)
        # speed_all_feature = torch.stack([flow_time_series_emb]+[speed_time_series_emb]+[occ_time_series_emb] +  [speed_node_emb] + [time_in_day_emb]+[day_in_week_emb], dim=2)
        #
        # flow_all_feature = self.flow_conv(flow_all_feature)
        # speed_all_feature = self.speed_conv(speed_all_feature)
        #
        # flow_all_feature = torch.squeeze(flow_all_feature,dim=2)
        # speed_all_feature = torch.squeeze(speed_all_feature, dim=2)
        #
        #
        #
        # flow_hidden = self.flow_mlp(flow_all_feature)
        # speed_hidden = self.speed_mlp(speed_all_feature)
        #
        # flow_pred = self.flow_layer(flow_hidden)
        # speed_pred = self.speed_layer(speed_hidden)
        #
        # f_output = self.fc_f(flow_pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        # s_output = self.fc_s(speed_pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)

        input_data = his_data[:, :, :, :-2]

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

        time_in_day_emb = None
        if self.if_time_in_day:
            t_i_d_data = his_data[:, 1, :, -2].long()
            time_in_day_emb = self.time_in_day_emb(t_i_d_data)

        day_in_week_emb = None
        if self.if_day_in_week:
            d_i_w_data = his_data[:, 1, :, -1].long()
            day_in_week_emb = self.day_in_week_emb(d_i_w_data)

        nodes_indx = torch.Tensor([list(range(num_nodes)) for _ in range(batch_size)]).long().type_as(d_i_w_data)
        flow_node_emb = self.flow_node_emb(nodes_indx)
        speed_node_emb = self.speed_node_emb(nodes_indx)

        flow_hidden = torch.cat([flow_time_series_emb]+[speed_time_series_emb]+[occ_time_series_emb]+[flow_node_emb]+[time_in_day_emb]+[day_in_week_emb],dim=2)
        speed_hidden = torch.cat([flow_time_series_emb]+[speed_time_series_emb]+[occ_time_series_emb] +  [speed_node_emb] + [time_in_day_emb]+[day_in_week_emb], dim=2)

        flow_hidden = flow_hidden.transpose(1, 2)
        speed_hidden = speed_hidden.transpose(1, 2)

        flow_hidden = self.flow_mlp(flow_hidden)
        speed_hidden = self.speed_mlp(speed_hidden)

        flow_hidden = flow_hidden.transpose(1, 2)
        speed_hidden = speed_hidden.transpose(1, 2)

        flow_pred = self.flow_regression_layer(flow_hidden)
        speed_pred = self.speed_regression_layer(speed_hidden)

        f_output = self.fc_f(flow_pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        s_output = self.fc_s(speed_pred).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)

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
    from torch.utils.data import DataLoader

    val_data = np.load('val.npy')
    val_loader = DataLoader(MyDataSet(val_data), batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    adj = np.load('adj_mat.npy')
    model = My_STID(model_param)
    for index,(x_sequence, y_flow,y_speed) in enumerate(val_loader):
        out_put = model(x_sequence)
        pass
    # for index,(x_sequence, y_flow,y_speed,aux_x_sequence) in enumerate(val_loader):
    #     out_put = model(x_sequence,aux_x_sequence)
    #     pass
