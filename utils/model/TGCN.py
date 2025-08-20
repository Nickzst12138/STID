import torch
import torch.nn as nn
import numpy as np
import sys
from ST import SpatioTemporalConv

from torch.utils.data import DataLoader

def calculate_laplacian_with_self_loop(matrix):
    matrix = matrix + torch.eye(matrix.size(0))
    row_sum = matrix.sum(1)
    d_inv_sqrt = torch.pow(row_sum, -0.5).flatten()
    d_inv_sqrt[torch.isinf(d_inv_sqrt)] = 0.0
    d_mat_inv_sqrt = torch.diag(d_inv_sqrt)
    normalized_laplacian = matrix.matmul(d_mat_inv_sqrt).transpose(0, 1).matmul(d_mat_inv_sqrt)
    return normalized_laplacian


class TGCNGraphConvolution(nn.Module):
    def __init__(self, adj, num_gru_units: int, output_dim: int, input_dim: int, bias: float = 0.0):
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
        self.graph_conv1 = TGCNGraphConvolution(self.adj, self._hidden_dim, self._hidden_dim * 2, self._hidden_dim, bias=1.0)
        self.graph_conv2 = TGCNGraphConvolution(self.adj, self._hidden_dim, self._hidden_dim, self._hidden_dim)

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
    def __init__(self, adj, input_dim: int = 12, hidden_dim: int = 64, **kwargs):
        super(TGCN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim)
        self.time_embedding = nn.Embedding(288, 32)
        self.week_embedding = nn.Embedding(7, 32)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim+32*2, self._hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self._hidden_dim//2, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
        )
        self.fc_f = nn.Linear(self._hidden_dim, 48)
        self.fc_s = nn.Linear(self._hidden_dim, 48)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes, input_dim = inputs.shape
        time_emb = self.time_embedding(inputs[:,:,:,-2].long())
        week_emb = self.week_embedding(inputs[:,:,:,-1].long())
        inputs = self.mlp(torch.cat([inputs[:,:,:,:-2], time_emb, week_emb], dim=3))
        assert self._input_dim == num_nodes

        flow_feature, speed_feature, occ_feature = inputs[:, :, :, :4], inputs[:, :, :, 4:8], inputs[:, :, :, 8:12]

        """
        进入GCN+GRU部分
        """
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

class TGCN_CNN(nn.Module):
    def __init__(self, adj, input_dim: int = 12, hidden_dim: int = 64, **kwargs):
        super(TGCN_CNN, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim)
        self.time_embedding = nn.Embedding(288, 32)
        self.week_embedding = nn.Embedding(7, 32)
        self.SpatioTemporal_Conv = SpatioTemporalConv()
        self.tgcn_cell_r = TGCNCell(self.adj, self._input_dim, self._hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim, self._hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self._hidden_dim//2, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
        )
        # self.fc_f = nn.Linear(self._hidden_dim, 16)
        # self.fc_s = nn.Linear(self._hidden_dim, 16)

        self.fc_s = nn.Linear(229, 48)
        self.fc_f = nn.Linear(229, 48)

        self.flow_temporal_encoder = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=72, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU(),
            nn.BatchNorm2d(72),
            nn.Conv2d(in_channels=72, out_channels=36, kernel_size=(1, 4), stride=(1, 1), padding=(0, 0)),
            nn.ReLU()
        )
        self.speed_temporal_encoder = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU()
        )
        self.occ_temporal_encoder = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=1, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.ReLU()
        )

        self.time_week_fusion = nn.Sequential(
            nn.Linear(32, self._hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self._hidden_dim//2, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, 10),
        )
        self.speed_temporal_fusion = nn.Sequential(
            nn.Linear(54, self._hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(self._hidden_dim // 2, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
        )

    def forward(self, inputs):
        batch_size, seq_len, num_nodes, input_dim = inputs.shape
        time_emb = self.time_embedding(inputs[:,:,:,-2].long())
        week_emb = self.week_embedding(inputs[:,:,:,-1].long())

        time_emb = time_emb[:,:,0,:]
        week_emb = week_emb[:,:,0,:][:,1:2,:]

        flow_feature,speed_feature,occ_feature = inputs[:,:,:,:4],inputs[:,:,:,4:8],inputs[:,:,:,8:12]
        fusion_feature = self.SpatioTemporal_Conv(flow_feature,speed_feature,occ_feature)
        fusion_w_t = torch.concat([time_emb,week_emb],dim=1)

        fusion_w_t = self.time_week_fusion(fusion_w_t).transpose(1, 2)
        fusion_feature = self.speed_temporal_fusion(fusion_feature)

        # inputs = self.mlp(torch.cat([inputs[:, :, :, :-2], time_emb, week_emb], dim=3))

        inputs = self.mlp(inputs[:,:,:,:-2])
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)

        for i in range(seq_len - 1, -1, -1):
            output_r, hidden_state = self.tgcn_cell_r(inputs[:, i, :, :], hidden_state)
            output_r = output_r.reshape((batch_size, num_nodes, self._hidden_dim))
        output = torch.cat([output, output_r], dim=2)

        output = torch.concat([output,fusion_feature,fusion_w_t],dim=2)

        f_output = self.fc_f(output).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        s_output = self.fc_s(output).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)
        # output = self.fc(output).reshape((batch_size, num_nodes, 2, 12, 4))
        # output = output.transpose(1, 2).transpose(0, 1).transpose(2, 3)  # 2, bs, 12, 10, 4
        return f_output, s_output

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}

class TGCN_speed(nn.Module):
    def __init__(self, adj, input_dim: int = 12, hidden_dim: int = 64, **kwargs):
        super(TGCN_speed, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim)
        self.time_embedding = nn.Embedding(288, 32)
        self.week_embedding = nn.Embedding(7, 32)
        self.tgcn_cell_r = TGCNCell(self.adj, self._input_dim, self._hidden_dim)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim+32*2, self._hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self._hidden_dim//2, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
        )
        self.fc_s = nn.Linear(self._hidden_dim*2, 48)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes, input_dim = inputs.shape
        time_emb = self.time_embedding(inputs[:,:,:,-2].long())
        week_emb = self.week_embedding(inputs[:,:,:,-1].long())

        flow_feature,speed_feature,occ_feature = inputs[:,:,:,:4],inputs[:,:,:,4:8],inputs[:,:,:,8:12]


        inputs = self.mlp(torch.cat([inputs[:,:,:,:-2], time_emb, week_emb], dim=3))
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))

        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)

        for i in range(seq_len - 1, -1, -1):
            output_r, hidden_state = self.tgcn_cell_r(inputs[:, i, :, :], hidden_state)
            output_r = output_r.reshape((batch_size, num_nodes, self._hidden_dim))
        output = torch.cat([output, output_r], dim=2)
        s_output = self.fc_s(output).reshape((batch_size, num_nodes, 12, 4)).transpose(1, 2)

        return s_output

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}

class TGCN_flow(nn.Module):
    def __init__(self, adj, input_dim: int = 12, hidden_dim: int = 64, **kwargs):
        super(TGCN_flow, self).__init__()
        self._input_dim = adj.shape[0]
        self._hidden_dim = hidden_dim
        self.register_buffer("adj", torch.FloatTensor(adj))
        self.tgcn_cell = TGCNCell(self.adj, self._input_dim, self._hidden_dim)
        self.tgcn_cell_r = TGCNCell(self.adj, self._input_dim, self._hidden_dim)
        self.time_embedding = nn.Embedding(288, 32)
        self.week_embedding = nn.Embedding(7, 32)
        self.mlp = nn.Sequential(
            nn.Linear(input_dim+32*2, self._hidden_dim//2),
            nn.ReLU(),
            nn.Linear(self._hidden_dim//2, self._hidden_dim),
            nn.ReLU(),
            nn.Linear(self._hidden_dim, self._hidden_dim),
        )
        self.fc_f = nn.Linear(self._hidden_dim, 48)
        self.fc_s = nn.Linear(self._hidden_dim, 48)

    def forward(self, inputs):
        batch_size, seq_len, num_nodes, input_dim = inputs.shape
        time_emb = self.time_embedding(inputs[:,:,:,-2].long())
        week_emb = self.week_embedding(inputs[:,:,:,-1].long())

        flow_feature,speed_feature,occ_feature = inputs[:,:,:,:4],inputs[:,:,:,4:8],inputs[:,:,:,8:12]


        inputs = self.mlp(torch.cat([inputs[:,:,:,:-2], time_emb, week_emb], dim=3))
        assert self._input_dim == num_nodes
        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)
        output = None
        for i in range(seq_len):
            output, hidden_state = self.tgcn_cell(inputs[:, i, :, :], hidden_state)
            output = output.reshape((batch_size, num_nodes, self._hidden_dim))

        hidden_state = torch.zeros(batch_size, num_nodes * self._hidden_dim).type_as(inputs)

        for i in range(seq_len - 1, -1, -1):
            output_r, hidden_state = self.tgcn_cell_r(inputs[:, i, :, :], hidden_state)
            output_r = output_r.reshape((batch_size, num_nodes, self._hidden_dim))
        output = torch.cat([output, output_r], dim=2)
        f_output = self.fc_f(output).reshape((batch_size, num_nodes,12, 4)).transpose(1, 2)

        return f_output

    @property
    def hyperparameters(self):
        return {"input_dim": self._input_dim, "hidden_dim": self._hidden_dim}




if __name__ == '__main__':
    sys.path.append("..")
    from Data_Load_TGCN import MyDataSet

    val_data = np.load('val.npy')
    val_loader = DataLoader(MyDataSet(val_data), batch_size=2, shuffle=True, num_workers=0, pin_memory=True)
    adj = np.load('adj_mat.npy')
    model = TGCN_CNN(adj)
    for index,(x_sequence, y_flow,y_speed) in enumerate(val_loader):
        out_put = model(x_sequence)
