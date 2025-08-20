import math
import torch.nn as nn
from torch.nn.modules.utils import _triple
import torch


class SpatioTemporalConv(nn.Module):
    def __init__(self):
        super(SpatioTemporalConv, self).__init__()
        self.flow_temporal_encoder = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=72, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Conv2d(in_channels=72, out_channels=18, kernel_size=(1, 4), stride=(1, 1), padding=(0, 0)),
            nn.ReLU()
        )
        self.speed_temporal_encoder = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=72, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Conv2d(in_channels=72, out_channels=18, kernel_size=(1, 4), stride=(1, 1), padding=(0, 0)),
            nn.ReLU()
        )
        self.occ_temporal_encoder = nn.Sequential(
            nn.Conv2d(in_channels=36, out_channels=72, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0)),
            nn.BatchNorm2d(72),
            nn.ReLU(),
            nn.Conv2d(in_channels=72, out_channels=18, kernel_size=(1, 4), stride=(1, 1), padding=(0, 0)),
            nn.ReLU()
        )

    def forward(self,flow_feature,speed_feature,occ_feature):
        """

        """
        flow_feature = self.flow_temporal_encoder(flow_feature)
        speed_feature = self.speed_temporal_encoder(speed_feature)
        occ_feature = self.occ_temporal_encoder(occ_feature)

        fusion_feature = torch.cat([flow_feature,speed_feature,occ_feature],dim=1)
        fusion_feature = fusion_feature.squeeze().transpose(1, 2)

        return fusion_feature