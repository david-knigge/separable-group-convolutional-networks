import torch
from torch import nn


class Dropout(nn.Module):

    def __init__(self, p):
        super(Dropout, self).__init__()
        self.dropout = nn.Dropout(p)

    def forward(self, x, grid_H):
        return self.dropout(x), grid_H