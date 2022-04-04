from torch import nn
import ck_g_cnn.nn.functional as gF


class GroupMaxPoolingRn(nn.Module):

    def __init__(self, kernel_size=2, stride_Rn=2):
        super().__init__()
        self.kernel_size = kernel_size
        self.stride_Rn = stride_Rn

    def forward(self, x, grid_H):
        return gF.group_max_pooling_Rn(
            x=x,
            kernel_size=self.kernel_size,
            stride_Rn=self.stride_Rn
        ), grid_H
