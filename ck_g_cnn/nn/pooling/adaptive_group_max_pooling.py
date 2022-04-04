from torch import nn
import ck_g_cnn.nn.functional as gF


class AdaptiveGroupMaxPooling(nn.Module):

    def __init__(self, output_size=1):
        super(AdaptiveGroupMaxPooling, self).__init__()
        self.output_size = output_size

    def forward(self, x, grid_H):
        return gF.adaptive_group_max_pooling_Rn(x, output_size=self.output_size), grid_H
