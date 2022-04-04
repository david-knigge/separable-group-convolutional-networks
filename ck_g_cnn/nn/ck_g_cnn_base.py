import torch
import torch.nn as nn

from ck_g_cnn.nn.conv.lifting_conv import LiftingConv
from ck_g_cnn.nn.conv.lifting_conv_2d import LiftingConv2D

from ck_g_cnn.nn.conv.nonseparable.group_conv import GroupConv

from ck_g_cnn.nn.conv.separable.group_conv_sep import GroupConvSeparable
from ck_g_cnn.nn.conv.gseparable.group_conv_gsep import GroupConvGSeparable

from ck_g_cnn.nn.conv.separable.group_conv_sep_2d import GroupConvSeparable2D
from ck_g_cnn.nn.conv.gseparable.group_conv_gsep_2d import GroupConvGSeparable2D

from ck_g_cnn.nn.conv.dseparable.group_conv_dsep import GroupConvDSeparable
from ck_g_cnn.nn.conv.dgseparable.group_conv_dgsep import GroupConvDGSeparable

from ck_g_cnn.nn.normalisation import BatchNorm, InstanceNorm, LayerNorm


class CKGCNNBase(nn.Module):

    def __init__(
            self,
            in_channels,
            out_channels,
            spatial_in_size,
            implementation,
            normalisation,
            kernel_size,
            hidden_sizes,
            widen_factor=1,
    ):
        """ Continuous Kernel Group Convolutional Neural Network. This module implements the CKGCNN for a given group.

        @param in_channels:
        @param out_channels:
        @param spatial_in_size:
        @param implementation:
        @param kernel_sizes:
        @param hidden_sizes:
        """
        super(CKGCNNBase, self).__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.spatial_in_size = spatial_in_size
        self.implementation = implementation

        self.kernel_size = kernel_size
        self.hidden_sizes = [int(widen_factor * h) for h in hidden_sizes]

        if implementation in ["nonseparable", "separable", "gseparable", "dseparable", "dgseparable"]:
            self.lifting_impl = LiftingConv
        elif implementation in ["separable+2d", "gseparable+2d"]:
            self.lifting_impl = LiftingConv2D
        else:
            raise ValueError("Incorrect implementation.")

        if implementation == "nonseparable":
            self.groupconv_impl = GroupConv
        elif implementation == "separable":
            self.groupconv_impl = GroupConvSeparable
        elif implementation == "gseparable":
            self.groupconv_impl = GroupConvGSeparable
        elif implementation == "separable+2d":
            self.groupconv_impl = GroupConvSeparable2D
        elif implementation == "gseparable+2d":
            self.groupconv_impl = GroupConvGSeparable2D
        elif implementation == "dseparable":
            self.groupconv_impl = GroupConvDSeparable
        elif implementation == "dgseparable":
            self.groupconv_impl = GroupConvDGSeparable
        else:
            raise ValueError("Incorrect implementation.")

        # define lifting convolution
        self.lifting = None

        # this modulelist should contain all torch modules after the lifting layer, except the final linear layer
        self.layers = nn.ModuleList()

        if normalisation == 'batchnorm':
            self.normalisation_fn = BatchNorm
        elif normalisation == 'layernorm':
            self.normalisation_fn = LayerNorm
        elif normalisation == 'instancenorm':
            self.normalisation_fn = InstanceNorm
        else:
            self.normalisation_fn = None

        # final linear layer to map from last hidden layer to output
        self.final_linear = nn.Sequential( #ivan sosnovik
            nn.Linear(self.hidden_sizes[-1], 256, bias=False),
            nn.BatchNorm1d(256),
            nn.ReLU(True),
            nn.Dropout(0.3),
            nn.Linear(256, self.out_channels)
        )
        # self.final_linear = torch.nn.Linear(self.hidden_sizes[-1], self.out_channels)

    def extra_repr(self) -> str:
        er = f"number of trainable parameters: {sum(p.numel() for p in self.parameters() if p.requires_grad)}\n"
        return er

    def set_group_elem(self, num_group_elem):
        """ Set the number of group elements to sample in a forward pass for each module.

        @param num_group_elem: Number of group elements to sample in each forward pass.
        """
        # set lifting convolution number of group elements
        self.lifting.kernel.num_group_elem = num_group_elem

        # set number of group elements for gconv blocks
        for gconv_block in self.gconvs:
            gconv_block.gconv_1.kernel.num_group_elem = num_group_elem
            gconv_block.gconv_2.kernel.num_group_elem = num_group_elem

        # set number of group elements for pooling conv kernel
        self.gconv_spatial_pooling.kernel.num_group_elem = num_group_elem

    def forward(self, x):
        """ Forward pass of CKGCNN, sequentially applies lifting and group convolutional layers. Uses normalisation,
        activation and optionally pooling functions defined in initialisation.

        @param x: Input sample.
        """

        x, grid_H = self.lifting(x)

        for mod in self.layers:
            x, grid_H = mod(x, grid_H)

        # take max over spatial and groups dimension to ensure invariance
        x = torch.amax(x, dim=(-3, -2, -1))

        x = x.view(-1, x.shape[1])
        x = self.final_linear(x)

        return x
