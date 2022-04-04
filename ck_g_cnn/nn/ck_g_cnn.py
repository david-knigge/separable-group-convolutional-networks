from ck_g_cnn.nn.ck_g_cnn_base import CKGCNNBase

from ck_g_cnn.nn.normalisation import LayerNorm, InstanceNorm, BatchNorm
from ck_g_cnn.nn.activation import ReLU
from ck_g_cnn.nn.pooling import GroupMaxPoolingRn


class CKGCNN(CKGCNNBase):

    def __init__(
            self,
            group,
            in_channels,
            out_channels,
            spatial_in_size,
            implementation,
            num_group_elem,
            kernel_size,
            hidden_sizes,
            ck_net_num_hidden,
            ck_net_hidden_size,
            ck_net_implementation,
            ck_net_first_omega_0,
            ck_net_omega_0,
            dropout,
            pooling,
            bias,
            sampling_method,
            normalisation,
            stride
    ):
        """ Continuous Kernel Group Convolutional Neural Network. This module implements the CKGCNN for a given group.

        @param group: Group instance, defining sampling method over Lie algebra, exponential and logarithmic map, and
            group actions on Rd and H.
        @param in_channels: Number of input channels.
        @param out_channels: Number of output channels.
        @param num_group_elem: Number of group elements to be sampled in each forward pass.
        @param kernel_sizes: Kernel sizes to be used in each hidden layer, if a single integer is given, all layers will
            have identical kernel sizes.
        @param hidden_sizes: Sizes of each hidden layer.
        @param ck_net_num_hidden: Number of hidden layers in the continuous kernel net.
        @param ck_net_hidden_size: Number of channels in each layer of the continuous kernel net.
        @param first_omega_0: Omega_0 for the first layer of the continuous kernel net.
        @param omega_0: Omega_0 for hidden layers of the continuous kernel net.
        @param normalisation_fn: Normalisation function to be used, if None, no normalisation is applied.
        @param activation_fn: Activation function to be used after each lifting or group convolution.
        @param pooling_fn: Pooling function to be used after each lifting or group convolution, if None, no pooling is
            applied
        @param sampling_method: Sampling method to use on the group, "discretise" deterministically discretizes the
            group, "uniform" discretizes the group, then afterwards applies a random perturbation.
        """
        super(CKGCNN, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            spatial_in_size=spatial_in_size,
            implementation=implementation,
            kernel_size=kernel_size,
            hidden_sizes=hidden_sizes,
            normalisation=normalisation
        )

        self.lifting = self.lifting_impl(
            group=group,
            kernel_size=self.kernel_size,
            padding=True,
            stride=1,
            in_channels=self.in_channels,
            out_channels=self.hidden_sizes[0],
            num_group_elem=num_group_elem,
            ck_net_num_hidden=ck_net_num_hidden,
            ck_net_hidden_size=ck_net_hidden_size,
            ck_net_implementation=ck_net_implementation,
            ck_net_first_omega_0=ck_net_first_omega_0,
            ck_net_omega_0=ck_net_omega_0,
            sampling_method=sampling_method,
        )

        # define activation and normalisation functions
        self.layers.extend([
            self.normalisation_fn(channels=self.hidden_sizes[0]),
            GroupMaxPoolingRn(),
            ReLU()
        ])

        # define groups convolutions
        for h_index in range(1, len(self.hidden_sizes)):
            self.layers.append(
                self.groupconv_impl(
                    group=group,
                    kernel_size=self.kernel_size,
                    stride=stride,
                    padding=True,
                    in_channels=self.hidden_sizes[h_index - 1],
                    out_channels=self.hidden_sizes[h_index],
                    num_group_elem=num_group_elem,
                    ck_net_num_hidden=ck_net_num_hidden,
                    ck_net_hidden_size=ck_net_hidden_size,
                    ck_net_implementation=ck_net_implementation,
                    ck_net_first_omega_0=ck_net_first_omega_0,
                    ck_net_omega_0=ck_net_omega_0,
                    sampling_method=sampling_method
                )
            )

            self.layers.append(self.normalisation_fn(channels=self.hidden_sizes[h_index]))

            if pooling:
                self.layers.append(GroupMaxPoolingRn())

            self.layers.append(ReLU())

        # final gconv to reduce spatial dims
        self.layers.extend([
            self.groupconv_impl(
                group=group,
                kernel_size=self.spatial_in_size // (2 ** len(hidden_sizes)),
                stride=1,
                in_channels=self.hidden_sizes[-1],
                out_channels=self.hidden_sizes[-1],
                num_group_elem=num_group_elem,
                padding=False,
                ck_net_num_hidden=ck_net_num_hidden,
                ck_net_hidden_size=ck_net_hidden_size,
                ck_net_implementation=ck_net_implementation,
                ck_net_first_omega_0=ck_net_first_omega_0,
                ck_net_omega_0=ck_net_omega_0,
                sampling_method=sampling_method
            ),
            self.normalisation_fn(channels=self.hidden_sizes[-1]),
            ReLU()
        ])
