import torch


class ContinuousKernelGroupConv(torch.nn.Module):

    def __init__(self,
                 in_channels,
                 out_channels,
                 group,
                 num_group_elem,
                 kernel_size,
                 sampling_method,
                 padding,
                 stride,
                 bias=False):
        """ Base class for group convolution operation.

        :param in_channels: Number of channels of input feature map.
        :param out_channels: Number of channels of output feature map.
        :param group: Group implementation.
        :param num_group_elem: Number of group elements to sample / discretise group by.
        :param kernel_size: Convolution kernel size.
        :param sampling_method: Sampling method over the group, can either be 'discretise' or 'uniform'.
        :param padding: Whether or not to apply padding to the input spatial dimensions.
        :param stride: Convolution stride.
        :param bias: Whether or not to apply bias.
        """
        super().__init__()

        self.in_channels = in_channels
        self.out_channels = out_channels
        self.group = group

        # number of group elements to sample for each convolution
        self.num_group_elem = num_group_elem
        self.sampling_method = sampling_method

        self.kernel_size = kernel_size
        self.stride = stride

        if padding:
            padding_width = self.kernel_size // 2
            padding_height = self.kernel_size // 2
            self.padding = (padding_width, padding_height) # no padding in the groups dimension
        else:
            self.padding = 0

        if bias:
            self.bias = torch.zeros(out_channels, device=self.group.identity.device)
        else:
            self.bias = None

    def forward(self, **kwargs):
        raise NotImplementedError()

    def extra_repr(self) -> str:
        er = ''
        er += f"(num_elem) {self.num_group_elem}\n"
        er += f"(kernel_size) {str(self.kernel_size)}\n"
        er += f"(padding) {str(self.padding)}\n"
        # er += f"(stride) {str(self.stride)}\n"
        er += f"(no_trainable_params) {str(sum(p.numel() for p in self.parameters() if p.requires_grad))}"
        return er
