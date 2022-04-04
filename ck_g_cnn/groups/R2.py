import math

import numpy as np
import torch

from .group import Group

GroupElement = torch.Tensor


class R2(Group):

    def __init__(self):
        super(R2, self).__init__(dimension=2, identity=[0., 0.])

    def product(self, g1, g2):
        return g1 + g2

    def inverse(self, g):
        return -g

    def logarithmic_map(self, g):
        return g

    def exponential_map(self, h):
        return h

    def determinant(self, m):
        return 1.

    def left_action_on_H(self, g, x):
        return x

    def left_action_on_Rd(self, g, x):
        return R2.prod(g, x)

    def representation(self, g):
        return g
