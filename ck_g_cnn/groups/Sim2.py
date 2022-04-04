# Taken from https://github.com/dwromero/g_selfatt

import math

import numpy as np
import torch

from .group import Group

GroupElement = torch.Tensor

from ck_g_cnn.groups.Rplus import Rplus
from ck_g_cnn.groups.SO2 import SO2


class SO2xRplus(Group):

    def __init__(self, max_scale=3.):
        """ Rotation scaling subgroup, elements of this group are of the form [angle, scale].

        @param max_scale: scale value at which to truncate scale group
        """
        super(SO2xRplus, self).__init__(dimension=2, identity=[0., 1.])
        self.SO2 = SO2()
        self.Rplus = Rplus(max_scale=max_scale)

    def product(self, g1, g2):
        """ Calculate the semi-direct group product of two elements of the rotation-scale group

        @param g1:
        @param g2:
        @return:
        """
        g1 = g1.clone()
        # rotation elements should be between 0 and 2pi
        g1[:, 0] = self.SO2.product(g1[:, 0], g2[:, 0])

        # scale elements should be between 1 and max scale
        g1[:, 1] = self.Rplus.product(g1[:, 1], g2[:, 1])
        return g1

    def inverse(self, g):
        """ Calculate the inverse of a group element of the rotation-scale group

        @param g:
        @type g:
        @return:
        @rtype:
        """
        g = g.clone()
        g[:, 0] = self.SO2.inverse(g[:, 0])
        g[:, 1] = self.Rplus.inverse(g[:, 1])
        return g

    def exponential_map(self, h):
        h = h.clone()
        h[:, 0] = self.SO2.exponential_map(h[:, 0])
        h[:, 1] = self.Rplus.exponential_map(h[:, 1])
        return h

    def logarithmic_map(self, g):
        g = g.clone()
        g[:, :, 0] = self.SO2.logarithmic_map(g[:, :, 0])
        g[:, :, 1] = self.Rplus.logarithmic_map(g[:, :, 1])
        return g

    def left_action_on_H(self, g, x):
        # expand to number of output group elements
        x = x.repeat(g.shape[0], 1, 1)
        # rotation product
        x[:, :, 0] = torch.remainder(g[:, 0].unsqueeze(-1) + x[:, :, 0], 2 * np.pi)
        # scale product
        x[:, :, 1] = g[:, 1].unsqueeze(-1) * x[:, :, 1]
        return x

    def left_action_on_Rd(self, g, x):
        """ Here, we make use of the fact that uniform scaling and rotation are commutative!

        @param g:
        @param x:
        """
        # obtain representations for scale and rotation group elements
        rot_rep, scale_rep = self.representation(g)

        # apply rotation, expand to number of output group elements
        x = torch.einsum('boi,ixy->boxy', rot_rep, x)

        # apply scaling
        x = scale_rep[:, :, None, None] * x
        return x

    def representation(self, g):
        return self.SO2.representation(g[:, 0]), self.Rplus.representation(g[:, 1])

    def determinant(self, m):
        """ Here, the determinant is equal to the scale parameter. (?)

        @param m:
        @type m:
        @return:
        @rtype:
        """
        return m[:, 1]

    def sample(self, num_elements, method='discretise', separable=False):
        """

        @param num_elements:
        @param method:
        """
        if type(num_elements) == list:
            SO2_elem = self.SO2.sample(num_elements=num_elements[0], method=method)
            Rplus_elem = self.Rplus.sample(num_elements=num_elements[1],
                                           method='discretise')  # always discretise scale group
        else:
            SO2_elem = self.SO2.sample(num_elements=num_elements, method=method)
            Rplus_elem = self.Rplus.sample(num_elements=num_elements, method='discretise') # always discretise scale group
        if separable:
            return SO2_elem, Rplus_elem
        else:
            return torch.stack(torch.meshgrid(SO2_elem, Rplus_elem)).view(self.dimension, -1).T

    def normalize(self, g):
        """

        @param g:
        """
        g[:, :, 0] = self.SO2.normalize(g[:, :, 0])
        # g[:, :, 1] = self.Rplus.normalize(g[:, :, 1])
        return g
