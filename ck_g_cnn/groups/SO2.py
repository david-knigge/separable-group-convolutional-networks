# Taken from https://github.com/dwromero/g_selfatt

import math

import numpy as np
import torch

from .group import Group

GroupElement = torch.Tensor


class SO2(Group):

    def __init__(self):
        super(SO2, self).__init__(dimension=1, identity=[0.])

    def product(self, g1, g2):
        """ Computes the group product of two group elements.

        @param g1:
        @param g2:
        """
        return torch.remainder(g1 + g2, 2 * np.pi)

    def inverse(self, g):
        """

        @param g:
        @return:
        """
        return -g

    def exponential_map(self, h):
        """ Exponential map from algebra to group

        @param h: a lie algebra element from the rotation group
        @return:
        """
        return torch.remainder(h, 2 * np.pi)

    def logarithmic_map(self, g):
        """ Logarithmic map from group to algebra

        :param g: a group element from the rotation group
        """
        return g

    def left_action_on_Rd(self, g_batch, x):
        """ Transform an Rd input meshgrid by group element g

        :param g: a tensor of group elements from the rotation group
        :param x: a meshgrid with relative positions on Rd,
            expected format: [d, num_el_x, num_el_y]
        """
        batch_rep = self.representation(g_batch)
        return torch.einsum('boi,ixy->boxy', batch_rep, x)

    def left_action_on_H(self, g_batch, x):
        """ Transform an Rd input meshgrid by group element g

        :param g: a tensor of group elements from the rotation group [num_elements]
        :param x: batch of rotation group grid points [num_elements]
        """
        return torch.remainder(g_batch.unsqueeze(-1) + x.repeat(g_batch.shape[0], 1), 2 * np.pi)

    def representation(self, g):
        """ Create a representation for a group element.

        :param g: a group element from the rotation group
        """
        cos_t = torch.cos(g)
        sin_t = torch.sin(g)

        batch_rep = torch.stack((
            cos_t, -sin_t,
            sin_t, cos_t
        ), dim=1).view(-1, 2, 2)
        return batch_rep

    def determinant(self, m):
        return torch.ones_like(m)

    def sample(self, num_elements, method="discretise"):
        """ Sample a set of group elements. These elements are subsequently used to transform
        the ckconv kernel grids. We sample a grid uniformly on the Lie algebra, which we map to the group with the
        exponential map.


        :param num_elements: number of group elements to sample
        :param method: sampling method
        """

        if method == "discretise":
            return self.exponential_map(torch.linspace(
                0, 2 * math.pi * float(num_elements - 1) / float(num_elements),
                num_elements,
                dtype=torch.float,
                device=self.identity.device
            ))
        elif method == "uniform":
            unif_grid = torch.linspace(
                0, 2 * math.pi * float(num_elements - 1) / float(num_elements),
                num_elements,
                dtype=torch.float,
                device=self.identity.device
            )

            # create a perturbation of the uniform grid of at most 1 group element
            # perturbation = torch.rand(1, device=self.identity.device) * 2 * math.pi * float(num_elements - 1) / float(num_elements ** 2)
            perturbation = torch.rand(1, device=self.identity.device) * 2 * math.pi * float(num_elements - 1) / float(
                num_elements ** 2)
            return self.exponential_map(unif_grid + perturbation)
        elif method == "normal":
            raise NotImplementedError

    def normalize(self, g):
        """ Normalize values of group elements to range between -1 and 1 for CKNet

        :param g:
        :return:
        """
        return (g / np.pi) - 1.
