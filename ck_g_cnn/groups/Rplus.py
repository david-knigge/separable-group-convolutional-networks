import math
import numpy as np

import torch

from .group import Group

GroupElement = torch.Tensor


class Rplus(Group):

    def __init__(self, max_scale=3):
        super(Rplus, self).__init__(dimension=1, identity=[1.])
        self.max_scale = torch.tensor(max_scale)
        # self.max_scale = max_scale

    def product(self, g1, g2):
        """ Calculate the group product of two elements of the scale group

        :param g1: a group element from the scaling group
        :param g2: a group element from the scaling group
        """
        return torch.clamp(g1 * g2, min=0, max=self.max_scale)

    def inverse(self, g):
        """ Return the inverse of a group element for the scaling group

        :param g: a group element from the scaling group
        """
        return 1/g

    def logarithmic_map(self, g):
        """ Logarithmic map from group to algebra

        :param g: a group element from the scaling group
        """
        return torch.log(g)

    def exponential_map(self, h):
        """ Exponential map from algebra to group

        @param h: a lie algebra element from the scaling group
        @return:
        """
        return torch.exp(h)

    def left_action_on_H(self, g_batch, x):
        return g_batch.unsqueeze(0).T * x.repeat(g_batch.shape[0], 1)

    def left_action_on_Rd(self, g_batch, x):
        """ Transform Rd input meshgrid by batch of group elements g

        @param g: set of scaling group elements [num_group_elements]
        @param x: a meshgrid with relative positions on Rd,
            expected format: [d, num_el_x, num_el_y]
        """
        g_rep = self.representation(g_batch)
        return torch.einsum('oi,ijk->oijk', g_rep, x)

    def representation(self, g):
        """ Create a representation for a group element. We model for uniform scaling, so we repeat
            the group element in both spatial dimensions.

        :param g: a group element from the scaling group
        """
        return g.unsqueeze(-1).repeat(1, 2)

    def determinant(self, m):
        """ Return the determinant for the matrix representation of
        a group element.

        :param m: a representation of a scaling group element
        """
        return m ** 2

    def sample(self, num_elements, method="discretise"):
        """ Sample a set of group elements. These elements are subsequently used to transform
        the ckconv kernel grids.

        :param num_elements: number of group elements to sample
        :param method: sampling method
        """
        if method == "discretise":
            return self.exponential_map(torch.linspace(
                0, np.log(self.max_scale.item()),
                num_elements,
                dtype=torch.float,
                device=self.identity.device
            ))

        elif method == "uniform":
            unif_grid = torch.linspace(
                0, np.log(self.max_scale.item()),
                num_elements,
                dtype=torch.float,
                device=self.identity.device
            )

            # create a perturbation of the uniform grid of at most 1 group element
            perturbation = torch.rand(1, device=self.identity.device) * (np.log(self.max_scale.item()) / num_elements)

            return self.exponential_map(torch.remainder(unif_grid + perturbation, self.max_scale))

        elif method == "normal":
            raise NotImplementedError()
