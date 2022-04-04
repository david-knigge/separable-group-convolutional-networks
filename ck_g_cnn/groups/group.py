# Taken from https://github.com/dwromero/g_selfatt

from typing import TypeVar

import torch

GroupElement = TypeVar("GroupElement")


class Group(torch.nn.Module):

    def __init__(self, dimension, identity):
        """ Implements a Lie group.

        @param dimension: Dimensionality of the lie group (number of dimensions in the basis of the algebra).
        @param identity: Identity element of the group.
        """
        super(Group, self).__init__()
        self.dimension = dimension
        self.register_buffer('identity', torch.Tensor(identity))

    def product(self, g1, g2):
        """ Defines group product on two group elements.

        @param g1: Group element 1
        @param g2: Group element 2
        """
        raise NotImplementedError()

    def inverse(self, g):
        """ Defines inverse for group element.

        @param g: A group element.
        """
        raise NotImplementedError()

    def logarithmic_map(self, g):
        """ Defines logarithmic map from lie group to algebra.

        @param g: A Lie group element.
        """
        raise NotImplementedError()

    def exponential_map(self, h):
        """ Defines exponential map from lie algebra to group.

        @param h: A Lie algebra element.
        """
        raise NotImplementedError()

    def determinant(self, m):
        """ Calculates the determinant of a representation of a given group element.

        @param m: matrix representation of a group element.
        """
        raise NotImplementedError()

    def left_action_on_Rd(self, g, x):
        """ Group action of an element from the subgroup H on a vector in Rd.

        @param g: Group element.
        @param x: Vector in Rd.
        """
        raise NotImplementedError()

    def left_action_on_H(self, g, x):
        """ Group action of an element from the subgroup H on an element in H.

        @param g: Group element
        @param x: Other group element
        """
        raise NotImplementedError()

    def representation(self, g):
        raise NotImplementedError()

    def normalize(self, g):
        """ Normalize values of group elements to range between -1 and 1 for CKNet

        :param g: group element
        """
        raise NotImplementedError()
