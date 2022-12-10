#!/usr/bin/env python3
# Project    : AutoDiff
# File       : node.py
# Description: Computational graph node for reverse mode autodiff
# Copyright 2022 Harvard University. All Rights Reserved.
import numpy as np

# from . import ops


class Node:
    """

    """
    num_node = 0

    def __init__(self, p=[], ddp=[]):
        self.id = Node.num_node
        Node.num_node += 1
        self.parent = p
        self.partial_func = ddp #DF/DX
        # EX. F=X**2 ddp=lambda x: 2x (function)
        # x= 2, val = ddp(2) = 4
        self.partial_val = []
        self.child = []
        self.received = set()
        self.adjoint = None

    def update(self, *args):
        """

        :param args:
        :return:
        """
        for ddp in self.partial_func:
            self.partial_val.append(ddp(*args))

    def notify(self, id, val):
        """

        :param id:
        :param val:
        :return:
        """
        assert id in self.child, 'Informed by unknown child'
        if self.adjoint is None:
            self.adjoint = val
        else:
            self.adjoint += val
        self.received.add(id)

    def compute(self):
        """

        :return:
        """
        if len(self.received) == len(self.child):
            if self.adjoint is None:
                self.adjoint =np.ones_like(self.partial_val[0])
            for p, dp in zip(self.parent, self.partial_val):
                p.notify(self.id, dp * self.adjoint)
            return True
        else:
            return False

    def clear(self):
        """

        :return:
        """
        self.partial_val = []
        self.received = set()
        self.adjoint = None
        self.child = []





