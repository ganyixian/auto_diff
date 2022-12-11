#!/usr/bin/env python3
# Project    : AutoDiff
# File       : node.py
# Description: Computational graph node for reverse mode autodiff
# Copyright 2022 Harvard University. All Rights Reserved.
import numpy as np

class Node:
    """A class represents one single node in the computational graph. In reverse mode, 
    every object of Expression class would be paired up with one Node object, which stores 
    the parents and children nodes of the current node. 
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
        """Update the partial value based on the input and partial derivative function.

        :param args: input list of the current node.
        """
        for ddp in self.partial_func:
            self.partial_val.append(ddp(*args))

    def notify(self, id, val):
        """Function to notify the partent after finishing computing.

        :param id: child id
        :param val: computed value by child
        """
        assert id in self.child, 'Informed by unknown child'
        if self.adjoint is None:
            self.adjoint = val
        else:
            self.adjoint = self.adjoint + val
        self.received.add(id)

    def compute(self):
        """Aggregate the results to get the adjoint of the current node and return the adjoint. 

        :return: boolean of whether the compute succeded.
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
        """Clear the calculated result of the node.
        """
        self.partial_val = []
        self.received = set()
        self.adjoint = None
        self.child = []





