#!/usr/bin/env python3
# Project    : AutoDiff
# File       : variable.py
# Description: autodiff variable class 
# Copyright 2022 Harvard University. All Rights Reserved.

from ..dual import Dual
from .function import Function

class Variable(Function):

    def __init__(self, name, val=None, mode='f'):
        super(Variable, self).__init__(mode=mode, name=name)
        self.name = name
        self.varname.add(name)

    @classmethod
    def vars(cls, varlist=[]):
        assert isinstance(varlist, (list, tuple)), 'Please provide a list of variable names.'
        return [cls(v) for v in varlist]

    def forward(self, inputs, seed):
        if self.val:
            return self.val

        assert seed is not None, 'Please provide a seed vector'

        if type(inputs) == dict and type(seed) == dict:
            self.val = Dual(inputs.get(self.name, 0), seed.get(self.name, 0))
        elif type(inputs) == int and type(seed) == int:
            self.val = Dual(inputs, seed)
        else:
            raise ValueError(
                f"Unsupported type {type(inputs)} for variable inputs and type {type(seed)} for seed vector")

        return self.val

    def clear(self):
        self.val = None