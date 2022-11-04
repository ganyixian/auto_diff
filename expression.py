#!/usr/bin/env python3
# Project    : AutoDiff
# File       : expression.py
# Description: autodiff functional expressions
# Copyright 2022 Harvard University. All Rights Reserved.

import numpy as np

import dual
import ops


class Expression:
    def __init__(self, mode='f'):
        self.val = None
        self.mode = mode

    def __call__(self, inputs, seed=None):
        if self.mode == 'f':
            if seed:
                res = self.forward(inputs, seed)

            else:
                pass

            y = [v.real for v in res]
            dy = [v.dual for v in res]
            return y, dy

        else:
            # TODO: add backward mode
            raise NotImplementedError('AutoDiff only support forward mode for now')

    def __add__(self, other):
        return Function(self, other, (lambda x, y: x + y))

    def __mul__(self, other):
        return Function(self, other, (lambda x, y: x * y))

    __radd__ = __add__
    __rmul__ = __mul__

    def __sub__(self, other):
        return Function(self, other, (lambda x, y: x - y))

    def __rsub__(self, other):
        return Function(other, self, (lambda x, y: x - y))

    def __truediv__(self, other):
        return Function(self, other, (lambda x, y: x / y))

    def __rdiv__(self, other):
        return Function(other, self, (lambda x, y: x / y))

    def __pow__(self, power, modulo=None):
        return Function(self, power, (lambda a, x: a ** x))

    def __neg__(self):
        return Function(self, f=(lambda x: -x))

    def forward(self, inputs, seed):
        raise NotImplementedError


class Variable(Expression):

    def __init__(self, name, val=None, mode='f'):
        super(Variable, self).__init__(mode=mode)
        self.name = name

    def forward(self, inputs, seed):
        if self.val:
            return self.val

        assert seed is not None, 'Please provide a seed vector'

        if type(inputs) == dict and type(seed) == dict:
            self.val = dual.Dual(inputs.get(self.name, 0), seed.get(self.name, 0))
        elif type(inputs) == int and type(seed) == int:
            self.val = dual.Dual(inputs, seed)
        else:
            raise ValueError(
                f"Unsupported type {type(inputs)} for variable inputs and type {type(seed)} for seed vector")

        return self.val


class Function(Expression):

    def __init__(self, e1, e2=None, f=None, mode='f'):
        super(Function, self).__init__(mode=mode)
        self.e1 = e1
        self.e2 = e2
        self.f = f

    def forward(self, inputs, seed):
        if self.val:
            return self.val
        
        res1 = self.e1.forward(inputs, seed)
        if self.e2:
            res2 = self.e2.forward(inputs, seed)
            return self.f(res1, res2)
        else:
            return self.f(res1)

