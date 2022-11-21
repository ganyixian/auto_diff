#!/usr/bin/env python3
# Project    : AutoDiff
# File       : expression.py
# Description: autodiff functional expressions
# Copyright 2022 Harvard University. All Rights Reserved.

import numpy as np

from .ops import ops
from .expression import Expression

class Function(Expression):

    def __init__(self, e1, e2=None, f=None, mode='f'):
        super(Function, self).__init__(mode=mode)
        self.e1 = e1
        self.e2 = e2
        self.f = f
        self.varname = e1.varname
        if e2:
            self.varname.update(e2.varname)

    def forward(self, inputs, seed):
        if self.val:
            return self.val

        res1 = self.e1.forward(inputs, seed)
        if self.e2:
            res2 = self.e2.forward(inputs, seed)
            self.val = self.f(res1, res2)
        else:
            self.val = self.f(res1)

        return self.val

    def clear(self):
        self.e1.clear()
        if self.e2:
            self.e2.clear()
        self.val = None

    def __add__(self, other):
        if isinstance(other, Expression):
            return Function(self, other, (lambda x, y: x + y))

        return Function(self, f=(lambda x: x + other))

    def __mul__(self, other):
        if isinstance(other, Expression):
            return Function(self, other, (lambda x, y: x * y))
        return Function(self, f=(lambda x: x * other))

    __radd__ = __add__
    __rmul__ = __mul__

    def __sub__(self, other):
        if isinstance(other, Expression):
            return Function(self, other, (lambda x, y: x - y))
        return Function(self, f=(lambda x: x - other))

    def __rsub__(self, other):
        if isinstance(other, Expression):
            return Function(other, self, (lambda x, y: x - y))
        return Function(self, f=(lambda x: other - x))

    def __truediv__(self, other):
        if isinstance(other, Expression):
            return Function(self, other, (lambda x, y: x / y))
        return Function(self, f=(lambda x: x / other))

    def __rdiv__(self, other):
        if isinstance(other, Expression):
            return Function(other, self, (lambda x, y: x / y))
        return Function(self, f=(lambda x: other / x))

    def __pow__(self, power, modulo=None):
        if isinstance(power, Expression):
            return Function(self, power, (lambda a, x: a ** x))
        return Function(self, f=(lambda a: a ** power))

    def __neg__(self):
        return Function(self, f=(lambda x: -x))

    def sin(x):
        if isinstance(x, (int, float)):
            return np.sin(x)
        return Function(x, f=ops._sin)


    def cos(x):
        if isinstance(x, (int, float)):
            return np.cos(x)
        return Function(x, f=ops._cos)


    def tan(x):
        if isinstance(x, (int, float)):
            return np.tan(x)
        return Function(x, f=ops._tan)


    def exp(x):
        if isinstance(x, (int, float)):
            return np.exp(x)
        return Function(x, f=ops._exp)


    def log(x):
        if isinstance(x, (int, float)):
            return np.log(x)
        return Function(x, f=ops._log)