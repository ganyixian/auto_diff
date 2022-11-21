#!/usr/bin/env python3
# Project    : AutoDiff
# File       : expression.py
# Description: autodiff functional expressions
# Copyright 2022 Harvard University. All Rights Reserved.

from .expression import Function


class Expression:
    def __init__(self, mode='f', name=None):
        self.val = None
        self.mode = mode
        self.varname = set()

    def __call__(self, inputs, seed=None):
        if isinstance(inputs, (float, int)):
            inputs = {k: inputs for k in self.varname}

        if self.mode == 'f':
            if seed:
                if isinstance(seed, (float, int)):
                    seed = {k: seed for k in self.varname}

                res = self.forward(inputs, seed)
                if isinstance(res, (int, float)):
                    res = [res]

                y = [v.real for v in res]
                dy = [v.dual for v in res]
                self.clear()
                return y, dy
            else:
                res = {}
                zero_vec = {k: 0 for k in inputs.keys()}
                for k in zero_vec.keys():
                    zero_vec[k] = 1
                    res[k] = self.forward(inputs, zero_vec)
                    zero_vec[k] = 0
                    self.clear()

                k, val = res.popitem()
                y = [v.real for v in val]
                dy = {f"df/d{k}": [v.dual for v in val]}
                for k, val in res.items():
                    dy[f"df/d{k}"] = [v.dual for v in val]
                return y, dy

        else:
            # TODO: add backward mode
            raise NotImplementedError('AutoDiff only support forward mode for now')

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

    def forward(self, inputs, seed):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError