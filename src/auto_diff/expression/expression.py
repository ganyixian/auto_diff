#!/usr/bin/env python3
# Project    : AutoDiff
# File       : expression.py
# Description: autodiff functional expressions
# Copyright 2022 Harvard University. All Rights Reserved.

# from . import Function


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
        raise NotImplementedError

    def __mul__(self, other):
        raise NotImplementedError

    __radd__ = __add__
    __rmul__ = __mul__

    def __sub__(self, other):
        raise NotImplementedError

    def __rsub__(self, other):
        raise NotImplementedError

    def __truediv__(self, other):
        raise NotImplementedError

    def __rdiv__(self, other):
        raise NotImplementedError

    def __pow__(self, power, modulo=None):
        raise NotImplementedError

    def __neg__(self):
        raise NotImplementedError

    def forward(self, inputs, seed):
        raise NotImplementedError

    def clear(self):
        raise NotImplementedError