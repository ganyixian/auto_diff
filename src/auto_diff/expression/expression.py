#!/usr/bin/env python3
# Project    : AutoDiff
# File       : expression.py
# Description: autodiff functional expressions
# Copyright 2022 Harvard University. All Rights Reserved.
import numpy as np

from ..dual import Dual
from .ops import ops

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

    def __eq__(self, other) -> bool:
        return isinstance(other, Expression) and self.val == other.val and \
            self.mode == other.mode and self.val == other.val and \
            self.varname == other.varname

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
        return Function(self, f=(lambda x: other - x))

    def __truediv__(self, other):
        if isinstance(other, Expression):
            return Function(self, other, (lambda x, y: x / y))
        return Function(self, f=(lambda x: x / other))

    def __rtruediv__(self, other):
        return Function(self, f=(lambda x: other / x))

    def __pow__(self, power, modulo=None):
        if isinstance(power, Expression):
            return Function(self, power, (lambda a, x: a ** x))
        return Function(self, f=(lambda a: a ** power))

    def __neg__(self):
        return Function(self, f=(lambda x: -x))

    @staticmethod
    def sin(x):
        if isinstance(x, (int, float)):
            return np.sin(x)
        return Function(x, f=ops._sin)

    @staticmethod
    def cos(x):
        if isinstance(x, (int, float)):
            return np.cos(x)
        return Function(x, f=ops._cos)

    @staticmethod
    def tan(x):
        if isinstance(x, (int, float)):
            return np.tan(x)
        return Function(x, f=ops._tan)

    @staticmethod
    def exp(x):
        if isinstance(x, (int, float)):
            return np.exp(x)
        return Function(x, f=ops._exp)

    @staticmethod
    def log(x):
        if isinstance(x, (int, float)):
            return np.log(x)
        return Function(x, f=ops._log)


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

    def __eq__(self, other):
        if type(other) != Function:
            return False
        e_same = self.e1 == other.e1 and self.e1 == other.e1
        if self.f or other.f:
            func_same = self.f.__code__.co_code == other.f.__code__.co_code
        else: 
            func_same = True
        expression_eq = super().__eq__(other)

        return e_same and func_same and expression_eq

class Variable(Expression):

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
    
    def __eq__(self, other):
        if type(other) != Variable:
            return False
        
        return super().__eq__(other) and self.name == other.name