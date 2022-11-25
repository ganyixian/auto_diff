#!/usr/bin/env python3
# Project    : AutoDiff
# File       : expression.py
# Description: autodiff functional expressions
# Copyright 2022 Harvard University. All Rights Reserved.
import numpy as np

from dual import Dual, DualVector
import ops


def _generate_base(inputs):
    assert isinstance(inputs, dict)

    zero_vec = {k: 0 if type(v) in [int, float] else np.zeros_like(v) for k, v in inputs.items()}
    for k, v in inputs.items():
        if type(v) in (int, float):
            zero_vec[k] = 1
            yield zero_vec, k
            zero_vec[k] = 0
        else:
            for i in range(len(v)):
                zero_vec[k][i] = 1
                yield zero_vec, k
                zero_vec[k][i] = 0


class Compose:
    def __init__(self, flist=[]):
        self.funcs = flist
        assert all([isinstance(f, Expression) for f in flist]), 'Illegal argument. Compose can only compose Expressions.'

    def __call__(self, inputs, seed=None, **kwargs):
        if seed is not None:
            res = [f(inputs, seed, keep_graph=True) for f in self.funcs]
        else:
            res_dict = {k: [] for k in inputs.keys()}
            for sd, k in _generate_base(inputs):
                res_dict[k].append([f(inputs, sd, keep_graph=True) for f in self.funcs])
                self.clear()
            res = self.merge(res_dict)
        return res

    def merge(self, res_dict):
        res_list = []
        for X, dFdX in res_dict.items():
            for dFdx in dFdX:
                for i, dfdx in enumerate(dFdx):
                    if len(res_list) <= i:
                        res_list.append((dfdx[0], {k: [] for k in res_dict.keys()}))

                    res_list[i][1][X].append(dfdx[1])
        return res_list

    def clear(self):
        for f in self.funcs:
            f.clear()

    def __len__(self):
        return len(self.funcs)

    def __iter__(self):
        for f in self.funcs:
            yield f

    def __str__(self):
        return str(self.funcs)


class Expression:
    def __init__(self, mode='f', name=None):
        self.val = None
        self.mode = mode
        self.varname = set()

    def __call__(self, inputs, seed=None, keep_graph=False):
        if isinstance(inputs, (float, int)):
            inputs = {k: inputs for k in self.varname}

        if self.mode == 'f':
            if seed:
                if isinstance(seed, (float, int)):
                    seed = {k: seed for k in self.varname}

                res = self.forward(inputs, seed)

                y = [v.get_real() for v in res]
                dy = [v.get_dual() for v in res]

                if not keep_graph:
                    self.clear()

                return y, dy
            else:
                res = {k: [] for k in inputs.keys()}
                # zero_vec = {k: 0 for k in inputs.keys()}
                # for k in zero_vec.keys():
                #     zero_vec[k] = 1
                #     res[k] = self.forward(inputs, zero_vec)
                #     zero_vec[k] = 0
                #     self.clear()

                for sd, k in _generate_base(inputs):
                    res[k].append(self.forward(inputs, sd))
                    self.clear()

                k, val = res.popitem()
                y = [v.get_real() for v in val]
                dy = {k: [v.get_dual() for v in val]}
                for k, val in res.items():
                    dy[k] = [v.get_dual() for v in val]
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
        assert isinstance(x, Expression)
        return Function(x, f=ops._sin)

    @staticmethod
    def cos(x):
        assert isinstance(x, Expression)
        return Function(x, f=ops._cos)

    @staticmethod
    def tan(x):
        assert isinstance(x, Expression)
        return Function(x, f=ops._tan)

    @staticmethod
    def exp(x):
        assert isinstance(x, Expression)
        return Function(x, f=ops._exp)

    @staticmethod
    def log(x):
        assert isinstance(x, Expression)
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

    def __str__(self):
        return f"Function object, function of {self.varname}"


class Variable(Expression):

    def __init__(self, name, mode='f'):
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
            inputs, seed = inputs.get(self.name, 0), seed.get(self.name, 0)
        if type(inputs) in [list, np.ndarray] and type(seed) in [list, np.ndarray]:
            self.val = DualVector(inputs, seed)
        elif type(inputs) in [int, float] and type(seed) in [int, float]:
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

    def __str__(self):
        return f"Variable object, name {self.name}"


if __name__ == '__main__':
    # tup = (1, {'x':1})
    # tup[0] = 2
    # print(tup)
    # exit()
    x, y = Variable.vars(['x', 'y'])
    # f = Compose([x**2+x*y, x-y])
    f = x ** 2 + x* y - Expression.sin(x)
    # f = Expression.sin(f)
    inputs = {'x': [1,2], 'y':2}
    seed = {'x': [0,0,0,1], 'y':0}
    # a = DualVector([1,1,1], [2,2,2])
    # print(type(a) == Dual)
    # exit()
    print(f(inputs))
    # v1 = [Dual(1,0), Dual(2,1), Dual(1,2),Dual(-1,1)]
    # v2 = [Dual(0,1), Dual(1,2), Dual(1,2),Dual(-1,1)]
    # print(v1 * v2)