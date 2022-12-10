#!/usr/bin/env python3
# Project    : AutoDiff
# File       : expression.py
# Description: autodiff functional expressions
# Copyright 2022 Harvard University. All Rights Reserved.
import numpy as np

from ..dual import Dual, DualVector
from . import ops
from .node import Node


def _generate_base(inputs):
    """

    :param inputs:
    :return:
    """
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
    """

    """
    def __init__(self, flist=[]):
        self.funcs = flist
        self.mode = flist[0].mode
        assert all(
            [isinstance(f, Expression) for f in flist]), 'Illegal argument. Compose can only compose Expressions.'

    def __call__(self, inputs, seed=None, **kwargs):
        if self.mode == 'f':
            if seed is not None:
                res = [f(inputs, seed, keep_graph=True) for f in self.funcs]
            else:
                res_dict = {k: [] for k in inputs.keys()}
                for sd, k in _generate_base(inputs):
                    res_dict[k].append([f(inputs, sd, keep_graph=True) for f in self.funcs])
                    self.clear()
                res = self.merge(res_dict)
            return res
        else:
            return [f(inputs, seed) for f in self.funcs]

    def merge(self, res_dict):
        """

        :param res_dict:
        :return:
        """
        res_list = []
        for X, dFdX in res_dict.items():
            for dFdx in dFdX:
                for i, dfdx in enumerate(dFdx):
                    if len(res_list) <= i:
                        res_list.append((dfdx[0], {k: [] for k in res_dict.keys()}))

                    res_list[i][1][X].append(dfdx[1])
        return res_list

    def clear(self):
        """

        :return:
        """
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
    """

    """
    def __init__(self, mode='f', name=None):
        self.val = None
        self.mode = mode
        self.varname = set()

    def __call__(self, inputs, seed=None, keep_graph=False):
        if isinstance(inputs, (float, int)):
            inputs = {k: inputs for k in self.varname}

        if self.mode == 'f':
            print(f'Now in Forward mode!')
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
            y = self.propagate(inputs)
            grad = self.backward()
            if not keep_graph:
                self.clear()
            return y, grad

    def __eq__(self, other) -> bool:
        return isinstance(other, Expression) and self.val == other.val and \
               self.mode == other.mode and self.val == other.val and \
               self.varname == other.varname

    def __add__(self, other):
        if isinstance(other, Expression):
            return Function(self, other, (lambda x, y: x + y), self.mode,
                            Node([self.node, other.node], [(lambda x, y: 1), (lambda x, y: 1)]))

        return Function(self, f=(lambda x: x + other), mode=self.mode, node=Node([self.node], [(lambda x: 1)]))

    def __mul__(self, other):
        if isinstance(other, Expression):
            return Function(self, other, (lambda x, y: x * y), self.mode,
                            Node([self.node, other.node], [(lambda x, y: y), (lambda x, y: x)]))
        return Function(self, f=(lambda x: x * other), mode=self.mode, node=Node([self.node], [(lambda x: other)]))

    __radd__ = __add__
    __rmul__ = __mul__

    def __sub__(self, other):
        if isinstance(other, Expression):
            return Function(self, other, (lambda x, y: x - y), self.mode,
                            Node([self.node, other.node], [(lambda x, y: 1), (lambda x, y: -1)]))
        return Function(self, f=(lambda x: x - other), mode=self.mode, node=Node([self.node], [(lambda x: 1)]))

    def __rsub__(self, other):
        return Function(self, f=(lambda x: other - x), mode=self.mode, node=Node([self.node], [(lambda x: -1)]))

    def __truediv__(self, other):
        if isinstance(other, Expression):
            return Function(self, other, (lambda x, y: x / y), self.mode,
                            Node([self.node, other.node], [(lambda x, y: 1 / y), (lambda x, y: -x / y ** 2)]))
        return Function(self, f=(lambda x: x / other), mode=self.mode, node=Node([self.node], [(lambda x: x / other)]))

    def __rtruediv__(self, other):
        return Function(self, f=(lambda x: other / x), mode=self.mode,
                        node=Node([self.node], [(lambda x: -other / x ** 2)]))

    def __pow__(self, power, modulo=None):
        if isinstance(power, Expression):
            return Function(self, power, (lambda a, x: a ** x), self.mode,
                            Node([self.node, power.node],
                                 [(lambda x, y: y * x ** (y - 1)), (lambda x, y: x ** y * np.log(x))]))
        return Function(self, f=(lambda a: a ** power), mode=self.mode,
                        node=Node([self.node], [(lambda x: power * x ** (power - 1))]))
    
    def __rpow__(self, other, modulo=None):
        if isinstance(other, Expression):
            return other.__power__(self)
        return Function(self, f=(lambda a: other ** a), mode=self.mode,
                        node=Node([self.node], [(lambda x: other ** x * np.log(other))]))

    def __neg__(self):
        return Function(self, f=(lambda x: -x), mode=self.mode, node=Node([self.node], [(lambda x: -1)]))

    @staticmethod
    def sin(x):
        """
        Create a Function object for sine operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._sin, mode=x.mode, node=Node([x.node], [(lambda x: np.cos(x))]))

    @staticmethod
    def cos(x):
        """
        Create a Function object for cosine operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._cos, mode=x.mode, node=Node([x.node], [(lambda x: -np.sin(x))]))

    @staticmethod
    def tan(x):
        """
        Create a Function object for tangent operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._tan, mode=x.mode, node=Node([x.node], [(lambda x: 1 / np.cos(x) ** 2)]))

    # new implemented inverse trig functions.
    @staticmethod
    def arcsin(x):
        assert isinstance(x, Expression)
        return Function(x, f=ops._arcsin, mode=x.mode, node=Node([x.node], [(lambda x: 1 / (1 - x * x) ** 0.5)]))

    @staticmethod
    def arccos(x):
        assert isinstance(x, Expression)
        return Function(x, f=ops._arccos, mode=x.mode, node=Node([x.node], [(lambda x: -1 / (1 - x * x) ** 0.5)]))

    @staticmethod
    def arctan(x):
        assert isinstance(x, Expression)
        return Function(x, f=ops._arctan, mode=x.mode, node=Node([x.node], [(lambda x: 1 / (1 + x * x))]))

    # new implemented hyperbolic functions.
    @staticmethod
    def sinh(x):
        assert isinstance(x, Expression)
        return Function(x, f=ops._sinh, mode=x.mode, node=Node([x.node], [(lambda x: (np.cosh(x)))]))

    @staticmethod
    def cosh(x):
        assert isinstance(x, Expression)
        return Function(x, f=ops._cosh, mode=x.mode, node=Node([x.node], [(lambda x: (np.sinh(x)))]))

    @staticmethod
    def tanh(x):
        assert isinstance(x, Expression)
        return Function(x, f=ops._tanh, mode=x.mode, node=Node([x.node], [(lambda x: 1 - (np.tanh(x)) ** 2)]))

    # new implemented standard logistic function.
    @staticmethod
    def sigmoid(x):
        assert isinstance(x, Expression)
        sig = 1 / (1 + Expression.exp(-x))
        return Function(x, f=ops._sigmoid, mode=x.mode, node=Node([x.node], [(lambda x: sig * (1 - sig))]))


    @staticmethod
    def exp(x):
        """
        Create a Function object for exponential operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._exp, mode=x.mode, node=Node([x.node], [(lambda x: np.exp(x))]))

    @staticmethod
    def log(x):
        """
        Create a Function object for natural logarithmic operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._log, mode=x.mode, node=Node([x.node], [(lambda x: 1 / x)]))

    @staticmethod
    def log_base(x, base):
        assert isinstance(x, Expression)
        return Function(x, f= (lambda x : ops._log_base(x, base)), mode=x.mode, node=Node([x.node], [(lambda x: 1 / (x * np.log(base)))]))

    @staticmethod
    def sqrt(x):
        assert isinstance(x, Expression)
        return Function(x, f=ops._sqrt, mode=x.mode, node=Node([x.node], [(lambda x: 0.5 * (x ** -0.5))]))


class Function(Expression):
    """

    """

    def __init__(self, e1, e2=None, f=None, mode='f', node=None):
        super(Function, self).__init__(mode=mode)
        self.e1 = e1
        self.e2 = e2
        self.f = f
        self.varname = e1.varname
        self.node = node
        if e2:
            self.varname.update(e2.varname)

    def forward(self, inputs, seed):
        """

        :param inputs:
        :param seed:
        :return:
        """
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
        """

        :return:
        """
        self.e1.clear()
        if self.e2:
            self.e2.clear()
        self.val = None
        self.node.clear()

    def propagate(self, inputs, child=None):
        """

        :param inputs:
        :param child:
        :return:
        """
        if child is not None:
            self.node.child.append(child)

        if self.val is not None:
            return self.val

        args = [self.e1.propagate(inputs, self.node.id)]
        if self.e2 is not None:
            args.append(self.e2.propagate(inputs, self.node.id))
        self.val = self.f(*args)
        self.node.update(*args)

        return self.val

    def backward(self):
        """

        :return:
        """
        res = {}
        if self.node.compute():
            res = res | (self.e1.backward())
            if self.e2 is not None:
                res = res | self.e2.backward()
        return res

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
    """

    """

    def __init__(self, name, mode='f'):
        super(Variable, self).__init__(mode=mode, name=name)
        self.name = name
        self.varname.add(name)
        self.node = Node()

    @classmethod
    def vars(cls, varlist=[], mode='f'):
        """

        :param varlist:
        :param mode:
        :return:
        """
        assert isinstance(varlist, (list, tuple)), 'Please provide a list of variable names.'
        return [cls(v, mode) for v in varlist]

    def forward(self, inputs, seed):
        """

        :param inputs:
        :param seed:
        :return:
        """
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
        """

        :return:
        """
        self.val = None
        self.node.clear()

    def propagate(self, inputs, child=None):
        """

        :param inputs:
        :param child:
        :return:
        """
        if child is not None:
            self.node.child.append(child)

        if self.val is not None:
            return self.val

        if type(inputs) == dict:
            self.val = inputs.get(self.name, 0)
        elif type(inputs) in [list, np.ndarray]:
            self.val = np.array(inputs)
        elif type(inputs) in [int, float]:
            self.val = inputs
        else:
            raise ValueError(
                f"Unsupported type {type(inputs)} for variable inputs.")

        return self.val

    def backward(self):
        """

        :return:
        """
        if self.node.compute():
            return {self.name: self.node.adjoint}
        else:
            return {}

    def __eq__(self, other):
        if type(other) != Variable:
            return False

        return super().__eq__(other) and self.name == other.name

    def __str__(self):
        return f"Variable object, name {self.name}"


if __name__ == '__main__':
    x, y = Variable('x', mode='r'), Variable('y', mode='r')
    f = Expression.sin(x * 4) + Expression.cos(y * 4)  # create a function f = sin(4x) + cos(4y)
    f = f + Expression.exp(x * y)  # f = sin(4x) + cos(4y) + e^(xy)
    f_val, f_deriv = f({'x': 1, 'y': 2})
    # return the value of f and the derivative in the direction of seed at x = 1, y = 1.
    print(f_val, f_deriv)
    # a = DualVector([1,1,1], [2,2,2])
    # print(type(a) == Dual)
    # exit()
    # print(f(inputs))
    # v1 = [Dual(1,0), Dual(2,1), Dual(1,2),Dual(-1,1)]
    # v2 = [Dual(0,1), Dual(1,2), Dual(1,2),Dual(-1,1)]
    # print(v1 * v2)
