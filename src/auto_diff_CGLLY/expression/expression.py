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
    """Function to generate zero vector for forward evaluation process

    :param inputs: Dictionary input
    :return: A list of int, float, or list.
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
    """A wrapper class that achieves evaluating mutiple functions together.
    """
    def __init__(self, flist=[]):
        """Initialize the Compose object by the input function list
        
        :param flist: Function list
        """
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
        """Merge the evaluation results from a dictionary to a single list.

        :param res_dict: Result Dictonary
        :return: Result list
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
        """Clear the input and other variable saved in the functions.
        """
        for f in self.funcs:
            f.clear()

    def __len__(self):
        return len(self.funcs)

    def __iter__(self):
        for f in self.funcs:
            yield f

    def __str__(self):
        return '\n'.join([str(func) for func in self.funcs])


class Expression:
    """Base class for Function and Variable. Defines the underlying common functions
    shared by both classes.
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
        """
        This allows for addition with Expression instances or scalar numbers. 
        :param other: Expression instance or scalar number
        :return: Function
        """
        if isinstance(other, Expression):
            return Function(self, other, (lambda x, y: x + y), self.mode,
                            Node([self.node, other.node], [(lambda x, y: 1), (lambda x, y: 1)]))

        return Function(self, f=(lambda x: x + other), mode=self.mode, node=Node([self.node], [(lambda x: 1)]))

    def __mul__(self, other):
        """
        This allows for multiplication with Expression instances or scalar numbers. 
        :param other: Expression instance or scalar number
        :return: Function
        """
        if isinstance(other, Expression):
            return Function(self, other, (lambda x, y: x * y), self.mode,
                            Node([self.node, other.node], [(lambda x, y: y), (lambda x, y: x)]))
        return Function(self, f=(lambda x: x * other), mode=self.mode, node=Node([self.node], [(lambda x: other)]))

    __radd__ = __add__
    __rmul__ = __mul__

    def __sub__(self, other):
        """
        This allows for substraction with Expression instances or scalar numbers. 
        :param other: Expression instance or scalar number
        :return: Function
        """

        if isinstance(other, Expression):
            return Function(self, other, (lambda x, y: x - y), self.mode,
                            Node([self.node, other.node], [(lambda x, y: 1), (lambda x, y: -1)]))
        return Function(self, f=(lambda x: x - other), mode=self.mode, node=Node([self.node], [(lambda x: 1)]))

    def __rsub__(self, other):
        """
        This is called when scalar number - Expression 
        :param other: scalar number
        :return: Function
        """
        return Function(self, f=(lambda x: other - x), mode=self.mode, node=Node([self.node], [(lambda x: -1)]))

    def __truediv__(self, other):
        """
        This allows for true division between Expression instances or scalar numbers. 
        :param other: Expression instance or scalar number
        :return: Function
        """
        if isinstance(other, Expression):
            return Function(self, other, (lambda x, y: x / y), self.mode,
                            Node([self.node, other.node], [(lambda x, y: 1 / y), (lambda x, y: -x / y ** 2)]))
        return Function(self, f=(lambda x: x / other), mode=self.mode, node=Node([self.node], [(lambda x: x / other)]))

    def __rtruediv__(self, other):
        """
        This is called when scalar number / Expression 
        :param other: scalar number
        :return: Function
        """
        return Function(self, f=(lambda x: other / x), mode=self.mode,
                        node=Node([self.node], [(lambda x: -other / x ** 2)]))

    def __pow__(self, power, modulo=None):
        """
        This allows for power operation between Expression instances or scalar numbers. 
        :param other: Expression instance or scalar number
        :return: Function
        """
        
        if isinstance(power, Expression):
            return Function(self, power, (lambda a, x: a ** x), self.mode,
                            Node([self.node, power.node],
                                 [(lambda x, y: y * x ** (y - 1)), (lambda x, y: x ** y * np.log(x))]))
        return Function(self, f=(lambda a: a ** power), mode=self.mode,
                        node=Node([self.node], [(lambda x: power * x ** (power - 1))]))

    def __rpow__(self, other, modulo=None):
        """
        This is called when scalar number ** Expression 
        :param other: scalar number
        :return: Function
        """
        return Function(self, f=(lambda a: other ** a), mode=self.mode,
                        node=Node([self.node], [(lambda x: other ** x * np.log(other))]))

    def __neg__(self):
        """
        This allows for negation of an Expression instance.
        :return: Function
        """
        return Function(self, f=(lambda x: -x), mode=self.mode, node=Node([self.node], [(lambda x: -1)]))

    @staticmethod
    def sin(x):
        """Create a Function object for sine operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._sin, mode=x.mode, node=Node([x.node], [(lambda x: np.cos(x))]))

    @staticmethod
    def cos(x):
        """Create a Function object for cosine operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._cos, mode=x.mode, node=Node([x.node], [(lambda x: -np.sin(x))]))

    @staticmethod
    def tan(x):
        """Create a Function object for tangent operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._tan, mode=x.mode, node=Node([x.node], [(lambda x: 1 / np.cos(x) ** 2)]))

    @staticmethod
    def arcsin(x):
        """Create a Function object for inverse of sine operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._arcsin, mode=x.mode, node=Node([x.node], [(lambda x: 1 / (1 - x * x) ** 0.5)]))

    @staticmethod
    def arccos(x):
        """Create a Function object for inverse of cosine operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._arccos, mode=x.mode, node=Node([x.node], [(lambda x: -1 / (1 - x * x) ** 0.5)]))

    @staticmethod
    def arctan(x):
        """Create a Function object for inverse of tangent operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._arctan, mode=x.mode, node=Node([x.node], [(lambda x: 1 / (1 + x * x))]))

    @staticmethod
    def sinh(x):
        """Create a Function object for hyperbolic sine operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._sinh, mode=x.mode, node=Node([x.node], [(lambda x: (np.cosh(x)))]))

    @staticmethod
    def cosh(x):
        """Create a Function object for hyperbolic cosine operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._cosh, mode=x.mode, node=Node([x.node], [(lambda x: (np.sinh(x)))]))

    @staticmethod
    def tanh(x):
        """Create a Function object for hyperbolic tangent operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._tanh, mode=x.mode, node=Node([x.node], [(lambda x: 1 - (np.tanh(x)) ** 2)]))

    @staticmethod
    def sigmoid(x):
        """Create a Function object for sigmoid operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        sig = 1 / (1 + Expression.exp(-x))
        return Function(x, f=ops._sigmoid, mode=x.mode, node=Node([x.node], [(lambda x: sig * (1 - sig))]))

    @staticmethod
    def exp(x):
        """Create a Function object for exponential operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._exp, mode=x.mode, node=Node([x.node], [(lambda x: np.exp(x))]))

    @staticmethod
    def log(x):
        """Create a Function object for natural logarithmic operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._log, mode=x.mode, node=Node([x.node], [(lambda x: 1 / x)]))

    @staticmethod
    def log_base(x, base):
        """Create a Function object for the logarithm operation with a chosen base

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f= (lambda x : ops._log_base(x, base)), mode=x.mode, node=Node([x.node], [(lambda x: 1 / (x * np.log(base)))]))

    @staticmethod
    def sqrt(x):
        """Create a Function object for square root operation of input

        :param x: Expression
        :return: Function
        """
        assert isinstance(x, Expression)
        return Function(x, f=ops._sqrt, mode=x.mode, node=Node([x.node], [(lambda x: 0.5 * (x ** -0.5))]))


class Function(Expression):
    """A class represents a mathamatical function. The function class supports building functions from variables 
    and evaluating in both forward mode and backward mode.
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
        """Forward mode differentiation for a Function.

        :param inputs: dictionary
        :param seed: dictionary
        :return: Dual or DualVector
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
        """Clear the previous evaluation result, and clears the expressions and node.
        """
        self.e1.clear()
        if self.e2:
            self.e2.clear()
        self.val = None
        if self.node:
            self.node.clear()

    def propagate(self, inputs, child=None):
        """Forward pass of the reverse mode differentiation.
        
        :param inputs: input dictionary
        :param child: node
        :return: evaluation result -> int, float, or np.array 
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
        """Backward pass of the reverse mode differentiation.
        :return: derivative result -> int, float, or np.array 
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
    """A class represents a mathematical variable.
    """

    def __init__(self, name, mode='f'):
        super(Variable, self).__init__(mode=mode, name=name)
        self.name = name
        self.varname.add(name)
        self.node = Node()

    @classmethod
    def vars(cls, varlist=[], mode='f'):
        """Create a list of function with in put name and mode

        :param varlist: variable name list
        :param mode: variable mode
        :return: list of variable
        """
        assert isinstance(varlist, (list, tuple)), 'Please provide a list of variable names.'
        return [cls(v, mode) for v in varlist]

    def forward(self, inputs, seed):
        """Forward mode differentiation for a variable.

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
        """Clear the previous differentiation results.
        """
        self.val = None
        if self.node:
            self.node.clear()

    def propagate(self, inputs, child=None):
        """Forward pass of the reverse mode differentiation for a variable.
        
        :param inputs: input dictionary
        :param child: node
        :return: evaluation result -> int, float, or np.array 
        """
        if child is not None:
            self.node.child.append(child)

        if self.val is not None:
            return self.val

        if type(inputs) == dict:
            inputs = inputs.get(self.name, 0)
        
        if type(inputs) in [list, np.ndarray]:
            self.val = np.array(inputs)
        elif type(inputs) in [int, float]:
            self.val = inputs
        else:
            raise ValueError(
                f"Unsupported type {type(inputs)} for variable inputs.")

        return self.val

    def backward(self):
        """Backward pass of the reverse mode differentiation for a variable.
        :return: derivative result -> int, float, or np.array 
        """
        if self.node.compute():
            return {self.name: self.node.adjoint}
        else:
            return {}

    def __eq__(self, other):
        """
        This allows for == operation between Variable instance and other class instance. 
        :param other 
        :return: Boolean
        """
        if type(other) != Variable:
            return False

        return super().__eq__(other) and self.name == other.name

    def __str__(self):
        return f"Variable object, name {self.name}"
