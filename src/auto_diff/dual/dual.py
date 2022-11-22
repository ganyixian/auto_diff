#!/usr/bin/env python3
# Project    : AutoDiff
# File       : dual.py
# Description: Dual number
# Copyright 2022 Harvard University. All Rights Reserved.


import numpy as np

"""
# This module implements dual numbers and provides mathematical operations on dual numbers
# Dual number is the underlying data structure for forward mode AutoDiff
# """

__all__ = ['Dual']


class Dual():
    def __init__(self, real=0, dual=0):
        self.real = real
        self.dual = dual

    def __add__(self, other):
        if type(other) in [int, float]:
            return Dual(self.real + other, self.dual)
        elif type(other) == Dual:
            return Dual(self.real + other.real, self.dual + other.dual)
        else:
            raise TypeError("Addition operation not supported for type DualNumber and {}".format(type(other)))

    def __mul__(self, other):
        if type(other) in [int, float]:
            return Dual(self.real * other, self.dual * other)
        elif type(other) == Dual:
            return Dual(self.real * other.real,
                        self.real * other.dual + self.dual * other.real)
        else:
            raise TypeError("Multiplication operation not supported for type DualNumber and {}".format(type(other)))

    def __sub__(self, other):
        return self.__add__(-other)

    def __rsub__(self, other):
        if type(other) in [int, float]:
            return Dual(other - self.real, - self.dual)

    def __neg__(self):
        return Dual(-self.real, -self.dual)

    def __truediv__(self, other):
        if type(other) in [int, float]:
            return Dual(self.real / other, self.dual / other)
        elif type(other) == Dual:
            return Dual(self.real / other.real,
                        (self.dual * other.real - self.real * other.dual) / other.real ** 2)
        else:
            raise TypeError("True Division operation not supported for type DualNumber and {}".format(type(other)))

    def __rdiv__(self, other):
        assert type(other) in [int, float]

        return Dual(other / self.real, - other * self.dual / self.real ** 2)

    def __pow__(self, power, modulo=None):
        if type(power) in [float, int]:
            c, d = power, 0
        elif type(power) == Dual:
            c, d = power.real, power.dual
        else:
            raise TypeError("Power operation not supported for type DualNumber and {}".format(type(power)))

        a, b = self.real, self.dual
        return Dual(a ** c, a ** c * (d * np.log(a) + b * c / a))

    __radd__ = __add__
    __rmul__ = __mul__

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __str__(self):
        return "real {}, dual {}".format(self.real, self.dual)

    def __eq__(self, other):
        return type(other) == Dual and np.isclose(self.real,other.real) and np.isclose(self.dual,other.dual)

    @classmethod
    def exp(cls, x):
        return cls(np.exp(x.real), x.dual * np.exp(x.real))

    @classmethod
    def log(cls, x):
        return cls(np.log(x.real), x.dual / x.real)

    @classmethod
    def sin(cls, x):
        return cls(np.sin(x.real), x.dual * np.cos(x.real))

    @classmethod
    def cos(cls, x):
        return cls(np.cos(x.real), - x.dual * np.sin(x.real))

    @classmethod
    def tan(cls, x):
        return cls(np.tan(x.real), x.dual / np.pow(np.cos(x.real), 2))


if __name__ == '__main__':
    d = Dual(1, 2)
    for a in d:
        print(a)