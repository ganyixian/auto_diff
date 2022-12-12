#!/usr/bin/env python3
# Project    : AutoDiff
# File       : dual.py
# Description: Dual number
# Copyright 2022 Harvard University. All Rights Reserved.


import numpy as np

"""This module implements dual numbers and provides mathematical operations on dual numbers
Dual number is the underlying data structure for forward mode AutoDiff
"""

def vec_dec(op):
    """Dual vector decorator

    :param op: The function to decorate
    :return: Decorated function
    """
    def func(d1, d2=None):
        if isinstance(d1, DualVector):
            if d2 is not None and type(d2) in [int, float, Dual]:
                res_list = [op(x, d2) for x in d1.dual_vec]
            elif d2 is not None:
                assert len(d1) == len(d2), f'operands length mismatch, found {len(d1)} and {len(d2)}.'
                res_list = [op(x1, x2) for x1, x2 in zip(d1, d2)]
            else:
                res_list = [op(x) for x in d1.dual_vec]
            res = DualVector(vec=res_list)
        else:
            res = op(d1, d2) if d2 is not None else op(d1)

        return res

    return func


class Dual:
    """Dual number object
    """
    def __init__(self, real=0, dual=0):
        self.real = real
        self.dual = dual

    @vec_dec
    def __add__(self, other):
        """
        This allows for addition with Dual Number instances or scalar numbers. 
        :param other: Dual number or scalar number
        :return: Dual number object
        :raises TypeError
        """
    
        if type(other) in [int, float]:
            return Dual(self.real + other, self.dual)
        elif type(other) == Dual:
            return Dual(self.real + other.real, self.dual + other.dual)
        else:
            raise TypeError("Addition operation not supported for type DualNumber and {}".format(type(other)))


    @vec_dec
    def __mul__(self, other):
        """
        This allows for multiplication with Dual Number instances or scalar numbers. 
        :param other: Dual number or scalar number
        :return: Dual number object
        :raises TypeError
        """
        if type(other) in [int, float]:
            return Dual(self.real * other, self.dual * other)
        elif type(other) == Dual:
            return Dual(self.real * other.real,
                        self.real * other.dual + self.dual * other.real)
        else:
            raise TypeError("Multiplication operation not supported for type DualNumber and {}".format(type(other)))

    @vec_dec
    def __sub__(self, other):
        """
        This allows for substraction with Dual Number instances or scalar numbers. 
        :param other: Dual number or scalar number
        :return: Dual number object
        :raises TypeError
        """
        return self.__add__(-other)

    @vec_dec
    def __rsub__(self, other):
        """
        This will be called when int/float - Dual Number instance. 
        :param other: int/float
        :return: Dual number object
        """
        if type(other) in [int, float]:
            return Dual(other - self.real, - self.dual)

    @vec_dec
    def __neg__(self):
        """
        This allows for negation of Dual number instance 
        :return: Dual number object
        """
        return Dual(-self.real, -self.dual)

    @vec_dec
    def __truediv__(self, other):
        """
        This allows for true division between Dual Number instances and scalar numbers. 
        :param other: Dual number or scalar number
        :return: Dual number object
        :raises TypeError
        """
        if type(other) in [int, float]:
            return Dual(self.real / other, self.dual / other)
        elif type(other) == Dual:
            return Dual(self.real / other.real,
                        (self.dual * other.real - self.real * other.dual) / other.real ** 2)
        else:
            raise TypeError("True Division operation not supported for type DualNumber and {}".format(type(other)))
            
    @vec_dec
    def __rtruediv__(self, other):
        """
        This will be called when (int/float) / Dual Number instance. 
        :param other: int/float
        :return: Dual number object
        """
        assert type(other) in [int, float]

        return Dual(other / self.real, - other * self.dual / self.real ** 2)

    @vec_dec
    def __pow__(self, power, modulo=None):
        """
        This allows for power operation between Dual Number instances and scalar numbers. 
        :param other: Dual number or scalar number
        :return: Dual number object
        :raises TypeError
        """
        if type(power) in [float, int]:
            c, d = power, 0
        elif type(power) == Dual:
            c, d = power.real, power.dual
        else:
            raise TypeError("Power operation not supported for type DualNumber and {}".format(type(power)))

        a, b = self.real, self.dual
        return Dual(a ** c, a ** c * (d * np.log(a) + b * c / a))
    
    @vec_dec
    def __rpow__(self, other, modulo=None):
        """
        This will be called when (int/float) ** Dual Number instance. 
        :param other: int/float
        :return: Dual number object
        :raises TypeError
        """
        if type(other) in [float, int]:
            c = other
        else:
            raise TypeError("Power operation not supported for type DualNumber and {}".format(type(other)))
        a, b = self.real, self.dual
        return Dual(c ** a, np.log(c) * c ** a * b)

    __radd__ = __add__
    __rmul__ = __mul__

    def __len__(self):
        return 1

    def __iter__(self):
        yield self

    def __str__(self):
        return "real {}, dual {}".format(self.real, self.dual)

    def __eq__(self, other):
        """
        This allows for == operation between Dual Number instance and other class instance. 
        :param other 
        :return: Boolean
        """
        return type(other) == Dual and np.isclose(self.real, other.real) and np.isclose(self.dual, other.dual)

    def __ne__(self, other):
        """
        This allows for != operation between Dual Number instance and other class instance. 
        :param other 
        :return: Boolean
        """
        return type(other) != Dual or not np.isclose(self.real, other.real) or not np.isclose(self.dual, other.dual)

    def get_real(self):
        """Get the real part of Dual number
        """
        return self.real

    def get_dual(self):
        """Get the dual part of Dual number
        """
        return self.dual

    @staticmethod
    @vec_dec
    def exp(x):
        """Calculate the exponential operation of input

        :param x: Dual number
        :return: Dual number
        """
        return Dual(np.exp(x.real), x.dual * np.exp(x.real))

    @staticmethod
    @vec_dec
    def log(x):
        """Calculate the natural logarithmic operation of input

        :param x: Dual number
        :return: Dual number
        """
        return Dual(np.log(x.real), x.dual / x.real)
    
    @staticmethod
    @vec_dec
    def log_base(x, base):
        """Calculate the logarithm of input with a chosen base (positive, not equal to 1)

        :param x: Dual number
        :param base: positive real number
        :return: Dual number
        """
        return Dual(np.log(x.real) / np.log(base), x.dual / (x.real * np.log(base)))

    @staticmethod
    @vec_dec
    def sin(x):
        """Calculate the sine operation of input

        :param x: Dual Number
        :return: Dual Number
        """
        return Dual(np.sin(x.real), x.dual * np.cos(x.real))

    @staticmethod
    @vec_dec
    def cos(x):
        """Calculate the cosine operation of input

        :param x: Dual Number
        :return: Dual Number
        """
        return Dual(np.cos(x.real), - x.dual * np.sin(x.real))

    @staticmethod
    @vec_dec
    def tan(x):
        """Calculate the tangent operation of input

        :param x: Dual Number
        :return: Dual Number
        """
        return Dual(np.tan(x.real), x.dual / np.power(np.cos(x.real), 2))
    
    @staticmethod
    @vec_dec
    def arcsin(x):
        """Calculate the inverse of sine operation of input

        :param x: Dual Number, and the real part domain:[-1,1]
        :return: Dual Number
        """
        return Dual(np.arcsin(x.real), x.dual /np.power(1 - x.real * x.real,0.5))

    @staticmethod
    @vec_dec
    def arccos(x):
        """Calculate the inverse of cosine operation of input

        :param x: Dual Number, and the real part domain:[-1,1]
        :return: Dual Number
        """
        return Dual(np.arccos(x.real), - x.dual /np.power(1 - x.real * x.real,0.5))

    @staticmethod
    @vec_dec
    def arctan(x):
        """Calculate the inverse of tangent operation of input

        :param x: Dual Number
        :return: Dual Number, and the real part domain is all real numbers
        """
        return Dual(np.arctan(x.real), x.dual /(1 + x.real * x.real))

    @staticmethod #(sinh, cosh, tanh)
    @vec_dec
    def sinh(x):
        """Calculate the hyperbolic sine operation of input

        :param x: Dual Number
        :return: Dual Number
        """
        return Dual(np.sinh(x.real), x.dual * np.cosh(x.real))
    
    @staticmethod 
    @vec_dec
    def cosh(x):
        """Calculate the hyperbolic cosine operation of input

        :param x: Dual Number
        :return: Dual Number
        """
        return Dual(np.cosh(x.real), x.dual * np.sinh(x.real))
    
    @staticmethod 
    @vec_dec
    def tanh(x):
        """Calculate the hyperbolic tangent operation of input

        :param x: Dual Number
        :return: Dual Number
        """
        return Dual(np.tanh(x.real), x.dual * (1 - np.power(np.tanh(x.real),2)))

    @staticmethod 
    @vec_dec
    def sigmoid(x):
        """Calculate the sigmoid operation of input

        :param x: Dual Number
        :return: Dual Number
        """
        sig = 1/(1 + np.exp(-x.real))
        return Dual(sig, x.dual * (sig * (1-sig)))

    @staticmethod 
    @vec_dec
    def sqrt(x):
        """Calculate the square root operation of input

        :param x: Dual Number
        :return: Dual Number
        """
        return Dual(np.sqrt(x.real), x.dual * (0.5 * np.power(x.real,-0.5)))

class DualVector(Dual):
    def __init__(self, real=[], dual=[], vec=None):
        if vec is not None:
            self.dual_vec = vec
        else:
            assert len(real) == len(dual), f"real {real}, dual {dual}"
            self.dual_vec = [Dual(r, d) for r, d in zip(real, dual)]
        self.len = len(self.dual_vec)

    def __len__(self):
        return self.len

    def __iter__(self):
        for d in self.dual_vec:
            yield d

    def __str__(self):
        ret = ""
        for s in self.dual_vec:
            ret += str(s) + "\n"
        return ret
    
    def __eq__(self, other):
        """
        This allows for == operation between Dual Vector instance and other class instance. 
        :param other 
        :return: Boolean
        """
        return type(other) == DualVector and \
            all([self.dual_vec[i] == other.dual_vec[i] for i in range(len(self.dual_vec))])

    def get_real(self):
        """Return the real part of Dual vector as a list
        """
        return [d.real for d in self.dual_vec]

    def get_dual(self):
        """Return the dual part of Dual vector as a list
        """
        return [d.dual for d in self.dual_vec]


