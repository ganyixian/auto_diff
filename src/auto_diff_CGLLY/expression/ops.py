#!/usr/bin/env python3
# Project    : AutoDiff
# File       : ops.py
# Description: mathematical operators for Function evaluation
# Copyright 2022 Harvard University. All Rights Reserved.


import numpy as np

from ..dual import Dual

"""
This module provides mathematical operations for Function evaluation. Operations include elementary functions 
like exp, log, sqrt, trigonometry functions, inverse trigonometry functions and hyperbolic functions.
All operators are compatible with Dual numbers.
"""

def _sin(x):
    """Calculate the sine operation of input

    :param x: Real or Dual Number
    :return: Corresponding input type
    """

    res = Dual.sin(x) if isinstance(x, Dual) else np.sin(x)
    return res


def _cos(x):
    """Calculate the cosine operation of input

    :param x: Real or Dual Number
    :return: Corresponding input type
    """
    res = Dual.cos(x) if isinstance(x, Dual) else np.cos(x)
    return res


def _tan(x):
    """Calculate the tangent operation of input

    :param x: Real or Dual Number
    :return: Corresponding input type
    """
    res = Dual.tan(x) if isinstance(x, Dual) else np.tan(x)
    return res

def _arcsin(x):
    """Calculate the inverse of sine operation of input

    :param x: Real or Dual Number
    :return: Corresponding input type
    """
    res = Dual.arcsin(x) if isinstance(x, Dual) else np.arcsin(x)
    return res

def _arccos(x):
    """Calculate the inverse of cosine operation of input

    :param x: Real or Dual Number
    :return: Corresponding input type
    """
    res = Dual.arccos(x) if isinstance(x, Dual) else np.arccos(x)
    return res

def _arctan(x):
    """Calculate the inverse of tangent operation of input

    :param x: Real or Dual Number
    :return: Corresponding input type
    """
    res = Dual.arctan(x) if isinstance(x, Dual) else np.arctan(x)
    return res

def _sinh(x):
    """Calculate the hyperbolic sine operation of input

    :param x: Real or Dual Number
    :return: Corresponding input type
    """
    res = Dual.sinh(x) if isinstance(x, Dual) else np.sinh(x)
    return res

def _cosh(x):
    """Calculate the hyperbolic cosine operation of input

    :param x: Real or Dual Number
    :return: Corresponding input type
    """
    res = Dual.cosh(x) if isinstance(x, Dual) else np.cosh(x)
    return res


def _tanh(x):
    """Calculate the hyperbolic tangent operation of input

    :param x: Real or Dual Number
    :return: Corresponding input type
    """
    res = Dual.tanh(x) if isinstance(x, Dual) else np.tanh(x)
    return res

def _exp(x):
    """Calculate the exponential operation of input

    :param x: Real or Dual Number
    :return: Corresponding input type
    """
    res = Dual.exp(x) if isinstance(x, Dual) else np.exp(x)
    return res


def _log(x):
    """Calculate the natural logarithmic operation of input

    :param x: Real or Dual Number
    :return: Corresponding input type
    """
    res = Dual.log(x) if isinstance(x, Dual) else np.log(x)
    return res

def _log_base(x, base):
    """Calculate the logarithm of input with a chosen base (positive, not equal to 1)

    :param x: Real or Dual number
    :param base: positive real number
    :return: Corresponding input type
    """
    res = Dual.log_base(x, base) if isinstance(x, Dual) else np.log(x)/np.log(base)
    return res

def _sigmoid(x):
    """Calculate the sigmoid operation of input

    :param x: Real or Dual Number
    :return: Corresponding input type
    """
    res = Dual.sigmoid(x) if isinstance(x, Dual) else 1/(1 + np.exp(-x))
    return res

def _sqrt(x):
    """Calculate the square root operation of input

    :param x: Real or Dual Number
    :return: Corresponding input type
    """
    res = Dual.sqrt(x) if isinstance(x, Dual) else np.sqrt(x)
    return res