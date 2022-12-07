#!/usr/bin/env python3
# Project    : AutoDiff
# File       : ops.py
# Description: mathematical operators for Function evaluation
# Copyright 2022 Harvard University. All Rights Reserved.


import numpy as np

from ..dual import Dual

"""
This module provides mathematical operations for Function evaluation. Operations include elementary functions 
like exp, log, and basic trigonometry functions.
All operators are compatible with Dual numbers.
"""

def _sin(x):
    res = Dual.sin(x) if isinstance(x, Dual) else np.sin(x)
    return res


def _cos(x):
    res = Dual.cos(x) if isinstance(x, Dual) else np.cos(x)
    return res


def _tan(x):
    res = Dual.tan(x) if isinstance(x, Dual) else np.tan(x)
    return res

def _arcsin(x):
    res = Dual.arcsin(x) if isinstance(x, Dual) else np.arcsin(x)
    return res

def _arccos(x):
    res = Dual.arccos(x) if isinstance(x, Dual) else np.arccos(x)
    return res

def _arctan(x):
    res = Dual.arctan(x) if isinstance(x, Dual) else np.arctan(x)
    return res

def _sinh(x):
    res = Dual.sinh(x) if isinstance(x, Dual) else np.sinh(x)
    return res

def _cosh(x):
    res = Dual.cosh(x) if isinstance(x, Dual) else np.cosh(x)
    return res


def _tanh(x):
    res = Dual.tanh(x) if isinstance(x, Dual) else np.tanh(x)
    return res

def _exp(x):
    res = Dual.exp(x) if isinstance(x, Dual) else np.exp(x)
    return res


def _log(x):
    res = Dual.log(x) if isinstance(x, Dual) else np.log(x)
    return res

def _log_base(x, base):
    res = Dual.log_base(x, base) if isinstance(x, Dual) else np.log(x)/np.log(base)
    return res

def _sigmoid(x):
    sig = 1/(1 + np.exp(-x))
    res = Dual.sigmoid(x) if isinstance(x, Dual) else sig
    return res

def _sqrt(x):
    res = Dual.sqrt(x) if isinstance(x, Dual) else np.sqrt(x)
    return res