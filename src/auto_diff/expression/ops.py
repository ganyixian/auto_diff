#!/usr/bin/env python3
# Project    : AutoDiff
# File       : ops.py
# Description: mathematical operators for Function evaluation
# Copyright 2022 Harvard University. All Rights Reserved.


import numpy as np

from ..dual import Dual
from .expression import Function

"""
This module provides mathematical operations for Function evaluation. Operations include elementary functions 
like exp, log, and basic trigonometry functions.
All operators are compatible with Dual numbers.
"""

class ops:
    def _sin(x):
        res = Dual.sin(x) if type(x) == Dual else np.sin(x)
        return res


    def _cos(x):
        res = Dual.cos(x) if type(x) == Dual else np.cos(x)
        return res


    def _tan(x):
        res = Dual.tan(x) if type(x) == Dual else np.tan(x)
        return res


    def _exp(x):
        res = Dual.exp(x) if type(x) == Dual else np.exp(x)
        return res


    def _log(x):
        res = Dual.log(x) if type(x) == Dual else np.log(x)
        return res


    def sin(x):
        if isinstance(x, (int, float)):
            return np.sin(x)
        return Function(x, f=ops._sin)


    def cos(x):
        if isinstance(x, (int, float)):
            return np.cos(x)
        return Function(x, f=ops._cos)


    def tan(x):
        if isinstance(x, (int, float)):
            return np.tan(x)
        return Function(x, f=ops._tan)


    def exp(x):
        if isinstance(x, (int, float)):
            return np.exp(x)
        return Function(x, f=ops._exp)


    def log(x):
        if isinstance(x, (int, float)):
            return np.log(x)
        return Function(x, f=ops._log)
