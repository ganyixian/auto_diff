import sys
sys.path.append('src/')
sys.path.append('../../src')
from auto_diff_CGLLY.dual import Dual
from auto_diff_CGLLY.expression import ops

import numpy as np
import pytest

class TestOps:

    def test_ops_sin(self):
        x1 = Dual(np.pi, 0)
        x2 = Dual(np.pi, np.pi)
        assert Dual.sin(x1) == Dual(0, 0)
        assert Dual.sin(x2) == Dual(0, -np.pi)
        assert ops._sin(0.0) == 0
        assert ops._sin(0) == 0


    def test_ops_cos(self):
        x1 = Dual(np.pi, 0)
        x2 = Dual(np.pi, np.pi)
        assert Dual.cos(x1) == Dual(-1, 0)
        assert Dual.cos(x2) == Dual(-1, 0)
        assert ops._cos(0.0) == 1
        assert ops._cos(0) == 1

    def test_ops_tan(self):
        x1 = Dual(np.pi/4, 0)
        x2 = Dual(np.pi/4, np.pi/2)
        assert Dual.tan(x1) == Dual(1, 0)
        assert Dual.tan(x2) == Dual(1, np.pi)
        assert ops._tan(0.0) == 0
        assert ops._tan(0) == 0

    def test_ops_arcsin(self):
        x1 = Dual(0, 0)
        x2 = Dual(0.5, 0.5)
        assert Dual.arcsin(x1) == Dual(0, 0)
        assert Dual.arcsin(x2) == Dual(0.5235987755982988, 0.5773502691896258)
        assert ops._arcsin(0.0) == 0
        assert ops._arcsin(0) == 0

    def test_ops_arccos(self):
        x1 = Dual(0, 0)
        x2 = Dual(0.5, 0)
        assert Dual.arccos(x1) == Dual(1.5707963267948966, 0)
        assert Dual.arccos(x2) == Dual(1.0471975511965976, 0)
        assert ops._arccos(1.0) == 0
        assert ops._arccos(1) == 0

    def test_ops_arctan(self):
        x1 = Dual(0, 0)
        x2 = Dual(np.pi/4, 0)
        assert Dual.arctan(x1) == Dual(0, 0)
        assert Dual.arctan(x2) == Dual(0.6657737500283538, 0)
        assert ops._arctan(0.0) == 0
        assert ops._arctan(0) == 0

    def test_ops_sinh(self):
        x1 = Dual(0, 0)
        x2 = Dual(0.5, 0.5)
        assert Dual.sinh(x1) == Dual(0, 0)
        assert Dual.sinh(x2) == Dual(0.5210953054937474, 0.5638129826031903)
        assert ops._sinh(0.0) == 0
        assert ops._sinh(0) == 0

    def test_ops_cosh(self):
        x1 = Dual(0, 0)
        x2 = Dual(0.5, 0)
        assert Dual.cosh(x1) == Dual(1, 0)
        assert Dual.cosh(x2) == Dual(1.1276259652063807, 0)
        assert ops._cosh(0.0) == 1
        assert ops._cosh(0) == 1

    def test_ops_tanh(self):
        x1 = Dual(0, 0)
        x2 = Dual(np.pi/4, 0)
        assert Dual.tanh(x1) == Dual(0, 0)
        assert Dual.tanh(x2) == Dual(0.6557942026326724, 0)
        assert ops._tanh(0.0) == 0
        assert ops._tanh(0) == 0

    def test_ops_exp(self):
        x1 = Dual(0, 0)
        x2 = Dual(0, 1)
        assert Dual.exp(x1) == Dual(1, 0)
        assert Dual.exp(x2) == Dual(1, 1)
        assert ops._exp(0.0) == 1
        assert ops._exp(0) == 1

    def test_ops_log(self):
        x1 = Dual(1, 0)
        x2 = Dual(np.e, np.e)
        assert Dual.log(x1) == Dual(0, 0)
        assert Dual.log(x2) == Dual(1, 1)
        assert ops._log(1.0) == 0
        assert ops._log(1) == 0

    def test_ops_log_base(self):
        x1 = Dual(1, 0)
        x2 = Dual(np.e, np.e)
        assert Dual.log_base(x1, 10) == Dual(0, 0)
        assert Dual.log_base(x2, np.e) == Dual(1, 1)
        assert Dual.log_base(x2, 10) == Dual(0.43429448190325176, 0.4342944819032518)
        assert ops._log_base(1.0, 10) == 0
        assert ops._log_base(1, 10) == 0

    def test_ops_sigmoid(self):
        x1 = Dual(0, 0)
        x2 = Dual(np.e, np.e)
        assert Dual.sigmoid(x1) == Dual(0.5, 0)
        assert Dual.sigmoid(x2) == Dual(0.9380968325850065, 0.1578537933353061)
        assert ops._sigmoid(0) == 0.5
        assert ops._sigmoid(0.0) == 0.5

    def test_ops_sqrt(self):
        x1 = Dual(1, 0)
        x2 = Dual(4, 1)
        assert Dual.sqrt(x1) == Dual(1, 0)
        assert Dual.sqrt(x2) == Dual(2, 0.25)
        assert ops._sqrt(1.0) == 1
        assert ops._sqrt(1) == 1
