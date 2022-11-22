from auto_diff.dual import Dual
from auto_diff.expression import ops

import numpy as np
import pytest

class TestOps:

    def test_ops_sin(self):
        x1 = Dual(np.pi, 0)
        x2 = Dual(np.pi, np.pi)
        assert Dual.sin(x1) == Dual(0, 0)
        assert Dual.sin(x2) == Dual(0, -np.pi)

    def test_ops_cos(self):
        x1 = Dual(np.pi, 0)
        x2 = Dual(np.pi, np.pi)
        assert Dual.cos(x1) == Dual(-1, 0)
        assert Dual.cos(x2) == Dual(-1, 0)

    def test_ops_tan(self):
        x1 = Dual(np.pi/4, 0)
        x2 = Dual(np.pi/4, np.pi/2)
        assert Dual.tan(x1) == Dual(1, 0)
        assert Dual.tan(x2) == Dual(1, np.pi)

    def test_ops_exp(self):
        x1 = Dual(0, 0)
        x2 = Dual(0, 1)
        assert Dual.exp(x1) == Dual(1, 0)
        assert Dual.exp(x2) == Dual(1, 1)

    def test_ops_log(self):
        x1 = Dual(1, 0)
        x2 = Dual(np.e, np.e)
        assert Dual.log(x1) == Dual(0, 0)
        assert Dual.log(x2) == Dual(1, 1)
