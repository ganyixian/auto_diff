from auto_diff.dual import Dual
import numpy as np
import pytest

class TestDual:

    def test_dual_init(self):
        x = Dual(1, 1)
        assert x.real == 1
        assert x.dual == 1

    def test_dual_add(self):
        x = Dual(1, 1)
        y1 = 1
        y2 = 1.1
        s = "string"
        assert x + y1 == Dual(2, 1)
        assert x + y2 == Dual(2.1, 1)
        assert x + -1.1 == Dual(-0.1, 1)
        assert x + x == Dual(2, 2)
        with pytest.raises(Exception):
            x + s

    def test_dual_mul(self):
        x = Dual(1, 1)
        x2 = Dual(2, 2)
        y1 = 1
        y2 = 1.1
        s = "string"
        assert x * y1 == Dual(1, 1)
        assert x * y2 == Dual(1.1, 1.1)
        assert x * x2 == Dual(2, 4)
        with pytest.raises(Exception):
            x * s

    def test_dual_sub(self):
        x = Dual(1, 1)
        x2 = Dual(2, 2)
        y1 = 1
        y2 = 1.1
        s = "string"
        assert x - y1 == Dual(0, 1)
        assert x - y2 == Dual(-0.1, 1)
        assert x2 - y2 == Dual(0.9, 2)
        assert x - x2 == Dual(-1, -1)
        with pytest.raises(Exception):
            x - s

    def test_dual_rsub(self):
        x = Dual(1, 1)
        x2 = Dual(2, 2)
        y1 = 1
        y2 = 1.1
        assert y1 - x == Dual(0, -1)
        assert y2 - x == Dual(0.1, -1)
        assert y2 - x2 == Dual(-0.9, -2)
        assert x2 - x == Dual(1, 1)

    def test_dual_neg(self):
        x = Dual(1, 1)
        assert -x == Dual(-1, -1)

    def test_dual_truediv(self):
        x = Dual(2, 1)
        x2 = Dual(1, 1)
        y1 = 1
        y2 = 0.5
        s = "string"
        assert x / y1 == Dual(2, 1)
        assert x / y2 == Dual(4, 2)
        assert x / x2 == Dual(2, -1)
        with pytest.raises(Exception):
            x / s

    def test_dual_rtruediv(self):
        x = Dual(2, 1)
        x2 = Dual(1, 1)
        y1 = 1
        y2 = 0.5
        s = "string"
        assert y1 / x == Dual(0.5, -0.25)
        assert y2 / x == Dual(0.25, -0.125)
        assert x2 / x == Dual(0.5, 0.25)
        with pytest.raises(Exception):
            x / s

    def test_dual_pow(self):
        x = Dual(2, 1)
        x2 = Dual(1, 1)
        p1 = 2
        p2 = 0.5
        s = "string"
        assert x ** p1 == Dual(4, 4)
        assert x ** p2 == Dual(np.sqrt(2), np.sqrt(2)/4)
        assert x ** x2 == Dual(2, 2*np.log(2)+1)
        with pytest.raises(Exception):
            x ** s

    def test_dual_len(self):
        x = Dual(2, 1)
        assert len(x) == 1

    def test_dual_iter(self):
        for x in Dual(0, 1):
            assert x == Dual(0, 1)


    def test_dual_str(self):
        x = Dual(2, 1)
        self.assertEqual(str(x), "real 2, dual 1")

    def test_dual_str(self):
        x = Dual(2, 3.33)
        y = Dual(2, 3.33)
        assert x == y

    def test_dual_exp(self):
        x1 = Dual(0, 0)
        x2 = Dual(0, 1)
        assert Dual.exp(x1) == Dual(1, 0)
        assert Dual.exp(x2) == Dual(1, 1)

    def test_dual_log(self):
        x1 = Dual(1, 0)
        x2 = Dual(np.e, np.e)
        assert Dual.log(x1) == Dual(0, 0)
        assert Dual.log(x2) == Dual(1, 1)

    def test_dual_sin(self):
        x1 = Dual(np.pi, 0)
        x2 = Dual(np.pi, np.pi)
        assert Dual.sin(x1) == Dual(0, 0)
        assert Dual.sin(x2) == Dual(0, -np.pi)

    def test_dual_cos(self):
        x1 = Dual(np.pi, 0)
        x2 = Dual(np.pi, np.pi)
        assert Dual.cos(x1) == Dual(-1, 0)
        assert Dual.cos(x2) == Dual(-1, 0)

    def test_dual_tan(self):
        x1 = Dual(np.pi/4, 0)
        x2 = Dual(np.pi/4, np.pi/2)
        assert Dual.tan(x1) == Dual(1, 0)
        assert Dual.tan(x2) == Dual(1, np.pi)

