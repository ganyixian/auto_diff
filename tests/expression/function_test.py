import pytest

from auto_diff.expression import Function, Variable

class TestFunctionUnit:

    def test_function_init(self):
        f1 = Function(Variable('a', val=1))
        f2 = Function(Variable('a', val=1), Variable('b', val=2))

    def test_function_clear(self):
        f1 = Function(Variable('a', val=1))
        f1.clear()
        assert f1.e1.val == None
        assert f1.val == None

        f2 = Function(Variable('a', val=1), Variable('b', val=2))
        f2.clear()
        assert f2.e1.val == None
        assert f2.e2.val == None
        assert f2.val == None

    # Failure testing started here
    def test_function_eq(self):
        f1 = Function(Variable('a', val=1))
        f2 = Function(Variable('a', val=1), Variable('b', val=2))

        assert f1 == Function(Variable('a', val=1))
        assert f2 == Function(Variable('a', val=1), Variable('b', val=2))
    
     # Failure testing started here
    def test_function_eq_fail(self):
        f1 = Function(Variable('a', val=1))
        f2 = Function(Variable('a', val=1), Variable('b', val=2))

        assert (f1 == f2) == False



