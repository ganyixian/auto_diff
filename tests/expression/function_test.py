import pytest

from auto_diff.dual.dual import Dual
from auto_diff.expression import Expression, Function, Variable

class TestFunctionUnit:

    def test_function_init(self):
        f1 = Function(Variable('a'))
        f2 = Function(Variable('a'), Variable('b'))

    def test_function__str__(self):
        f1 = Function(Variable('a'))
        assert str(f1) == "Function object, function of {'a'}"

    def test_function_clear(self):
        f1 = Function(Variable('a'))
        f1.clear()
        assert f1.e1.val == None
        assert f1.val == None

        f2 = Function(Variable('a'), Variable('b'))
        f2.clear()
        assert f2.e1.val == None
        assert f2.e2.val == None
        assert f2.val == None

    # Failure testing started here
    def test_function_eq(self):
        f1 = Function(Variable('a'))
        f2 = Function(Variable('a'), Variable('b'))

        assert f1 == Function(Variable('a'))
        assert f2 == Function(Variable('a'), Variable('b'))
    
     # Failure testing started here
    def test_function_eq_fail(self):
        f1 = Function(Variable('a'))
        f2 = Function(Variable('a'), Variable('b'))

        assert (f1 == f2) == False
        assert (f1 == 1) == False
    
    def test_function_forward_e1(self):
        f1 = Variable('a') ** 2
        result = f1.forward({'a': 1} ,{'a': 1})
        assert result == Dual(1, 2)

    def test_function_forward_e1_e2_1(self):
        f1 = Variable('a') * Variable('b')
        result = f1.forward({'a': 1, 'b': 2} ,{'a': 0, 'b': 1})
        assert result == Dual(2, 1)

    def test_function_forward_e1_e2_2(self):
        f1 = Variable('a') * Variable('b')
        result = f1.forward({'a': 1, 'b': 2} ,{'a': 1, 'b': 1})
        assert result == Dual(2, 3)
    
    def test_function_forward_e1_e2_2(self):
        f1 = Variable('a') * Variable('b')
        result = f1.forward({'a': 1, 'b': 2} ,{'a': 1, 'b': 1})
        result = f1.forward({'a': 1, 'b': 2} ,{'a': 1, 'b': 1})

        assert result == Dual(2, 3)

    def test_function_backward_e1_e2(self):
        x, y = Variable('x', mode='r'), Variable('y', mode='r')
        f = Expression.sin(x * 4) + Expression.cos(y * 4)
        f_val, f_deriv = f({'x': 1, 'y': 2})
        print(f_val, f_deriv)

        assert f_val == -0.9023025291165417
        assert f_deriv == {'x': -2.6145744834544478, 'y': -3.957432986493527}

if __name__ == '__main__':
      # create a function f = sin(4x) + cos(4y)
    f = f + Expression.exp(x * y)  # f = sin(4x) + cos(4y) + e^(xy)
    f_val, f_deriv = f({'x': 1, 'y': 2})
    # return the value of f and the derivative in the direction of seed at x = 1, y = 1.
    print(f_val, f_deriv)
    # a = DualVector([1,1,1], [2,2,2])
    # print(type(a) == Dual)
    # exit()
    # print(f(inputs))
    # v1 = [Dual(1,0), Dual(2,1), Dual(1,2),Dual(-1,1)]
    # v2 = [Dual(0,1), Dual(1,2), Dual(1,2),Dual(-1,1)]
    # print(v1 * v2)


