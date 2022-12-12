import sys
sys.path.append('src/')
sys.path.append('../../src')
import numpy as np
import pytest

from auto_diff_CGLLY.expression import Expression, Variable, Function, ops

class TestExpressionUnit:
    """
    Expression class serves as the base class of variable and function, there's not 
    oo much to test in unit test for the functionality.
    The correctness will be ensured in integration tests with function and variable.
    """

    def test_expression_init(self):
        e1 = Expression()
        assert True
    
    def test_expression_eq(self):
        e1 = Expression()
        e2 = Expression() 
        assert e1 == e2

class TestExpressionIntegration:
    def test_expression_add_expression(self):
        e1 = Variable('a')
        e2 = Variable('b')
        func = lambda x, y: x + y

        result = e1 + e2
        assert result == Function(e1, e2, func)
        
    def test_expression_add_not_expression(self):
        e1 = Variable('a')
        other = 2
        func = lambda x, y: x + other

        result = e1 + other
        assert result == Function(e1, f=func)
        
    def test_expression_radd_not_expression(self):
        e1 = Variable('a')
        other = 2
        func = lambda x, y: x + other

        result = other + e1
        assert result == Function(e1, f=func)
    
    def test_expression_mul_expression(self):
        e1 = Variable('a')
        e2 = Variable('b')
        func = lambda x, y: x * y

        result = e1 * e2
        assert result == Function(e1, e2, func)
        
    def test_expression_mul_not_expression(self):
        e1 = Variable('a')
        other = 2
        func = lambda x: x * other

        result = e1 * other
        assert result == Function(e1, f=func)
    
    def test_expression_rmul_not_expression(self):
        e1 = Variable('a')
        other = 2
        func = lambda x: x * other

        result = other * e1
        assert result == Function(e1, f=func)
    
    def test_expression_sub_expression(self):
        e1 = Variable('a')
        e2 = Variable('b')
        func = lambda x, y: x - y

        result = e1 - e2
        assert result == Function(e1, e2, func)
        
    def test_expression_sub_not_expression(self):
        e1 = Variable('a')
        other = 2
        func = lambda x: x - other

        result = e1 - other
        assert result == Function(e1, f=func)

    def test_expression_rsub_not_expression(self):
        e1 = Variable('a')
        other = 2
        func = lambda x: other - x

        result = other - e1
        assert result == Function(e1, f=func)

    def test_expression_truediv_expression(self):
        e1 = Variable('a')
        e2 = Variable('b')
        func = lambda x, y: x / y

        result = e1 / e2
        assert result == Function(e1, e2, func)
        
    def test_expression_truediv_not_expression(self):
        e1 = Variable('a')
        other = 2
        func = lambda x: x / other

        result = e1 / other
        assert result == Function(e1, f=func)

    def test_expression_rtruediv_not_expression(self):
        e1 = Variable('a')
        other = 2
        func = lambda x: other / x

        result = other / e1
        assert result == Function(e1, f=func)

    def test_expression_pow_expression(self):
        e1 = Variable('a')
        e2 = Variable('b')
        func = lambda x, y: x ** y

        result = e1 ** e2
        assert result == Function(e1, e2, func)
        
    def test_expression_pow_not_expression(self):
        e1 = Variable('a')
        other = 2
        func = lambda x: x ** other

        result = e1 ** other
        assert result == Function(e1, f=func)
    
        
    def test_expression_rpow_not_expression(self):
        e1 = Variable('a')
        other = 2
        func = lambda x: other ** x

        result = other ** e1
        assert result == Function(e1, f=func)
    
    def test_expression_neg(self):
        e1 = Variable('a')
        func = lambda x: -x

        result = -e1
        assert result == Function(e1, f=func)
    
    def test_expression_sin_expression(self):
        e1 = Variable('a')
        func = ops._sin

        result = Expression.sin(e1)
        assert result == Function(e1, f=func)
        
    def test_expression_sin_not_expression(self):
        e1 = np.pi/2.

        with pytest.raises(Exception):
            Expression.sin(e1)

    def test_expression_cos_expression(self):
        e1 = Variable('a')
        func = ops._cos

        result = Expression.cos(e1)
        assert result == Function(e1, f=func)
        
    def test_expression_cos_not_expression(self):
        e1 = np.pi/2.

        with pytest.raises(Exception):
            Expression.cos(e1)
    
    def test_expression_tan_expression(self):
        e1 = Variable('a')
        func = ops._tan

        result = Expression.tan(e1)
        assert result == Function(e1, f=func)
        
    def test_expression_tan_not_expression(self):
        e1 = np.pi/2.

        with pytest.raises(Exception):
            Expression.tan(e1)

    def test_expression_arcsin_expression(self):
        e1 = Variable('a')
        func = ops._arcsin

        result = Expression.arcsin(e1)
        assert result == Function(e1, f=func)
        
    def test_expression_arcsin_not_expression(self):
        e1 = np.pi/2.

        with pytest.raises(Exception):
            Expression.arcsin(e1)


    def test_expression_arccos_expression(self):
        e1 = Variable('a')
        func = ops._arccos

        result = Expression.arccos(e1)
        assert result == Function(e1, f=func)
        
    def test_expression_arccos_not_expression(self):
        e1 = np.pi/2.

        with pytest.raises(Exception):
            Expression.arccos(e1)
    
    def test_expression_arctan_expression(self):
        e1 = Variable('a')
        func = ops._arctan

        result = Expression.arctan(e1)
        assert result == Function(e1, f=func)
        
    def test_expression_arctan_not_expression(self):
        e1 = np.pi/2.

        with pytest.raises(Exception):
            Expression.arctan(e1)

    def test_expression_sinh_expression(self):
        e1 = Variable('a')
        func = ops._sinh

        result = Expression.sinh(e1)
        assert result == Function(e1, f=func)
        
    def test_expression_sinh_not_expression(self):
        e1 = np.pi/2.

        with pytest.raises(Exception):
            Expression.sinh(e1)

    def test_expression_cosh_expression(self):
        e1 = Variable('a')
        func = ops._cosh

        result = Expression.cosh(e1)
        assert result == Function(e1, f=func)
        
    def test_expression_cosh_not_expression(self):
        e1 = np.pi/2.

        with pytest.raises(Exception):
            Expression.cosh(e1)
    
    def test_expression_tanh_expression(self):
        e1 = Variable('a')
        func = ops._tanh

        result = Expression.tanh(e1)
        assert result == Function(e1, f=func)
        
    def test_expression_tanh_not_expression(self):
        e1 = np.pi/2.

        with pytest.raises(Exception):
            Expression.tanh(e1)
    
    def test_expression_sigmoid_expression(self):
        e1 = Variable('a')
        func = ops._sigmoid

        result = Expression.sigmoid(e1)
        assert result == Function(e1, f=func)
        
    def test_expression_sigmoid_not_expression(self):
        e1 = np.pi/2.

        with pytest.raises(Exception):
            Expression.sigmoid(e1)
    
    def test_expression_exp_expression(self):
        e1 = Variable('a')
        func = ops._exp

        result = Expression.exp(e1)
        assert result == Function(e1, f=func)
        
    def test_expression_exp_not_expression(self):
        e1 = np.pi/2.

        with pytest.raises(Exception):
            Expression.exp(e1)
    
    def test_expression_log_expression(self):
        e1 = Variable('a')
        func = ops._log

        result = Expression.log(e1)
        assert result == Function(e1, f=func)
        
    def test_expression_log_not_expression(self):
        e1 = np.pi/2.

        with pytest.raises(Exception):
            Expression.log(e1)
    
    def test_expression_log_base_expression(self):
        e1 = Variable('a')
        base = 10

        result = Expression.log_base(e1, base)
        assert result == Function(e1, f=lambda x : ops._log_base(x, base))
        
    def test_expression_log_base_not_expression(self):
        e1 = np.pi/2.
        base = 10

        with pytest.raises(Exception):
            Expression.log_base(e1, base)

    def test_expression_sqrt_expression(self):
        e1 = Variable('a')

        result = Expression.sqrt(e1)
        assert result == Function(e1, f=ops._sqrt)
        
    def test_expression_sqrt_not_expression(self):
        e1 = np.pi/2.

        with pytest.raises(Exception):
            Expression.sqrt(e1)

    def test_expression_call_int_input(self):
        a = Variable('a')
        b = Variable('b')
        c = Variable('c')
        d = Variable('d')
        f = a+b+c-d
        real, dual = f(1)

        assert real == [2]
        print(dual)
        assert dual == {'a': [1], 'b': [1], 'c': [1], 'd': [-1]}

    def test_expression_call_int_input_no_seed(self):
        a = Variable('a')
        b = Variable('b')
        c = Variable('c')
        d = Variable('d')
        f = a+b+c-d
        real, dual = f(1)

        assert real == [2]
        assert dual == {'a': [1], 'b': [1], 'c': [1], 'd': [-1]}
    
    def test_expression_call_dict_input_no_seed(self):
        a = Variable('a')
        b = Variable('b')
        c = Variable('c')
        d = Variable('d')
        f = a*b+c-d
        real, dual = f({'a':1, 'b': 2, 'c': 3, 'd': 4})

        assert real == [1]
        assert dual == {'a': [2], 'b': [1], 'c': [1], 'd': [-1]}

    def test_expression_call_int_input_with_int_seed(self):
        a = Variable('a')
        b = Variable('b')
        c = Variable('c')
        d = Variable('d')
        f = a+b+c-d
        real, dual = f(1, 1)

        assert real == [2]
        assert dual == [2]
    
    def test_expression_call_dict_input_with_int_seed(self):
        a = Variable('a')
        b = Variable('b')
        c = Variable('c')
        d = Variable('d')
        f = a*b+c-d
        real, dual = f({'a':1, 'b': 2, 'c': 3, 'd': 4}, 1)

        assert real == [1]
        assert dual == [3]


    # def test_expression_call_backward_not_implemented(self):
    #     a = Variable('a', mode='b')
        
    #     with pytest.raises(NotImplementedError):
    #         a(1)
