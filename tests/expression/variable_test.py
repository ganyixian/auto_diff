import pytest

from auto_diff.dual import Dual
from auto_diff.expression import Variable

class TestVariable:

    def test_variable_init(self):
        v1 = Variable('a', val = 1)

    def test_variable_vars(self):
        varlist = ['a', 'b', 'c']
        expected_result = [Variable('a'), Variable('b'), Variable('c')]
        result = Variable.vars(varlist)

        assert type(result) is list
        assert result == expected_result
    
    def test_variable_eq(self):
        v1 = Variable('a', val = 1)
        result = v1 == Variable('a', val = 1)

        assert result
    
    def test_variable_forward_seed_is_None(self):
        v1 = Variable('a')
        with pytest.raises(Exception):
            v1.forward({},None)

    def test_variable_forward_dict_dict(self):
        v1 = Variable('a')
        result = v1.forward({'a': 1},{'a': 2})
        assert result == Dual(1, 2)
    
    def test_variable_forward_int_int(self):
        v1 = Variable('a')
        result = v1.forward({'a': 1},{'a': 2})
        assert result == Dual(1, 2)

    def test_variable_forward_int_dict_invalid(self):
        v1 = Variable('a')
        with pytest.raises(Exception):
            v1.forward(1, {})

    def test_variable_forward_with_val(self):
        v1 = Variable('a')
        result = v1.forward({'a': 1},{'a': 2})
        result = v1.forward({},{})

        assert result == Dual(1, 2)
        