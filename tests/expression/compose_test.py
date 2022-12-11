from auto_diff.expression import Variable, Compose

import numpy as np
import pytest

class TestCompose:

    def test_compose_init(self):
        f1 = Variable('a')
        f2 = Variable('b')

        f_all = Compose([f1, f2])
        assert f_all

    def test_compose_len(self):
        f1 = Variable('a')
        f2 = Variable('b')

        f_all = Compose([f1, f2])
        assert len(f_all) == 2
    
    def test_compose_iter(self):
        f1 = Variable('a')
        f2 = Variable('b')

        f_all = Compose([f1, f2])
        assert list(f_all) == [f1, f2]

    def test_compose_str(self):
        f1 = Variable('a')
        f2 = Variable('b')

        f_all = Compose([f1, f2])
        assert str(f_all) == '\n'.join([str(func) for func in [f1, f2]])
    
    def test_compose_call_no_seed(self):
        f1 = Variable('a')
        f2 = Variable('b')

        f_all = Compose([f1, f2])
        input = {'a': 1, 'b': 2}
        result = f_all(input)

        assert result == [([1], {'a': [[1]], 'b': [[0]]}), ([2], {'a': [[0]], 'b': [[1]]})]

    def test_compose_call_with_seed(self):
        f1 = Variable('a')
        f2 = Variable('b')

        f_all = Compose([f1, f2])
        result = f_all({'a': 1, 'b': 2}, {'a': 0, 'b': 1})
        assert result == [([1], [0]), ([2], [1])]
    
    def test_compose_call_with_seed(self):
        f1 = Variable('a')
        f2 = Variable('b')

        f_all = Compose([f1, f2])
        result = f_all({'a': [1, 0], 'b': [2, 0]}, {'a': 0, 'b': 1})
        print(result)
        assert result == [([1], [0]), ([2], [1])]



    # def test_compose__str__(self):
    #     f1 = Function(Variable('a'))
    #     assert str(f1) == "Function object, function of {'a'}"