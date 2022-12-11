import sys
sys.path.append('src/')
sys.path.append('../../src')
from auto_diff_CGLLY.expression import Variable, Compose

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
    
    def test_compose_call_reverse_with_seed(self):
        x, y = Variable.vars(['x', 'y'], 'r')
        f = Compose([x+y, x-y])
        inputs = {'x': np.array([1, 2]), 'y': 2}
        result = f(inputs)

        assert str(result) == "[(array([3, 4]), {'x': 1, 'y': 1}), (array([-1,  0]), {'x': 1, 'y': -1})]"
    
    def test_compose_call_array_input(self):
        f1 = Variable('a')
        f2 = Variable('b')

        f_all = Compose([f1, f2])
        result = f_all({'a': np.array([1, 2]), 'b': 2})
        print(result)
        assert result == [([1, 2], {'a': [[1, 0], [0, 1]], 'b': [[0, 0]]}), \
                            ([2], {'a': [[0], [0]], 'b': [[1]]})]
    
    def test_compose_call_clear(self):
        f1 = Variable('a')
        f2 = Variable('b')

        f_all = Compose([f1, f2])
        result = f_all({'a': np.array([1, 2]), 'b': 2})
        f_all.clear()

        assert f1.val == None
        assert f2.val == None

