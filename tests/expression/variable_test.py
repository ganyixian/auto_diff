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