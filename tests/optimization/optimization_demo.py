# Performing gradient descent 

from auto_diff.expression import Expression, Variable, Function, ops, Compose
import numpy as np

x = 3
step_scale = 0.3
num_steps = 20
print(x)

for step_i in range(num_steps):
    x = Variable('x')
    x, y = Variable.vars(['x', 'y'], 'r')
    f = Compose([x ** 2 + x * y, x-y])
    inputs = {'x': np.array([1, 2]), 'y': 2}
    res = f(inputs)
    x = x - step_scale * res[1] 
    print(x)