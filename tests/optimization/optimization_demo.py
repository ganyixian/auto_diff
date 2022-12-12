import sys
sys.path.append('src/')
sys.path.append('../../src')

from auto_diff_CGLLY.expression import Variable
import numpy as np

''' A simple program to find the mininum of f = x^2'''

x = Variable('x', mode = 'r')
f = x ** 2
learning_rate = 0.2
tol = 1e-12
iter_count = 500
initial_guess = 10
position = initial_guess
for i in range(iter_count):
    f_val, grad = f({'x': float(position)}) 
    grad = grad.get('x')
    diff = -learning_rate * grad
    if np.abs(diff) <= tol:
        break
    position += diff
print("The minimum for f = x^2 is", position)
 
