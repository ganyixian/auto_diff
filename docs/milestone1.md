# Team04 Final Project: Automatic Differentiation Package

## Introduction
*Automatic Differentiation (AD)* project is a python package that realizes forward mode automatic differentiation method on custom input fucntions. 

In scientific research or engineering projects, sometimes we would want to compute the derivative of certain functions (the `f'(x)` term in Newton's method for example.) For simple input fucntions, we can compute an exact analytical solution with ease. However, once the inputs became complicated, it may be hard or even impossible to calculate an analytical solution. This problem becomes especially intractable in deep learning, where we are interested in the derivative of model losses with respect to input features, both of which could be vectors with hundreds of dimensions.

An alternative way is to compute the derivative using numerical method like Automatic Differentiation. It breaks down large, complex input function into the product of elementary functions, whose derivative are trivial to compute. By tracing the gradient of intermediate results and repeatedly applying chain rule, Automatic Differentiation is able to compute the gradient of input function in a certain direction. This carries significant importance as almost all machine learning methods rely on gradient descent, and the absolute prerequisite of gradient descent is to compute the gradient.    

### Background

### Usage

### Software Organization

### Implementation

### Licensing


