# Team04 Final Project: Automatic Differentiation Package

## Introduction
**Automatic Differentiation (AD)** project is a python package that realizes forward mode automatic differentiation method on custom input fucntions. 

In scientific research or engineering projects, sometimes we would want to compute the derivative of certain functions (the `f'(x)` term in Newton's method for example.) For simple input fucntions, we can compute an exact analytical solution with ease. However, once the inputs became complicated, it may be hard or even impossible to calculate an analytical solution. This problem becomes especially intractable in deep learning, where we are interested in the derivative of model losses with respect to input features, both of which could be vectors with hundreds of dimensions.

An alternative way is to compute the derivative using numerical method like automatic differentiation. It breaks down large, complex input function into the product of elementary functions, whose derivative are trivial to compute. By tracing the gradient of intermediate results and repeatedly applying chain rule, AD is able to compute the gradient of any input function in a certain direction. This carries significant importance as almost all machine learning methods rely on gradient descent, and the absolute prerequisite of gradient descent is to compute the gradient.    

## Background

*This section provides a brief overview of the mechanism of AD. Users not interested in the math may skip to* **Usage** *section below*

- **Elementary Operation**

  The key concept of AD is to break down a complicated function into small, managable steps, and solving each step individually. Typically, each step in AD would only perform one elementary operation. Here, "Elementary Operations" refer to both arithmatic operation (`+`, `-`, `*`, scalar division, power, etc.), and elementary functions (`exp`, `log`, `sin `, `cos`, etc.) These elementary operations should take only one or two inputs, and its partial derivative with respect to both inputs should be easy to compute. We would later chain these intermediate derivatives to get the overall result.
  
- **Chain Rule**

  Chain rule in calculus is the rule to compute the derivative of compound functions. 

- **Directional Derivative**


- **Compute Graph**

- **Trace**


### Usage

### Software Organization

### Implementation

### Licensing


