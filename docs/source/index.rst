.. auto_diff_team_04 documentation master file, created by
   sphinx-quickstart on Sun Dec 11 01:34:11 2022.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to ``auto_diff_team_04``'s documentation!
=================================================

.. toctree::
   :maxdepth: 2
   :caption: Contents:

intro

Introduction
************

``Automatic Differentiation (auto_diff_team_04)`` package is a python package that realizes forward and reverse mode automatic differentiation method on custom input functions. 

Motivation
**********

In scientific research or engineering projects, sometimes we would want to compute the derivative of certain functions (For example, the $f'(x)$ term in Newton's method.) For simple input functions, we can compute an exact analytical solution with ease. However, once the inputs become complicated, it may be hard or even impossible to calculate an analytical solution. This problem becomes especially intractable in deep learning, where we are interested in the derivative of model losses with respect to input features, both of which could be vectors with hundreds of dimensions.

An alternative way is to compute the derivative using numerical methods like automatic differentiation. It breaks down large, complex input function into the product of elementary functions, whose derivatives are trivial to compute. By tracing the gradient of intermediate results and repeatedly applying Chain Rule, AutoDiff is able to compute the gradient of any input function in a certain direction. This carries significant importance as almost all machine learning methods rely on gradient descent, and the absolute prerequisite of gradient descent is to compute the gradient. 


Modules
*******

.. toctree::
   :maxdepth: 4

   auto_diff


.. * :ref:`source/auto_diff`
.. * :ref:`source/auto_diff.dual`
.. * :ref:`source/auto_diff.expression`


Indices and tables
==================

* :ref:`genindex`
* :ref:`modindex`
* :ref:`search`

