#!/usr/bin/env python3
# Project    : AutoDiff
# File       : expression.py
# Description: autodiff functional expressions
# Copyright 2022 Harvard University. All Rights Reserved.

from .expression import Expression

class Function(Expression):

    def __init__(self, e1, e2=None, f=None, mode='f'):
        super(Function, self).__init__(mode=mode)
        self.e1 = e1
        self.e2 = e2
        self.f = f
        self.varname = e1.varname
        if e2:
            self.varname.update(e2.varname)

    def forward(self, inputs, seed):
        if self.val:
            return self.val

        res1 = self.e1.forward(inputs, seed)
        if self.e2:
            res2 = self.e2.forward(inputs, seed)
            self.val = self.f(res1, res2)
        else:
            self.val = self.f(res1)

        return self.val

    def clear(self):
        self.e1.clear()
        if self.e2:
            self.e2.clear()
        self.val = None