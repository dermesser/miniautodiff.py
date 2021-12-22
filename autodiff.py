"""
Copyright (c) 2021 Lewin Bormann

Simple backpropagation, stateful, naive approach.
"""

import numpy as np

class Expression:
    def __init__(self, left, right):
        self.l = left
        self.r = right

        self.eval_l = None
        self.eval_r = None
    
    def fw(self):
        pass
    def bw(self, grad):
        pass

    def __add__(self, other):
        return OpPlus(self, other)
    def __sub__(self, other):
        return OpMinus(self, other)
    def __neg__(self, other):
        return Num(name=self.name, id=self.id)
    def __mul__(self, other):
        return OpMult(self, other)
    def __div__(self, other):
        return OpDiv(self, other)

class OpPlus(Expression):
    def fw(self):
        return self.l.fw() + self.r.fw()

    def bw(self, grad):
        self.l.bw(grad)
        self.r.bw(grad)

class OpMinus(Expression):
    def fw(self):
        return self.l.fw() - self.r.fw()

    def bw(self, grad):
        self.l.bw(grad)
        self.r.bw(-grad)

class OpMult(Expression):
    def fw(self):
        self.eval_l = self.l.fw()
        self.eval_r = self.r.fw()
        return self.eval_l * self.eval_r

    def bw(self, grad):
        self.l.bw(self.eval_r * grad)
        self.r.bw(self.eval_l * grad)

class OpDiv(Expression):
    def fw(self):
        self.eval_l = self.l.fw()
        self.eval_r = self.r.fw()
        return self.eval_l / self.eval_r

    def bw(self, grad):
        self.l.bw(grad/self.eval_r)
        self.r.bw(-grad*self.eval_l/self.eval_r**2)

class UnaryExpression(Expression):
    def __init__(self, op, dop, e):
        self.l = e
        self.op = op
        self.dop = dop

    def fw(self):
        self.eval_l = self.l.fw()
        return self.op(self.eval_l)

    def bw(self, grad):
        self.l.bw(self.dop(self.eval_l) * grad)

def exp(e):
    return UnaryExpression(np.exp, np.exp, e)

def sin(e):
    return UnaryExpression(np.sin, np.cos, e)

def cos(e):
    return UnaryExpression(np.cos, lambda x: -np.sin(x), e)

class Num(Expression):
    def __init__(self, name=None, value=None):
        self.name = name
        self.grad = 0
        self.val = value

    def set_val(self, value):
        self.val = value

    def fw(self):
        self.grad = 0
        return self.val

    def bw(self, grad):
        self.grad += grad

a = Num('a')
b = Num('b')
c = Num('c')

e = [a * b * sin(c),
        a+c,
        a * cos(b)]

a.set_val(3)
b.set_val(4)
c.set_val(5)

def jacobian(f, at):
    j = np.zeros((len(f), len(at)))
    
    for i, ff in enumerate(f):
        for v in at:
            v.grad = 0
        ff.fw()
        ff.bw(1)
        grad = np.array([v.grad for v in at])
        j[i, :] = grad

    return j

print(jacobian(e, [a,b,c]))
