"""
Copyright (c) 2021 Lewin Bormann

Reverse-mode automatic differentiation algorithm.

First an expression tree is built. When evaluating a gradient, the initial
seed gradient is propagated backwards through the expression tree. Finally,
the accumulated chain-rule products are summed at each input variable node
and read out by the "driver function" (jacobian()).

Simple backpropagation; stateful, naive approach.
"""

import numpy as np
import time

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

    def _autoconv(self, e):
        if isinstance(e, Expression):
            return e
        if type(e) in (int, float):
            return Const(e)
        return e

    def __add__(self, other):
        return OpPlus(self, self._autoconv(other))
    def __sub__(self, other):
        return OpMinus(self, self._autoconv(other))
    def __neg__(self):
        return (self._autoconv(0)-self)
    def __mul__(self, other):
        return OpMult(self, self._autoconv(other))
    def __truediv__(self, other):
        return OpDiv(self, self._autoconv(other))

class Const(Expression):
    def __init__(self, val):
        self.v = val

    def fw(self):
        return self.v

    def bw(self,grad):
        pass

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

def sqrt(e):
    return UnaryExpression(np.sqrt, lambda x: 1/(2*np.sqrt(x)), e)

def log(e):
    return UnaryExpression(np.log, lambda x: 1/x, e)

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

def jacobian(f, at):
    """Returns function value and jacobian."""
    if type(at) not in (tuple, list, np.ndarray):
        at = [at]
    if type(f) not in (tuple, list, np.ndarray):
        f = [f]
    j = np.zeros((len(f), len(at)))
    val = np.zeros(len(f))
    for i, ff in enumerate(f):
        for v in at:
            v.grad = 0
        val[i] = ff.fw()
        ff.bw(1)
        grad = np.array([v.grad for v in at])
        j[i, :] = grad

    return val, j

def gradify(f):
    c = {}
    def newf(*newxs):
        if 'variables' not in c:
            variables = [Num() for x in newxs]
            expr = f(*variables)
            c['variables'] = variables
            c['expr'] = expr
        else:
            variables = c['variables']
            expr = c['expr']
        for v, x in zip(variables, newxs):
            v.set_val(x)
        return jacobian(expr, variables)
    return newf

a = Num('a')
b = Num('b')
c = Num('c')

e = [a * b * sin(c),
        a+c,
        a * cos(b)]

a.set_val(3)
b.set_val(4)
c.set_val(5)

print(jacobian(e, [a,b,c]))

@gradify
def complex_calculation(x,y,z):
    a = x + y
    b = x - z
    c = a * b
    for i in range(4):
        c = c + a*b
    return c, a, b, a*b

@gradify
def complex_calculation2(*x):
    y = np.array([x[i]+x[i+1] for i in range(len(x)-1)])
    z = np.array([sqrt(log(e)) for e in y])
    return z

@gradify
def pres_calculation(x1, x2, x3):
    return x1*x2 + exp(x1*x3)*cos(x2)

before = time.time_ns()
print(pres_calculation(1,4,5))
after = time.time_ns()
print((after-before)/1e9)

before = time.time_ns()
print(complex_calculation(2,8,10))
after = time.time_ns()
print((after-before)/1e9)

before = time.time_ns()
print(complex_calculation2(*list(range(1, 400, 2)))[1].shape)
after = time.time_ns()
print((after-before)/1e9)
