"""
Copyright (c) 2021 Lewin Bormann

Simple automatic differentiation algorithm using "parallel forward mode".

First, an expression tree is built (and cached, when using @gradify). When evaluating
a gradient, the expression tree is recursively evaluated, propagating derivatives
from the bottom-up using Jacobian-gradient products.

See the end of this file for examples.
"""

import numpy as np
import time

class Expression:
    def __init__(self, left, right, ade=None):
        self.l = left
        self.r = right

        self.eval_l = None
        self.eval_r = None

        self.ade = ade or getattr(left, 'ade', None) or getattr(right, 'ade', None)
    
    def fw(self, v):
        pass
    def bw(self):
        pass

    def _autoconv(self, v):
        if isinstance(v, Expression):
            return v
        elif type(v) in (int, float):
            return self.ade.const(v)
        assert False, f'unknown type used in expression: {type(v)}'

    def __add__(self, other):
        return OpPlus(self, self._autoconv(other))
    def __sub__(self, other):
        return OpMinus(self, self._autoconv(other))
    def __neg__(self):
        return OpMinus(self.ade.const(0), self._autoconv(self))
    def __mul__(self, other):
        return OpMult(self, self._autoconv(other))
    def __truediv__(self, other):
        return OpDiv(self, self._autoconv(other))
    def __pow__(self, other):
        return OpPow(self, self._autoconv(other))

class Num(Expression):
    def __init__(self, i, n, ade=None):
        self.i = i
        self.g = np.zeros((n, 1))
        self.g[i] = 1
        self.ade = ade

    def fw(self, v):
        return v[self.i]

    def bw(self):
        return self.g

class Const(Expression):
    def __init__(self, n, v):
        self.v = v
        self.n = n

    def fw(self, v):
        return self.v

    def bw(self):
        return np.zeros(self.n)[:,None]

class OpPlus(Expression):
    def fw(self, v):
        return self.l.fw(v) + self.r.fw(v)

    def bw(self):
        return self.l.bw() + self.r.bw()

class OpMinus(Expression):
    def fw(self, v):
        return self.l.fw(v) - self.r.fw(v)

    def bw(self):
        return self.l.bw() - self.r.bw()

class OpMult(Expression):
    def fw(self, v):
        self.eval_l = self.l.fw(v)
        self.eval_r = self.r.fw(v)
        return self.eval_l * self.eval_r

    def bw(self):
        # effectively: jacobian-gradient product.
        return self.l.bw()*self.eval_r + self.r.bw()*self.eval_l

class OpDiv(Expression):
    def fw(self, v):
        self.eval_l = self.l.fw(v)
        self.eval_r = self.r.fw(v)
        return self.eval_l / self.eval_r

    def bw(self):
        gl = self.l.bw()
        gr = self.r.bw()
        g = np.array([1/self.eval_r, -self.eval_l/self.eval_r**2])[:,None]
        # Jacobian.T x gradient
        J = np.hstack((gl, gr))
        return J @ g

class OpPow(Expression):
    def fw(self, v):
        self.eval_l = self.l.fw(v)
        self.eval_r = self.r.fw(v)
        return self.eval_l**self.eval_r

    def bw(self):
        gl = self.l.bw()
        gr = self.r.bw()
        # gradient w.r.t. the two arguments to the power function.
        v = np.exp(self.eval_r * np.log(self.eval_l))
        thisgrad = np.array([
            self.eval_r/self.eval_l * v,
            np.log(self.eval_l) * v])[:,None]
        return np.hstack((gl, gr)) @ thisgrad

class UnaryExpression(Expression):
    def __init__(self, op, dop, e):
        self.l = e
        self.op = op
        self.dop = dop

    def fw(self, v):
        self.eval_l = self.l.fw(v)
        return self.op(self.eval_l)

    def bw(self):
        return self.l.bw() * self.dop(self.eval_l)

def exp(e):
    return UnaryExpression(np.exp, np.exp, e)

def sin(e):
    return UnaryExpression(np.sin, np.cos, e)

def cos(e):
    return UnaryExpression(np.cos, lambda x: -np.sin(x), e)

def sinh(e):
    return UnaryExpression(np.sinh, np.cosh, e)

def cosh(e):
    return UnaryExpression(np.cosh, np.sinh, e)

def tanh(e):
    return UnaryExpression(np.tanh, lambda x: 1-np.tanh(x)**2, e)

def sqrt(e):
    return UnaryExpression(np.sqrt, lambda x: 1/(2*np.sqrt(x)), e)

def log(e):
    return UnaryExpression(np.log, lambda x: 1/x, e)

class ADE:
    def __init__(self, n_variables):
        self.n = n_variables

    def vars(self):
        return [Num(i, self.n, ade=self) for i in range(self.n)]

    def const(self, v):
        return Const(self.n, v)

    def eval(self, expr, vals):
        v = np.array(vals)
        if type(expr) in [list, np.ndarray, tuple]:
            return [e.fw(v) for e in expr]
        return expr.fw(v)

    def grad(self, expr, at):
        value = self.eval(expr, at)
        if type(expr) in [list, np.ndarray, tuple]:
            # Calculate jacobian
            return value, np.vstack([e.bw().T for e in expr])
        return value, expr.bw().T

    def __enter__(self):
        return self.vars()

    def __exit__(self):
        pass

def gradify(f):
    """Decorate a function in order to automatically obtain its Jacobian.

    The wrapped function will return a tuple (value, jacobian).

    Additionally, the computational graph is cached automatically, accelerating repeated invocations.
    """
    c = {}
    def newf(*xs):
        if 'ade' not in c:
            ade = ADE(len(xs))
            newxs = ade.vars()
            expr = f(*newxs)
            c['expr'] = expr
            c['ade'] = ade
        return c['ade'].grad(c['expr'], xs)
    return newf

ade = ADE(3)
[x,y,z] = ade.vars()
v = np.array([x,y,z])

f = x*y + y/z
g = sin(x) + cos(y)

# Either use manually with function components...
print(ade.grad([f, g], [1,2,3]))

@gradify
def complex_calculation(x,y,z):
    a = x + y
    b = z - x
    c = a * b
    for i in range(4):
        c = c + a*b
    return c, a, b, a*b

@gradify
def pres_calculation(x1, x2, x3):
    return x1*x2 + exp(x1*x3)*cos(x2)

@gradify
def complex_calculation2(*x):
    y = np.array([x[i]+x[i+1]**2 for i in range(len(x)-1)])
    z = np.array([sqrt(log(e)) for e in y])
    return z

# ...or automatically using @gradify
# Equivalent to (without @gradify): print(ade.grad([complex_calculation(x,y,z)], [1,4,5]))
before = time.time_ns()
print(pres_calculation(1,4,5))
after = time.time_ns()
print((after-before)/1e9)

before = time.time_ns()
print(complex_calculation2(*list(range(1, 100, 2)))[1].shape)
after = time.time_ns()
print((after-before)/1e9)
