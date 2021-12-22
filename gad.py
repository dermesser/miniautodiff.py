import numpy as np

class Expression:
    def __init__(self, left, right):
        self.l = left
        self.r = right
        print(left, right)

        self.eval_l = None
        self.eval_r = None
    
    def fw(self, v):
        pass
    def bw(self):
        pass

    def __add__(self, other):
        return OpPlus(self, other)
    def __sub__(self, other):
        return OpMinus(self, other)
    def __neg__(self, other):
        return Num(name=self.name, id=self.id)
    def __mul__(self, other):
        return OpMult(self, other)
    def __truediv__(self, other):
        return OpDiv(self, other)
    def __pow__(self, other):
        return OpPow(self, other)

class Num(Expression):

    def __init__(self, i, n):
        self.i = i
        self.g = np.zeros(n)
        self.g[i] = 1

    def fw(self, v):
        return v[self.i]

    def bw(self):
        return self.g

def Const(Expression):
    def __init__(self, v):
        self.v = v

    def fw(self, v):
        return self.v

    def bw(self):
        return 0

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
        gl = self.l.bw()[:,None]
        gr = self.r.bw()[:,None]
        g = np.array([1/self.eval_r, -self.eval_l/self.eval_r**2])
        # Jacobian.T x gradient
        J = np.hstack((gl, gr))
        print(J.shape, g.shape)
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
            np.log(self.eval_l) * v])
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

class ADE:
    def __init__(self, n_variables):
        self.n = n_variables

    def vars(self):
        return [Num(i, self.n) for i in range(self.n)]

    def eval(self, expr, vals):
        v = np.array(vals)
        if type(expr) in [list, np.ndarray]:
            return [e.fw(v) for e in expr]
        return expr.fw(v)

    def grad(self, expr, at):
        value = self.eval(expr, at)
        if type(expr) in [list, np.ndarray]:
            # Calculate jacobian
            return value, np.vstack([e.bw() for e in expr])
        return value, expr.bw()

    def __enter__(self):
        return self.vars()

    def __exit__(self):
        pass

ade = ADE(3)
[x,y,z] = ade.vars()

f = x*y + y/z
g = sqrt(x**Const(2) + y**Const(2) + z**Const(2))

print(ade.eval(f, [1,2,3]))

print(ade.grad([f, g], [1,2,3]))
