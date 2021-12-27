import autograd.numpy as np
from autograd import jacobian

Nsteps = 1000

def LV(arg):
    N1, N2 = arg[0:2]
    for i in range(1, Nsteps):
        _N1 = N1
        N1 += N1 * (arg[2]-arg[4]*N2)
        N2 += -N2 * (arg[3]-arg[5]*_N1)

    return np.array([N1, N2])

(N1, N2, eps1, eps2, gam1, gam2) = (120., 60., 7e-3, 4e-2, 5e-4, 5e-4)

print(LV([N1, N2, eps1, eps2, gam1, gam2]))
print(jacobian(LV)(np.array([N1, N2, eps1, eps2, gam1, gam2])))




