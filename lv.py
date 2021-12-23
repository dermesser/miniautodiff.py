"""Example: Get derivative of iterative Lotka-Volterra differential equation,
using Tensorflow for AD.

"""
import numpy as np
import tensorflow as tf


def LV(N1, N2, eps1, eps2, gam1, gam2):
    dt = tf.constant(1.)
    states = [(N1, N2)]
    for i in range(1, 13):
        states.append((states[i-1][0] + (states[i-1][0] * (eps1-gam1*states[i-1][1])) * dt, states[i-1][1] - states[i-1][1] * (eps2-gam2*states[i-1][0])) * dt)
    return states[-1]

with tf.GradientTape(persistent=True) as tape:
    (N1, N2, eps1, eps2, gam1, gam2) = arg = [tf.Variable(x) for x in [120., 60., 7e-3, 4e-2, 5e-4, 5e-4]]
    (fN1, fN2) = LV(N1, N2, eps1, eps2, gam1, gam2)

print(tape.jacobian(fN1, arg))
print(tape.jacobian(fN2, arg))
