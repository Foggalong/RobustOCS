#!/usr/bin/env python3

import numpy as np
import robustocs as rocs

# SETUP
# -----

# load in the problem variables
sigma, mubar, omega, n = rocs.load_problem(
    "A1000.txt",
    "EBV1000.txt",
    "S1000.txt",
    issparse=True
)
sires = range(0, n, 2)
dams = range(1, n, 2)
lam = 0.5
kap = 1

# load in true solution to the standard and robust problems
true_std = np.loadtxt('solution_std.txt')
true_rob = np.loadtxt('solution_rob.txt')

# tolerance for comparisons
tol = 1e-3


# TESTS
# -----

w, obj = rocs.highs_standard_genetics(sigma, mubar, sires, dams, lam, n)
assert ((w - true_std) < tol).all(), "QP in HiGHS was incorrect"

w, z, obj = rocs.highs_robust_genetics(
    sigma, mubar, omega, sires, dams, lam, kap, n)
assert ((w - true_rob) < tol).all(), "conic in HiGHS was incorrect"

print("Success!")
