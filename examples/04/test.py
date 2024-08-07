#!/usr/bin/env python3

"""
While this script may be used as an example of using RobustOCS, it's primarily
intended for use as a test as part of the 'Check Build' GitHub action.
https://github.com/Foggalong/RobustOCS/actions/workflows/check-build.yml
"""

import numpy as np
import robustocs as rocs

# SETUP
# -----

# load in the problem variables (using underscore for those dropped)
sigma, mubar, omega, n, _, _, _ = rocs.load_problem(
    sigma_filename="A04.txt",
    mu_filename="EBV04.txt",
    omega_filename="S04.txt",
    issparse=True
)

sires = range(0, n, 2)
dams = range(1, n, 2)
lam = 0.5
kap = 1

# true solution to the standard and robust problems
true_std = np.array([0, 0, 0.5, 0.5])
true_rob = np.array([0.382, 0.382, 0.118, 0.118])

# tolerance for comparisons; NOTE this test uses a much stricter tolerance
# for the non-robust problem since both solvers should compute it exactly.
tol_std = 1e-7
tol_rob = 1e-3


# TESTS
# -----

w, obj = rocs.highs_standard_genetics(sigma, mubar, sires, dams, lam, n)
assert ((w - true_std) < tol_std).all(), "QP in HiGHS was incorrect"

w, obj = rocs.gurobi_standard_genetics(sigma, mubar, sires, dams, lam, n)
assert ((w - true_std) < tol_std).all(), "QP in Gurobi was incorrect"

w, z, obj = rocs.gurobi_robust_genetics(
    sigma, mubar, omega, sires, dams, lam, kap, n)
assert ((w - true_rob) < tol_rob).all(), "conic in Gurobi was incorrect"

w, z, obj = rocs.gurobi_robust_genetics_sqp(
    sigma, mubar, omega, sires, dams, lam, kap, n)
assert ((w - true_rob) < tol_rob).all(), "SQP in Gurobi was incorrect"

w, z, obj = rocs.highs_robust_genetics(
    sigma, mubar, omega, sires, dams, lam, kap, n)
assert ((w - true_rob) < tol_rob).all(), "conic in HiGHS was incorrect"

print("Success!")
