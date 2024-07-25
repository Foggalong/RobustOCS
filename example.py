#!/usr/bin/env python3

import robustocs

# key problem variables loaded from standard format txt files
sigma, mubar, omega, n, sires, dams, names = robustocs.load_problem(
    sigma_filename="examples/04/A04.txt",
    mu_filename="examples/04/EBV04.txt",
    omega_filename="examples/04/S04.txt",
    sex_filename="examples/04/SEX04.txt",
    issparse=True
)

# parameters with which to solve the problem
lam = 0.5
kap = 1

# computes the standard and robust genetic selection solutions
w_std, obj_std = robustocs.highs_standard_genetics(
    sigma, mubar, sires, dams, lam, n)
w_rbs, z_rbs, obj_rbs = robustocs.highs_robust_genetics(
    sigma, mubar, omega, sires, dams, lam, kap, n)

robustocs.print_compare_solutions(
    w_std, w_rbs, obj_std, obj_rbs, z2=z_rbs, name1="w_std", name2="w_rbs")


if not robustocs.check_uncertainty_constraint(z_rbs, w_rbs, omega, debug=True):
    raise ValueError

print("\nDone!")
