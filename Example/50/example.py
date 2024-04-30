#!/usr/bin/env python3

import sys; sys.path.insert(1, '../../')  # HACK pre-module workaround
import alphargs

# key problem variables loaded from standard format txt files
sigma, mubar, omega, n = alphargs.load_problem(
    "A50.txt",
    "EBV50.txt",
    "S50.txt"
)

# NOTE this trick of handling sex data is specific to the initial simulation
# data which is structured so that candidates alternate between sires (which
# have even indices) and dams (which have odd indices).
sires = range(0, n, 2)
dams = range(1, n, 2)

lam = 0.5
kap = 1

# computes the standard and robust genetic selection solutions
w_std, obj_std = alphargs.gurobi_standard_genetics(
    sigma, mubar, sires, dams, lam, n)
w_rbs, z_rbs, obj_rbs = alphargs.gurobi_robust_genetics(
    sigma, mubar, omega, sires, dams, lam, kap, n)

alphargs.print_compare_solutions(
    w_std, w_rbs, obj_std, obj_rbs, z2=z_rbs, name1="w_std", name2="w_rbs")


print("\nDone!")
