#!/usr/bin/env python3

import sys; sys.path.insert(1, '../../')  # HACK pre-module workaround
import alphargs
import time

# key problem variables loaded from standard format txt files
sigma, mubar, omega, n = alphargs.load_problem(
    "A50.txt",
    "EBV50.txt",
    "S50.txt",
    issparse=True
)

# NOTE this trick of handling sex data is specific to the initial simulation
# data which is structured so that candidates alternate between sires (which
# have even indices) and dams (which have odd indices).
sires = range(0, n, 2)
dams = range(1, n, 2)

lam = 0.5
kap = 1


# Gurobi using direct optimization
t0 = time.time()
w_grb, z_grb, obj_grb = alphargs.gurobi_robust_genetics(
    sigma, mubar, omega, sires, dams, lam, kap, n)
t1 = time.time()
print(f"Gurobi took {t1-t0:.5f} seconds (direct)")

# Gurobi using sequential quadratic programming
t0 = time.time()
w_grb, z_grb, obj_grb = alphargs.gurobi_robust_genetics_sqp(
    sigma, mubar, omega, sires, dams, lam, kap, n)
t1 = time.time()
print(f"Gurobi took {t1-t0:.5f} seconds (SQP)")

# HiGHS using sequential quadratic programming
t0 = time.time()
w_hig, z_hig, obj_hig = alphargs.highs_robust_genetics_sqp(
    sigma, mubar, omega, sires, dams, lam, kap, n)
t1 = time.time()
print(f"HiGHS took  {t1-t0:.5f} seconds (SQP)")


print("\nSQP Methods:")
alphargs.print_compare_solutions(
    w_grb, w_hig, obj_grb, obj_hig, z1=z_grb, z2=z_hig,
    name1="Gurobi", name2="HiGHS ", tol=1e-7
)

print("\nDone!")
