#!/usr/bin/env bash

LOOPS=3  # run the command this many times and take the average
REPS=5   # take the best average time from this many averages

# code block which sets up the problem, same across all three
SETUP='
import sys; sys.path.insert(1, "../../")
import alphargs as A

sigma, mubar, omega, n = A.load_problem(
    "A1000.txt",
    "EBV1000.txt",
    "S1000.txt",
    issparse=True
)

sires = range(0, n, 2)
dams = range(1, n, 2)

lam = 0.5
kap = 1'

# array of solver commands to time
SOLVERS=(
    'A.gurobi_standard_genetics(sigma, mubar, sires, dams, lam, n)'
    'A.highs_standard_genetics(sigma, mubar, sires, dams, lam, n)'
    'A.gurobi_robust_genetics(sigma, mubar, omega, sires, dams, lam, kap, n)'
    'A.gurobi_robust_genetics_sqp(sigma, mubar, omega, sires, dams, lam, kap, n)'
    'A.highs_robust_genetics_sqp(sigma, mubar, omega, sires, dams, lam, kap, n)'
)

# run timing code for each solver command
for f in "${SOLVERS[@]}"; do
    echo -e "\033[1mTesting ${f}\033[0m"  # print command in bold
    python3 -m timeit -v -s "$SETUP" -n $LOOPS -r $REPS -u "sec" "$f"
    echo -e "\n"  # padding for readability
done
