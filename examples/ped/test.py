#!/usr/bin/env python3

import numpy as np
import robustocs as rocs
from time import time

"""
While this script may be used as an example of using RobustOCS, it's primarily
intended for use as a test as part of the 'Test Pedigree' GitHub action.
https://github.com/Foggalong/RobustOCS/actions/workflows/check-pedigree.yml
"""


ped = rocs.load_ped("example.ped")


def printMatrix(matrix, description="ans =", precision=3):
    """Quick function for nicely printing a matrix"""
    print(f"{description}\n", np.round(matrix, precision))


# test constructing a WNRM
# ========================

tic = time()
A = rocs.makeA(ped)
printMatrix(A, f"\n{time()-tic:.2E} secs for makeA")

A_true = np.array([
    [1., 0., 0.5, 0.5, 0.5, 0.75, 0.625],
    [0., 1., 0., 0.5, 0.25, 0.25, 0.25],
    [0.5, 0., 1., 0.25, 0.625, 0.375, 0.5],
    [0.5, 0.5, 0.25, 1., 0.625, 0.75, 0.6875],
    [0.5, 0.25, 0.625, 0.625, 1.125, 0.5625, 0.84375],
    [0.75, 0.25, 0.375, 0.75, 0.5625, 1.25, 0.90625],
    [0.625, 0.25, 0.5, 0.6875, 0.84375, 0.90625, 1.28125]
])

# check makeA produced correct answer
assert np.all(np.abs(A-A_true) < 1e-7)


# test inverting a WNRM
# =====================

tic = time()
B = rocs.makeA(ped)
invA1 = np.linalg.inv(B)
printMatrix(invA1, f"\n{time()-tic:.2E} secs for np.linalg.inv")

tic = time()
invA2 = rocs.make_invA(ped)
printMatrix(invA2, f"\n{time()-tic:.2E} secs for make_invA")

tic = time()
invA3 = rocs.pedigree.make_invA_decomposition(ped)
printMatrix(invA3, f"\n{time()-tic:.2E} secs for make_invA_decomposition")

# all three inverses should be mutually within tolerance of each other
assert np.all(np.abs(invA1-invA2) < 1e-7)
assert np.all(np.abs(invA2-invA3) < 1e-7)
assert np.all(np.abs(invA1-invA3) < 1e-7)
