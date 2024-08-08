# -*- coding: utf-8 -*-
"""
RobustOCS provides functions for loading and solving robust genetic selection
problems. It uses file structures from the AlphaGenes family, NumPy and SciPy
for internal data structures, and depends on Gurobi or HiGHS for its solvers.

Documentation is available in the docstrings and online at
https://github.com/Foggalong/RobustOCS/wiki
"""

from .loaders import load_ped, load_problem
from .pedigree import makeA, make_invA
from .solvers import (
    gurobi_standard_genetics, gurobi_robust_genetics,
    gurobi_robust_genetics_sqp, gurobi_robust_genetics_conic,
    highs_standard_genetics, highs_robust_genetics,
    highs_robust_genetics_sqp
)
from .utils import print_compare_solutions, check_uncertainty_constraint

__author__ = "Josh Fogg"
__version__ = "0.2.0"

# controls what's imported on `from robustocs import *`
__all__ = [
    # from loaders.py
    "load_ped", "load_problem",
    # from pedigree.py
    "makeA", "make_invA",
    # from solvers.py
    "gurobi_standard_genetics", "gurobi_robust_genetics",
    "gurobi_robust_genetics_sqp", "gurobi_robust_genetics_conic",
    "highs_standard_genetics", "highs_robust_genetics",
    "highs_robust_genetics_sqp",
    # from utils.py
    "solveROCS", "sparsity", "eigmax", "cond", "expected_genetic_merit",
    "group_coancestry", "group_coancestry_fast", "print_compare_solutions",
    "check_uncertainty_constraint"
]
