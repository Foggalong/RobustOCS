# -*- coding: utf-8 -*-
"""
RobustOCS provides functions for loading and solving robust genetic selection
problems. It uses file structures from the AlphaGenes family, NumPy and SciPy
for internal data structures, and depends on Gurobi or HiGHS for its solvers.
"""

from .loaders import *
from .solvers import *
from .utils import *

__author__ = "Josh Fogg"
__version__ = "0.0.1"

# controls what's imported on `from robustocs import *`
__all__ = [
    # from loaders.py
    "load_ped", "makeA", "load_problem",
    # from solvers.py
    "gurobi_standard_genetics", "gurobi_robust_genetics",
    "gurobi_robust_genetics_sqp", "gurobi_robust_genetics_conic",
    "highs_standard_genetics", "highs_robust_genetics",
    "highs_robust_genetics_sqp",
    # from utils.py
    "print_compare_solutions", "check_uncertainty_constraint"
]
