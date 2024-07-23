# -*- coding: utf-8 -*-
"""Utilities

This file contains additional utilities, such as for printing and comparing
portfolios produced by the solvers.
"""

import math                 # used for math.sqrt
import numpy as np          # defines matrix structures
import numpy.typing as npt  # variable typing definitions for NumPy
from scipy import sparse    # used for sparse matrix format

# controls what's imported on `from robustocs.utils import *`
__all__ = ["print_compare_solutions", "check_uncertainty_constraint"]


def print_compare_solutions(
    portfolio1: npt.NDArray[np.floating],
    portfolio2: npt.NDArray[np.floating],
    objective1: float,
    objective2: float,
    precision: int = 5,
    z1: float | None = None,
    z2: float | None = None,
    name1: str = "First",
    name2: str = "Second",
    tol: float | None = None
) -> None:
    """
    Given two solutions to a portfolio optimization problem (robust or non-
    robust) this prints a comparison of the two solutions to the terminal.

    Parameters
    ----------
    portfolio1 : ndarray
        The portfolio vector for the first solution to compare.
    portfolio2 : ndarray
        The portfolio vector for the second solution to compare.
    objective1 : float
        Objective value for the first solution vector.
    objective2 : float
        Objective value for the second solution vector.
    precision : int, optional
        The number of decimal places to display values to. Default is `5`.
    z1 : float or None, optional
        Variable associated with uncertainty for the first solution.
        Default is `None`, which corresponds to the non-robust problem.
    z2 : float or None, optional
        Variable associated with uncertainty for the second solution.
        Default is `None`, which corresponds to the non-robust problem.
    name1: str, optional
        Name to use for the first solution in output. Default is `"First"`.
    name2: str, optional
        Name to use for the second solution in output. Default is `"Second"`.
    tol: float, optional
        Tolerance below which not to show values. Default is None, all shown.

    Examples
    --------
    This function does not have a return value, but when used it produces a
    terminal output like the following:
    >>> print_compare_solutions(..., z2=z_rbs, name1="w_std", name2="w_rbs")
    i  w_std    w_rbs
    1  0.00000  0.38200
    2  0.00000  0.38200
    3  0.50000  0.11800
    4  0.50000  0.11800
    w_std objective: 1.87500
    w_rbs objective: 0.77684 (z = 0.37924)
    Maximum change: 0.38200
    Average change: 0.38200
    Minimum change: 0.38200
    """

    dimension = portfolio1.size
    order = len(str(dimension))

    # HACK header breaks if precision < 3 or len(problem1) != 5
    print(f"i{' '*(order-1)}  {name1}  {' '*(precision-3)}{name2}")
    for candidate in range(dimension):
        # if a tolerance given, skip if both values less than it
        if tol and portfolio1[candidate] < tol and portfolio2[candidate] < tol:
            continue

        print(
            f"{candidate+1:0{order}d}  "
            f"{portfolio1[candidate]:.{precision}f}  "
            f"{portfolio2[candidate]:.{precision}f}"
        )

    def obj_string(name: str, value: float, precision: int,
                   z: float | None = None) -> str:
        """Helper function which handles the optional z1 and z2 values"""
        obj_str = f"{name}: {value:.{precision}f}"
        return f"{obj_str} (z = {z:.{precision}f})" if z else f"{obj_str}"

    portfolio_abs_diff = np.abs(portfolio1-portfolio2)
    print(
        f"\n{obj_string(f'{name1} objective', objective1, precision, z1)}"
        f"\n{obj_string(f'{name2} objective', objective2, precision, z2)}"
        f"\nMaximum change: {max(portfolio_abs_diff):.{precision}f}"
        f"\nAverage change: {np.mean(portfolio_abs_diff):.{precision}f}"
        f"\nMinimum change: {min(portfolio_abs_diff):.{precision}f}"
    )


def check_uncertainty_constraint(
    z: float,
    w: npt.NDArray[np.floating],
    omega: npt.NDArray[np.floating] | sparse.sparray,
    tol: float = 1e-7,
    debug: bool = False
) -> bool:
    """
    Check the gap in the robust genetic selection uncertainty constraint.

    In our model for robust genetic selection we relax the objective term
    `sqrt(w.transpose()@Omega@w)` with `z >= sqrt(w.transpose()@Omega@w)`
    to keep the problem in a Gurobi-friendly form. While mathematically
    this should be equivalent, this function can be used to check how close
    z and the right hand side are.

    Parameters
    ----------
    z : float
        Auxiliary variable from a solution to the robust selection problem.
    w : ndarray
        Portfolio vector from a solution to the robust selection problem.
    omega : ndarray
        Covariance matrix for the distribution of expected values.
    tol : float, optional
        Tolerance with which to compare the two values. Default is `1e-8`.
    debug : float, optional
        Determines whether to print a comparison of the variables to the
        terminal. Default is `False`.

    Returns
    -------
    bool
        True if successful, False otherwise. As a side effect, it can print
        the two variables and their difference to the terminal (see below).

    Examples
    --------
    If using the debug output, some like the following will be printed to
    the terminal:

    >>> check_uncertainty_constraint(..., debug=True)
         z: 0.37923871642022844
    wT*Ω*w: 0.3792386953366983
      Diff: 2.1083530143961582e-08
    """

    # using np.sqrt returns an ndarray type object, even if it's only
    # got one entry, which in turn messes with the return typing.
    rhs = math.sqrt(w.transpose()@omega@w)

    if debug:
        print(
            f"\n     z: {z}"
            f"\nw'*Ω*w: {rhs}"
            f"\n  Diff: {z-rhs}"
        )

    return (rhs - z) < tol
