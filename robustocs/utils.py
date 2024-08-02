# -*- coding: utf-8 -*-
"""Utilities

This file contains additional utilities, such as for printing and comparing
portfolios produced by the solvers.
"""

import math                 # used for math.sqrt
import numpy as np          # defines matrix structures
import numpy.typing as npt  # variable typing definitions for NumPy
from scipy import sparse    # used for sparse matrix format

# local imports used in solveROCS
from . import solvers
from . import loaders

# controls what's imported on `from robustocs.utils import *`
__all__ = [
    "solveROCS",
    "expected_genetic_merit",
    "group_coancestry",
    "group_coancestry_fast",
    "print_compare_solutions",
    "check_uncertainty_constraint"
]


# WRAPPER UTILITY
# The solveROCS combines all of the features in this module into
# a single interface, handling everything from loading the data
# from file through to analysing the solution found. While it's
# convenient for single calls, repeated calls should probably use
# the more granular functions provided in `solvers`.

def solveROCS(
    sigma_filename: str,
    mu_filename: str,
    sex_filename: str,
    lam: float,
    omega_filename: str | None = None,
    kappa: float | None = None,
    upper_bound: npt.ArrayLike | float = 1.0,
    lower_bound: npt.ArrayLike | float = 0.0,
    method: str = 'standard',
    solver: str = 'highs',
    issparse: bool = False,
    time_limit: float | None = None,
    max_iterations: int = 1000,
    robust_gap_tol: float = 1e-7,
    model_output: str = '',
    debug: bool = False
):
    """
    A convenient wrapper which accesses RobustOCS through a single
    interface, handling everything from loading the data from file
    through to analysing the solution found. Best suited for single
    calls, for more complicated work see the functions in `solvers`.

    Parameters
    ----------
    sigma_filename : str
        Filename for a file which encodes sigma, the relationship matrix
        for the cohort. This can be either in sparse matrix coordinate
        format (COO) or pedigree format.
    mu_filename : str
        Filename for a file which encodes mu, the expected breeding values
        of candidates in the cohort. For robust optimization this is mubar,
        the expected values of the expected returns for candidates in the
        cohort. The format should be a single value per line.
    sex_filename: str
        Filename for a file which encodes sex data, i.e. whether candidates
        in the cohort are male (sires) or female (dams). The format is a space
        separated values file with the first column being a label for the
        and the second being 'M' if the candidate is male and 'F' if female.
    lam: float,
        Lambda value to optimize for, which controls the balance between risk
        and return. Lower values will give riskier portfolios, higher values
        more conservative ones.
    omega_filename : str, optional
        Filename for a file which encodes omega, the covariance matrix for
        expected returns for candidates in the cohort for selection. This is
        specific to robust optimization and will be ignored if doing standard
        (non-robust) optimization. File should be in sparse coordinate format.
    kappa : float, optional
        Kappa value to optimize for, which controls how resilient the solution
        must be to variation in expected values. This is specific to robust
        optimization and will be ignored if doing standard optimization.
    upper_bound : ndarray, list, or float, optional
        Upper bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `1.0`.
    lower_bound : ndarray, list, or float, optional
        Lower bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `0.0`.
    method : str, optional
        String which specifies whether to carry out 'standard' or 'robust'
        optimization. Default value is `standard`.
    solver : str, optional
        String which specifies whether to use HiGHS or Gurobi as the solver.
        Value can either be `highs` (default) or `gurobi`.
    issparse : bool, optional
        Signifies whether sigma and omega should be loaded primarily as a dense
        matrix or a sparse one. Default value is `False` (i.e. as dense).
    time_limit : float, optional
        Maximum amount of time in seconds to give the underlying solver to find
        a solution. Default value is `None`, i.e. no time limit.
    max_iterations : int, optional
        Maximum number of iterations that can be taken in solving the robust
        problem using SQP. This is specific to robust optimization and will be
        ignored if doing standard optimization. Default value is `1000`.
    robust_gap_tol : float, optional
        Tolerance when checking whether an approximating constraint is active
        and whether the SQP overall has converged. This is specific to robust
        optimization and will be ignored if doing standard optimization.
        Default value is 10^-7.
    model_output : str, optional
        Flag which controls whether the solver saves the model file to the
        working directory. If given, the string is used as the file name,
        'str.mps', Default value is the empty string, i.e. it doesn't save.
    debug : bool, optional
        Flag which controls whether the solver prints its output to terminal.
        Default value is `False`.
    """

    # INPUT VALIDATION
    # ----------------

    # STRINGS
    # validate key problem variables were provided as strings
    file_parameters = {
        "sigma": sigma_filename,
        "mu": mu_filename,
        "sexes": sex_filename,
        "model output": model_output
    }

    # omega optional to provide, but needs checking for robust problems
    if omega_filename is not None:
        file_parameters["omega"] = omega_filename

    # check variable type is a string
    for name, value in file_parameters.items():
        if not isinstance(value, str):
            raise ValueError(f"filename must be given for {name} data")

    # POSITIVE FLOATS
    # validate variables required to be greater than or equal to zero
    positive_parameters = {"lambda": lam}

    # kappa optional to provide, but needs checking for robust problems
    if kappa is not None:
        positive_parameters["kappa"] = kappa

    # check possible to coerce variable into a float which is positive
    for name, value in positive_parameters.items():
        try:
            if float(lam) < 0:
                raise ValueError
        except ValueError:
            raise ValueError(f"{name} parameter must be a positive number")

    # X IN [0,1]
    # validate bounds are floats or arrays in interval [0,1]
    for name, value in {"upper": upper_bound, "lower": lower_bound}.items():
        if isinstance(value, float):
            if value < 0 or value > 1:
                raise ValueError(f"{name} bound must be between 0 and 1")
        elif isinstance(value, npt.ArrayLike):
            if any(value < 0 or value > 1):
                raise ValueError(f"every {name} bound must be between 0 and 1")
        else:
            raise ValueError(f"{name} bound must be a number or array")

    # SOLVER & METHOD
    # validate the method and solver were specified
    if solver not in ('highs', 'gurobi'):
        raise ValueError("solver isn't valid, must be 'gurobi' or 'highs'")
    if method not in ('standard', 'robust'):
        raise ValueError("method isn't valid, must be 'standard' or 'robust'")
    # if solving ROCS, validate additional necessary variables are here
    if (method == 'robust') and kappa is None or omega_filename is None:
        raise ValueError("robust optimization requires 'kappa' and 'omega'")

    # BOOLEANS
    # validate sparsity and debug status were specified using booleans
    for name, value in {"issparse": issparse, "debug": debug}.items():
        if not isinstance(value, bool):
            raise ValueError(f"'{name}' must be 'True' or 'False' boolean")

    # STRICTLY POSITIVE FLOATS
    # validate time_limit is strictly positive
    for name, value in {"time limit": time_limit}.items():
        try:
            if value is not None and float(value) <= 0:
                raise ValueError
        except ValueError:
            raise ValueError(f"{name} must be a positive number")

    # LOADING & SOLVING
    # -----------------

    # try to load the variables from file into NumPy
    sigma, mu, omega, n, sires, dams, _ = loaders.load_problem(
        sigma_filename, mu_filename, omega_filename, sex_filename,
        issparse=True
    )

    if method == 'standard':
        # Gurobi and HiGHs have the same input variables for standard OCS
        rocs_solver = getattr(solvers, f"{solver}_standard_genetics")
        portfolio, objective = rocs_solver(
            sigma, mu, sires, dams, lam, n, upper_bound, lower_bound,
            time_limit, model_output, debug
        )
    elif solver == 'highs':  # and method is robust
        portfolio, z_value, objective = solvers.highs_robust_genetics(
            sigma, mu, omega, sires, dams, lam, kappa, n, upper_bound,
            lower_bound, time_limit, max_iterations, robust_gap_tol,
            model_output, debug
        )
    else:  # solver is gurobi, method is robust
        portfolio, z_value, objective = solvers.gurobi_robust_genetics(
            sigma, mu, omega, sires, dams, lam, kappa, n, upper_bound,
            lower_bound, time_limit, model_output, debug
        )

    # CHECKING
    # --------

    # check the relaxation of the robust term was reasonable
    if method == 'robust':
        if not check_uncertainty_constraint(z_value, portfolio, omega,
                                            robust_gap_tol, debug):
            raise ValueError("approximation method wasn't within tolerance")

    # quicker to compute useful metrics rather than after the fact
    EGM = expected_genetic_merit(portfolio, mu)
    GCA = group_coancestry_fast(mu, lam, objective, portfolio)

    return portfolio, objective, EGM, GCA


# ANALYSIS UTILITIES
# These functions are useful for analysing particular solutions

def expected_genetic_merit(w: npt.ArrayLike, mu: npt.ArrayLike) -> np.floating:
    """
    Shortcut function for computing the expected genetic merit (w'μ) of a
    particular selection of candidates (w).

    Parameters
    ----------
    w : ndarray
        Portfolio vector representing a particular selection.
    mu : ndarray
        Vector of expected breeding values for the cohort.

    Returns
    -------
    float
        The expected genetic merit of the selection.
    """

    return w.transpose() @ mu


def group_coancestry(w: npt.ArrayLike, sigma: npt.ArrayLike) -> np.floating:
    """
    Shortcut function for computing the group co-ancestry (w'Σw) of a
    particular selection of candidates (w).

    Parameters
    ----------
    w : ndarray
        Portfolio vector representing a particular selection.
    sigma : ndarray
        Relationship matrix for the cohort.

    Returns
    -------
    float
        The expected genetic merit of the selection.
    """

    return w.transpose() @ sigma @ w


def group_coancestry_fast(
    mu: npt.NDArray[np.floating],
    lam: float,  # cannot be called `lambda`, that's reserved in Python
    obj: float,
    w: npt.ArrayLike,
    kappa: float = 0,
    z: np.floating = 0
) -> np.floating:
    """
    Quicker way to compute the group co-ancestry (w'Σw) of a particular
    selection of candidates (w) in situations where the OCS problem has
    already been solved. While computing the matrix product directly is
    O(n²), using the objective value we can do it in O(n).

    Parameters
    ----------
    mu : ndarray
        Vector of expected returns for candidates in the cohorts for selection.
    lam : float
        Lambda parameter used when solving the OCS problem.
    obj : float
        Value of the objective function for returned solution vector.
    w: ndarray
        Portfolio vector which a solver determined was part of a solution.
    kappa : float, optional
        Kappa parameter used when solving the OCS problem. If not provided
        then assumes that it's solving the non-robust OCS problem. Default
        value is zero.
    z : float, optional
        Auxiliary variable which a solver determined was part of a solution.
        If not provided then assumes that it's solving the non-robust OCS
        problem. Default value is zero.

    Returns
    -------
    float
        The expected genetic merit of the selection.
    """

    return (2/lam) * (w.transpose()@mu - kappa*z - obj)


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


# CHECKING UTILITIES
# These functions are useful for checking the validity particular solutions

def check_uncertainty_constraint(
    z: float,
    w: npt.NDArray[np.floating],
    omega: npt.NDArray[np.floating] | sparse.spmatrix,
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
