# -*- coding: utf-8 -*-
"""Defining Solvers

With the problems properly loaded into Numpy, this section contains functions
for solving those under various formulations and methods.
"""

import numpy as np          # defines matrix structures
import numpy.typing as npt  # variable typing definitions for NumPy
import gurobipy as gp       # Gurobi optimization interface (1)
from gurobipy import GRB    # Gurobi optimization interface (2)


def gurobi_standard_genetics(
    sigma: npt.NDArray[np.float64],
    mu: npt.NDArray[np.float64],
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,  # cannot be called `lambda`, that's reserved in Python
    dimension: int,
    upper_bound: npt.NDArray[np.float64] | float = 1.0,
    lower_bound: npt.NDArray[np.float64] | float = 0.0,
    time_limit: float | None = None,
    max_duality_gap: float | None = None,
    debug: bool = False
) -> tuple[npt.NDArray[np.float64], float]:
    """
    Solve the standard genetic selection problem using Gurobi.

    Given a standard genetic selection problem
    ```
        max_w w'mu - (lambda/2)*w'*sigma*w
        subject to lb <= w <= ub,
                   w_S*e_S = 1/2,
                   w_D*e_D = 1/2,
    ```
    this function uses Gurobi to find the optimum w and the objective for that
    portfolio. Additional parameters give control over long Gurobi can spend
    on the problem, to prevent indefinite hangs.

    Parameters
    ----------
    sigma : ndarray
        Covariance matrix of the candidates in the cohorts for selection.
    mu : ndarray
        Vector of expected returns for candidates in the cohorts for selection.
    sires : Any
        An object representing an index set for sires (male candidates) in the
        cohort. Type is not restricted.
    dams : Any
        An object representing an index set for dams (female candidates) in the
        cohort. Type is not restricted.
    lam : float
        Lambda value to optimize for, which controls the balance between risk
        and return. Lower values will give riskier portfolios, higher values
        more conservative ones.
    dimension : int
        Number of candidates in the cohort, i.e. the dimension of the problem.
    upper_bound : ndarray or float, optional
        Upper bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `1.0`.
    lower_bound : ndarray or float, optional
        Lower bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `0.0`.
    time_limit : float or None, optional
        Maximum amount of time in seconds to give Gurobi to solve the problem.
        Default value is `None`, i.e. no time limit.
    max_duality_gap : float or None, optional
        Maximum allowable duality gap to give Gurobi when solving the problem.
        Default value is `None`, i.e. do not allow any duality gap.
    debug : bool, optional
        Flag which controls both whether Gurobi prints its output to terminal
        and whether it saves the model file to the working directory (filename
        is hardcoded as `standard-opt.mps`). Default value is `False.

    Returns
    -------
    ndarray
        Portfolio vector which Gurobi has determined is a solution.
    float
        Value of the objective function for returned solution vector.
    """

    # create models for standard and robust genetic selection
    model = gp.Model("standard-genetics")

    # Gurobi spews all its output into the terminal by default, this restricts
    # that behaviour to only happen when the `debug` flag is used.
    if not debug:
        model.setParam('OutputFlag', 0)

    # integrating bounds within variable definitions is more efficient than
    # as a separate constraint, which Gurobi would convert to bounds anyway
    w = model.addMVar(shape=dimension, lb=lower_bound, ub=upper_bound,
                      vtype=GRB.CONTINUOUS, name="w")

    model.setObjective(
        # NOTE Gurobi introduces error if we use `np.inner(w, sigma@w)` here
        w.transpose()@mu - (lam/2)*w.transpose()@(sigma@w),
        GRB.MAXIMIZE
    )

    # set up the two sum-to-half constraints
    M = np.zeros((2, dimension), dtype=int)
    # define the M so that column i is [1;0] if i is a sire and [0;1] otherwise
    M[0, sires] = 1
    M[1, dams] = 1
    # define the right hand side of the constraint Mx = m
    m = np.full(2, 0.5)
    model.addConstr(M@w == m, name="sum-to-half")

    # optional controls to stop Gurobi taking too long
    if time_limit:
        model.setParam(GRB.Param.TimeLimit, time_limit)
    if max_duality_gap:
        model.setParam('MIPGap', max_duality_gap)

    # model file can be used externally for verification
    if debug:
        model.write("standard-opt.mps")

    model.optimize()
    return w.X, model.ObjVal


def gurobi_robust_genetics(
    sigma: npt.NDArray[np.float64],
    mubar: npt.NDArray[np.float64],
    omega: npt.NDArray[np.float64],
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,  # cannot be called `lambda`, that's reserved in Python
    kappa: float,
    dimension: int,
    upper_bound: npt.NDArray[np.float64] | float = 1.0,
    lower_bound: npt.NDArray[np.float64] | float = 0.0,
    time_limit: float | None = None,
    max_duality_gap: float | None = None,
    debug: bool = False
) -> tuple[npt.NDArray[np.float64], float, float]:
    """
    Solve the robust genetic selection problem using Gurobi.

    Given a robust genetic selection problem
    ```
        max_w (min_mu w'mu subject to mu in U) - (lambda/2)*w'*sigma*w
        subject to lb <= w <= ub,
                   w_S*e_S = 1/2,
                   w_D*e_D = 1/2,
    ```
    where U is a quadratic uncertainty set for mu~N(mubar, omega), this
    function uses Gurobi to find the optimum w and the objective for that
    portfolio. It first uses the KKT conditions to find exactly the solution
    to the inner problem, substitutes that into the outer problem, and then
    relaxes the uncertainty term into a constraint before solving.

    Additional parameters give control over long Gurobi can spend
    on the problem, to prevent indefinite hangs.

    Parameters
    ----------
    sigma : ndarray
        Covariance matrix of the candidates in the cohorts for selection.
    mubar : ndarray
        Vector of expected values of the expected returns for candidates in the
        cohort for selection.
    omega : ndarray
        Covariance matrix for expected returns for candidates in the cohort for
        selection.
    sires : Any
        An object representing an index set for sires (male candidates) in the
        cohort. Type is not restricted.
    dams : Any
        An object representing an index set for dams (female candidates) in the
        cohort. Type is not restricted.
    lam : float
        Lambda value to optimize for, which controls the balance between risk
        and return. Lower values will give riskier portfolios, higher values
        more conservative ones.
    kappa : float
        Kappa value to optimize for, which controls how resilient the solution
        must be to variation in expected values.
    dimension : int
        Number of candidates in the cohort, i.e. the dimension of the problem.
    upper_bound : ndarray or float, optional
        Upper bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `1.0`.
    lower_bound : ndarray or float, optional
        Lower bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `0.0`.
    time_limit : float or None, optional
        Maximum amount of time in seconds to give Gurobi to solve the problem.
        Default value is `None`, i.e. no time limit.
    max_duality_gap : float or None, optional
        Maximum allowable duality gap to give Gurobi when solving the problem.
        Default value is `None`, i.e. do not allow any duality gap.
    debug : bool, optional
        Flag which controls both whether Gurobi prints its output to terminal
        and whether it saves the model file to the working directory (filename
        is hardcoded as `standard-opt.mps`). Default value is `False.

    Returns
    -------
    ndarray
        Portfolio vector which Gurobi has determined is a solution.
    float
        Auxillary variable corresponding to uncertainty associated with the
        portfolio vector which Gurobi has determined is a solution.
    float
        Value of the objective function for returned solution vector.
    """

    # create models for standard and robust genetic selection
    model = gp.Model("robust-genetics")

    # Gurobi spews all its output into the terminal by default, this restricts
    # that behaviour to only happen when the `debug` flag is used.
    if not debug:
        model.setParam('OutputFlag', 0)

    # integrating bounds within variable definitions is more efficient than
    # as a separate constraint, which Gurobi would convert to bounds anyway
    w = model.addMVar(shape=dimension, lb=lower_bound, ub=upper_bound,
                      vtype=GRB.CONTINUOUS, name="w")
    z = model.addVar(lb=0.0, name="z")

    model.setObjective(
        # NOTE Gurobi introduces error if we use `np.inner(w, sigma@w)` here
        w.transpose()@mubar - (lam/2)*w.transpose()@(sigma@w) - kappa*z,
        GRB.MAXIMIZE
    )

    # set up the two sum-to-half constraints
    M = np.zeros((2, dimension), dtype=int)
    # define the M so that column i is [1;0] if i is a sire and [0;1] otherwise
    M[0, sires] = 1
    M[1, dams] = 1
    # define the right hand side of the constraint Mx = m
    m = np.full(2, 0.5, dtype=float)
    model.addConstr(M@w == m, name="sum-to-half")

    # conic constraint which comes from robust optimization
    model.addConstr(z**2 >= w.transpose()@omega@w, name="uncertainty")

    # optional controls to stop Gurobi taking too long
    if time_limit:
        model.setParam(GRB.Param.TimeLimit, time_limit)
    if max_duality_gap:
        model.setParam('MIPGap', max_duality_gap)

    # model file can be used externally for verification
    if debug:
        model.write("robust-opt.mps")

    model.optimize()
    return w.X, z.X, model.ObjVal
