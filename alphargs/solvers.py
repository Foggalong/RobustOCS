# -*- coding: utf-8 -*-
"""Defining Solvers

With the problems properly loaded into Numpy, this section contains functions
for solving those under various formulations and methods.
"""

import numpy as np          # defines matrix structures
import numpy.typing as npt  # variable typing definitions for NumPy
import gurobipy as gp       # Gurobi optimization interface
import highspy              # HiGHS optimization interface
from math import sqrt       # used within the robust constraint
from scipy import sparse    # used for sparse matrix format

# controls what's imported on `from alphargs.solvers import *`
__all__ = [
    "gurobi_standard_genetics",
    "gurobi_robust_genetics",
    "gurobi_robust_genetics_sqp"
]


def gurobi_standard_genetics(
    sigma: npt.NDArray[np.float64] | sparse.spmatrix,
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
    sigma : ndarray or spmatrix
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
                      vtype=gp.GRB.CONTINUOUS, name="w")

    model.setObjective(
        # NOTE Gurobi introduces error if we use `np.inner(w, sigma@w)` here
        w.transpose()@mu - (lam/2)*w.transpose()@(sigma@w),
        gp.GRB.MAXIMIZE
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
        model.setParam(gp.GRB.Param.TimeLimit, time_limit)
    if max_duality_gap:
        model.setParam('MIPGap', max_duality_gap)

    # model file can be used externally for verification
    if debug:
        model.write("standard-opt.mps")

    model.optimize()
    return np.array(w.X), model.ObjVal  # HACK np.array avoids issue #9


def gurobi_robust_genetics(
    sigma: npt.NDArray[np.float64] | sparse.spmatrix,
    mubar: npt.NDArray[np.float64],
    omega: npt.NDArray[np.float64] | sparse.spmatrix,
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
    sigma : ndarray or spmatrix
        Covariance matrix of the candidates in the cohorts for selection.
    mubar : ndarray
        Vector of expected values of the expected returns for candidates in the
        cohort for selection.
    omega : ndarray or spmatrix
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
                      vtype=gp.GRB.CONTINUOUS, name="w")
    z = model.addVar(lb=0.0, name="z")

    model.setObjective(
        # NOTE Gurobi introduces error if we use `np.inner(w, sigma@w)` here
        w.transpose()@mubar - (lam/2)*w.transpose()@(sigma@w) - kappa*z,
        gp.GRB.MAXIMIZE
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
        model.setParam(gp.GRB.Param.TimeLimit, time_limit)
    if max_duality_gap:
        model.setParam('MIPGap', max_duality_gap)

    # model file can be used externally for verification
    if debug:
        model.write("robust-opt.mps")

    model.optimize()
    return np.array(w.X), z.X, model.ObjVal  # HACK np.array avoids issue #9


def gurobi_robust_genetics_sqp(
    sigma: npt.NDArray[np.float64] | sparse.spmatrix,
    mubar: npt.NDArray[np.float64],
    omega: npt.NDArray[np.float64] | sparse.spmatrix,
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,  # cannot be called `lambda`, that's reserved in Python
    kappa: float,
    dimension: int,
    upper_bound: npt.NDArray[np.float64] | float = 1.0,
    lower_bound: npt.NDArray[np.float64] | float = 0.0,
    time_limit: float | None = None,
    max_duality_gap: float | None = None,
    max_iterations: int = 1000,
    robust_gap_tol: float = 1e-8,
    debug: bool = False
) -> tuple[npt.NDArray[np.float64], float, float]:
    """
    Solve the robust genetic selection problem using SQP in Gurobi.

    Given a robust genetic selection problem
    ```
        max_w (min_mu w'mu subject to mu in U) - (lambda/2)*w'*sigma*w
        subject to lb <= w <= ub,
                   w_S*e_S = 1/2,
                   w_D*e_D = 1/2,
    ```
    where U is a quadratic uncertainty set for mu~N(mubar, omega), this
    function uses Gurobi to find the optimum w and the objective for that
    portfolio. It does this using sequential quadratic programming (SQP),
    approximating the conic constraint associated with robustness using
    a series of linear constraints.

    Additional parameters give control over long Gurobi can spend
    on the problem, to prevent indefinite hangs.

    Parameters
    ----------
    sigma : ndarray or spmatrix
        Covariance matrix of the candidates in the cohorts for selection.
    mubar : ndarray
        Vector of expected values of the expected returns for candidates in the
        cohort for selection.
    omega : ndarray or spmatrix
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
        Maximum amount of time in seconds to give Gurobi to solve sub-problem.
        Note it does *not* constrain how much time is taken overall. Default
        value is `None`, i.e. no time limit.
    max_duality_gap : float or None, optional
        Maximum allowable duality gap to give Gurobi when solving the problem.
        Default value is `None`, i.e. do not allow any duality gap.
    max_iterations : int, optional
        Maximum number of iterations that can be taken in solving the problem,
        i.e. the maximum number of constraints to use to approximate the conic
        constraint. Default value is `1000`.
    robust_gap_tol : float, optional
        Tolerance when checking whether an approximating constraint is active
        and whether the SQP overall has converged. Default value is 10^-8.
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
    model = gp.Model("robust-genetics-sqp")

    # Gurobi spews all its output into the terminal by default, this restricts
    # that behaviour to only happen when the `debug` flag is used.
    if not debug:
        model.setParam('OutputFlag', 0)

    # integrating bounds within variable definitions is more efficient than
    # as a separate constraint, which Gurobi would convert to bounds anyway
    w = model.addMVar(shape=dimension, lb=lower_bound, ub=upper_bound,
                      vtype=gp.GRB.CONTINUOUS, name="w")
    z = model.addVar(lb=0.0, name="z")

    model.setObjective(
        # NOTE Gurobi introduces error if we use `np.inner(w, sigma@w)` here
        w.transpose()@mubar - (lam/2)*w.transpose()@(sigma@w) - kappa*z,
        gp.GRB.MAXIMIZE
    )

    # set up the two sum-to-half constraints
    M = np.zeros((2, dimension), dtype=int)
    # define the M so that column i is [1;0] if i is a sire and [0;1] otherwise
    M[0, sires] = 1
    M[1, dams] = 1
    # define the right hand side of the constraint Mx = m
    m = np.full(2, 0.5, dtype=float)
    model.addConstr(M@w == m, name="sum-to-half")

    # optional controls to stop Gurobi taking too long
    if time_limit:
        model.setParam(gp.GRB.Param.TimeLimit, time_limit)
    if max_duality_gap:
        model.setParam('MIPGap', max_duality_gap)

    for i in range(max_iterations):
        # optimization of the model, print weights and objective
        model.optimize()
        if debug:
            print(f"{i}: {w.X}, {model.ObjVal:g}")

        # assess which constraints are currently active
        active_const: bool = False
        for c in model.getConstrs():
            if abs(c.Slack) > robust_gap_tol:
                active_const = True
                if debug:
                    print(f"{c.ConstrName} active, slack {c.Slack:g}")
        if debug and not active_const:
            print("No active constraints!")

        # z coefficient for the new constraint
        w_star: npt.NDArray[np.float64] = np.array(w.X)
        alpha: float = sqrt(w_star.transpose()@omega@w_star)

        # if gap between z and w'Omega w has converged, done
        if abs(z.X - alpha) < robust_gap_tol:
            break

        # add a new plane to the approximation of the uncertainty cone
        model.addConstr(alpha*z >= w_star.transpose()@omega@w, name=f"P{i}")

    # model file can be used externally for verification
    if debug:
        model.write("robust-sqp-opt.mps")

    return np.array(w.X), z.X, model.ObjVal  # HACK np.array avoids issue #9


def highspy_robust_genetics_sqp(
    sigma_start: npt.NDArray[np.float64],
    sigma_index: npt.NDArray[np.float64],
    sigma_value: npt.NDArray[np.float64],
    mubar: npt.NDArray[np.float64],
    omega_start: npt.NDArray[np.float64],
    omega_index: npt.NDArray[np.float64],
    omega_value: npt.NDArray[np.float64],
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,  # cannot be called `lambda`, that's reserved in Python
    kappa: float,
    dimension: int,
    upper_bound: npt.NDArray[np.float64] | float = 1.0,
    lower_bound: npt.NDArray[np.float64] | float = 0.0,
    time_limit: float | None = None,
    max_duality_gap: float | None = None,
    max_iterations: int = 1000,
    robust_gap_tol: float = 1e-8,
    debug: bool = False
) -> tuple[npt.NDArray[np.float64], float]:
    """
    Solve the robust genetic selection problem using SQP in HiGHS.

    Given a robust genetic selection problem
    ```
        max_w (min_mu w'mu subject to mu in U) - (lambda/2)*w'*sigma*w
        subject to lb <= w <= ub,
                   w_S*e_S = 1/2,
                   w_D*e_D = 1/2,
    ```
    where U is a quadratic uncertainty set for mu~N(mubar, omega), this
    function uses HiGHS to find the optimum w and the objective for that
    portfolio. It does this using sequential quadratic programming (SQP),
    approximating the conic constraint associated with robustness using
    a series of linear constraints.

    NOTE: unlike Gurobi, this doesn't yet have controls for `time_limit`
    or `max_duality_gap` to be passed to HiGHS.

    Parameters
    ----------
    TODO: split this out into corresponding three values
    sigma : ndarray
        Covariance matrix of the candidates in the cohorts for selection.
    mubar : ndarray
        Vector of expected values of the expected returns for candidates in the
        cohort for selection.
    TODO: split this out into corresponding three values
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
    max_iterations : int, optional
        Maximum number of iterations that can be taken in solving the problem,
        i.e. the maximum number of constraints to use to approximate the conic
        constraint. Default value is `1000`.
    robust_gap_tol : float, optional
        Tolerance when checking whether an approximating constraint is active
        and whether the SQP overall has converged. Default value is 10^-8.
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

    # initialise an empty model
    model = highspy.Highs()

    # HiGHS spews all its output into the terminal by default, this restricts
    # that behaviour to only happen when the `debug` flag is used.
    if not debug:
        pass  # TODO find suppression variable for highspy

    # integrating bounds within variable definitions is more efficient than
    # as a separate constraint, which Gurobi would convert to bounds anyway
    w = model.FOO  # TODO replace with HiGHS variable defining code
    z = model.BAR  # TODO replace with HiGHS variable defining code

    # construct objective via linear cost and Hessian
    # TODO code for feeding linear term mu-bar of objective into HiGHS
    # TODO code for feeding (lam/2)*Hessian from objective into HiGHS
    # TODO work out how to handle -kappa*z: within linear term?

    # TODO code for feeding matrix of inequality constraints into HiGHS

    # set up the two sum-to-half constraints
    M = np.zeros((2, dimension), dtype=int)
    # define the M so that column i is [1;0] if i is a sire and [0;1] otherwise
    M[0, sires] = 1
    M[1, dams] = 1
    # define the right hand side of the constraint Mx = m
    m = np.full(2, 0.5, dtype=float)
    # TODO work out including equality constraints in HiGHS

    # optional controls to stop HiGHS taking too long
    # TODO explore what options HiGHS offers for timeout exiting

    for i in range(max_iterations):
        # optimization of the model, print weights and objective
        model.run()
        solution = model.getSolution()

        # TODO code for debug print of solution
        if debug:
            pass

        # TODO code for checking which constraints are active in HiGHS

        # z coefficient for the new constraint
        w_star: npt.NDArray[np.float64] = np.array(solution)

        # TODO work out how to compute this when Omega in C{R/C}F
        alpha: float = sqrt(w_star.transpose()@omega@w_star)

        # if gap between z and w'Omega w has converged, done
        if abs(z.X - alpha) < robust_gap_tol:
            break

        # add a new plane to the approximation of the uncertainty cone
        # TODO work out how to add this to an existing HiGHS model
        model.addConstr(alpha*z >= w_star.transpose()@omega@w, name=f"P{i}")

    # model file can be used externally for verification
    if debug:
        # TODO check if HIGHS has a
        model.write("robust-sqp-opt.mps")

    # TODO check how to access objective value
    return np.array(solution), model.ObjVal  # HACK np.array avoids issue #9
