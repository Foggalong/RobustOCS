# -*- coding: utf-8 -*-
"""Defining Solvers

With an optimal contribution selection problems properly loaded into Python,
`solvers` contains functions for solving those under various formulations and
methods.

Documentation is available in the docstrings and online at
https://github.com/Foggalong/RobustOCS/wiki
"""

import numpy as np          # defines matrix structures
import numpy.typing as npt  # variable typing definitions for NumPy
import gurobipy as gp       # Gurobi optimization interface
import highspy              # HiGHS optimization interface
from math import sqrt       # used within the robust constraint
from scipy import sparse    # used for sparse matrix format
from time import time       # used for timing SQP methods

# controls what's imported on `from robustocs.solvers import *`
__all__ = [
    "gurobi_standard_genetics",
    "gurobi_robust_genetics",  # alias
    "gurobi_robust_genetics_conic",
    "gurobi_robust_genetics_sqp",
    "highs_standard_genetics",
    "highs_robust_genetics",  # alias
    "highs_robust_genetics_sqp"
]


def gurobi_standard_genetics(
    sigma: npt.NDArray[np.floating] | sparse.spmatrix,
    mu: npt.NDArray[np.floating],
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,  # cannot be called `lambda`, that's reserved in Python
    dimension: int,
    upper_bound: npt.NDArray[np.floating] | float = 1.0,
    lower_bound: npt.NDArray[np.floating] | float = 0.0,
    time_limit: float | None = None,
    model_output: str = '',
    debug: bool = False
) -> tuple[npt.NDArray[np.floating], float]:
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
    model_output : str, optional
        Flag which controls whether Gurobi saves the model file to the working
        directory. If given, the string is used as the file name, 'str.mps',
        Default value is the empty string, i.e. the file isn't saved.
    debug : bool, optional
        Flag which controls whether Gurobi prints its output to terminal.
        Default value is `False`.

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
    M = np.zeros((2, dimension), dtype=np.bool)
    # define the M so that column i is [1;0] if i is a sire and [0;1] otherwise
    M[0, sires] = 1
    M[1, dams] = 1
    # define the right hand side of the constraint Mx = m
    m = np.full(2, 0.5)
    model.addConstr(M@w == m, name="sum-to-half")

    # optional controls to stop Gurobi taking too long
    if time_limit:
        model.setParam(gp.GRB.Param.TimeLimit, time_limit)

    # model file can be used externally for verification
    if model_output:
        model.write(f"{model_output}.mps")

    model.optimize()
    return np.array(w.X), model.ObjVal  # HACK np.array avoids issue #9


def gurobi_robust_genetics_conic(
    sigma: npt.NDArray[np.floating] | sparse.spmatrix,
    mubar: npt.NDArray[np.floating],
    omega: npt.NDArray[np.floating] | sparse.spmatrix,
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,  # cannot be called `lambda`, that's reserved in Python
    kappa: float,
    dimension: int,
    upper_bound: npt.NDArray[np.floating] | float = 1.0,
    lower_bound: npt.NDArray[np.floating] | float = 0.0,
    time_limit: float | None = None,
    model_output: str = '',
    debug: bool = False
) -> tuple[npt.NDArray[np.floating], float, float]:
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
    model_output : str, optional
        Flag which controls whether Gurobi saves the model file to the working
        directory. If given, the string is used as the file name, 'str.mps',
        Default value is the empty string, i.e. the file isn't saved.
    debug : bool, optional
        Flag which controls whether Gurobi prints its output to terminal.
        Default value is `False`.

    Returns
    -------
    ndarray
        Portfolio vector which Gurobi has determined is a solution.
    float
        Auxiliary variable corresponding to uncertainty associated with the
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
    M = np.zeros((2, dimension), dtype=np.bool)
    # define the M so that column i is [1;0] if i is a sire and [0;1] otherwise
    M[0, sires] = 1
    M[1, dams] = 1
    # define the right hand side of the constraint Mx = m
    m = np.full(2, 0.5, dtype=np.floating)
    model.addConstr(M@w == m, name="sum-to-half")

    # conic constraint which comes from robust optimization
    model.addConstr(z**2 >= w.transpose()@omega@w, name="uncertainty")

    # optional controls to stop Gurobi taking too long
    if time_limit:
        model.setParam(gp.GRB.Param.TimeLimit, time_limit)

    # model file can be used externally for verification
    if model_output:
        model.write(f"{model_output}.mps")

    model.optimize()
    return np.array(w.X), z.X, model.ObjVal  # HACK np.array avoids issue #9


def gurobi_robust_genetics_sqp(
    sigma: npt.NDArray[np.floating] | sparse.spmatrix,
    mubar: npt.NDArray[np.floating],
    omega: npt.NDArray[np.floating] | sparse.spmatrix,
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,  # cannot be called `lambda`, that's reserved in Python
    kappa: float,
    dimension: int,
    upper_bound: npt.NDArray[np.floating] | float = 1.0,
    lower_bound: npt.NDArray[np.floating] | float = 0.0,
    time_limit: float | None = None,
    max_iterations: int = 1000,
    robust_gap_tol: float = 1e-7,
    model_output: str = '',
    debug: bool = False
) -> tuple[npt.NDArray[np.floating], float, float]:
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
        Maximum amount of time in seconds to give Gurobi to solve the problem.
        Default value is `None`, i.e. no time limit.
    max_iterations : int, optional
        Maximum number of iterations that can be taken in solving the problem,
        i.e. the maximum number of constraints to use to approximate the conic
        constraint. Default value is `1000`.
    robust_gap_tol : float, optional
        Tolerance when checking whether an approximating constraint is active
        and whether the SQP overall has converged. Default value is 10^-7.
    model_output : str, optional
        Flag which controls whether Gurobi saves the model file to the working
        directory. If given, the string is used as the file name, 'str.mps',
        Default value is the empty string, i.e. the file isn't saved.
    debug : bool, optional
        Flag which controls whether Gurobi prints its output to terminal.
        Default value is `False`.

    Returns
    -------
    ndarray
        Portfolio vector which Gurobi has determined is a solution.
    float
        Auxiliary variable corresponding to uncertainty associated with the
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
    M = np.zeros((2, dimension), dtype=np.bool)
    # define the M so that column i is [1;0] if i is a sire and [0;1] otherwise
    M[0, sires] = 1
    M[1, dams] = 1
    # define the right hand side of the constraint Mx = m
    m = np.full(2, 0.5, dtype=np.floating)
    model.addConstr(M@w == m, name="sum-to-half")

    # optional controls to stop Gurobi taking too long
    if time_limit:
        time_remaining: float = time_limit

    for i in range(max_iterations):
        # optional controls to stop Gurobi taking too long
        if time_limit:
            model.setParam(gp.GRB.Param.TimeLimit, time_remaining)

        # optimization of the model, print weights and objective
        try:
            model.optimize()
        except RuntimeError as e:
            raise RuntimeError(f"Gurobi failed with error:\n{e}")

        # subtract time take from time remaining
        if time_limit:
            time_remaining -= model.Runtime
            if time_remaining < 0:
                raise RuntimeError(f"Gurobi hit time limit of {time_limit} "
                                   f"seconds without reaching optimality")

        # return model and solution at every approximation to help debug
        if model_output:
            model.write(f"{model_output}.mps")
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
        w_star: npt.NDArray[np.floating] = np.array(w.X)
        alpha: float = sqrt(w_star.transpose()@omega@w_star)

        # if gap between z and w'Omega w has converged, done
        if abs(z.X - alpha) < robust_gap_tol:
            break

        # add a new plane to the approximation of the uncertainty cone
        model.addConstr(alpha*z >= w_star.transpose()@omega@w, name=f"P{i}")

    return np.array(w.X), z.X, model.ObjVal  # HACK np.array avoids issue #9


# make gurobi_robust_genetics(...) an alias of the fastest method
gurobi_robust_genetics = gurobi_robust_genetics_conic


def highs_bound_like(dimension: int,
                     value: float | list[float] | npt.NDArray[np.floating]
                     ):  # BUG broke: npt.NDArray[np.floating] | list[float]
    """
    Helper function which allows HiGHS to interpret variable bounds specified
    either as a vector or a single floating point value. If `value` is an array
    will just return that array. If `value` is a float, it'll return a NumPy
    array in the shape of `vector` with every entry being `value`.
    """

    return [value]*dimension if type(value) is float else value


def highs_standard_genetics(
    sigma: sparse.spmatrix,
    mu: npt.NDArray[np.floating],
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,  # cannot be called `lambda`, that's reserved in Python
    dimension: int,
    upper_bound: npt.NDArray[np.floating] | list[float] | float = 1.0,
    lower_bound: npt.NDArray[np.floating] | list[float] | float = 0.0,
    time_limit: float | None = None,
    model_output: str = '',
    debug: bool = False
) -> tuple[npt.NDArray[np.floating], float]:
    """
    Solve the standard genetic selection problem using HiGHS.

    Given a standard genetic selection problem
    ```
        max_w w'mu - (lambda/2)*w'*sigma*w
        subject to lb <= w <= ub,
                   w_S*e_S = 1/2,
                   w_D*e_D = 1/2,
    ```
    this function uses HiGHS to find the optimum w and the objective for that
    portfolio. Additional parameters give control over long HiGHS can spend
    on the problem, to prevent indefinite hangs.

    Parameters
    ----------
    sigma : spmatrix
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
    upper_bound : ndarray, list, or float, optional
        Upper bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `1.0`.
    lower_bound : ndarray, list, or float, optional
        Lower bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `0.0`.
    time_limit : float or None, optional
        Maximum amount of time in seconds to give HiGHS to solve the problem.
        Default value is `None`, i.e. no time limit.
    model_output : str, optional
        Flag which controls whether Gurobi saves the model file to the working
        directory. If given, the string is used as the file name, 'str.mps',
        Default value is the empty string, i.e. the file isn't saved.
    debug : bool, optional
        Flag which controls whether Gurobi prints its output to terminal.
        Default value is `False`.

    Returns
    -------
    ndarray
        Portfolio vector which HiGHS has determined is a solution.
    float
        Value of the objective function for returned solution vector.
    """

    # initialise an empty model
    h = highspy.Highs()
    model = highspy.HighsModel()

    # NOTE HiGHS doesn't support typing for model parameters
    model.lp_.model_name_ = "standard-genetics"
    model.lp_.num_col_ = dimension
    model.lp_.num_row_ = 2

    # HiGHS does minimization so negate objective
    model.lp_.col_cost_ = -mu

    # bounds on w using a helper function
    model.lp_.col_lower_ = highs_bound_like(dimension, lower_bound)
    model.lp_.col_upper_ = highs_bound_like(dimension, upper_bound)

    # define the quadratic term in the objective
    model.hessian_.dim_ = dimension
    model.hessian_.start_ = sigma.indptr
    model.hessian_.index_ = sigma.indices
    # HiGHS multiplies Hessian by 1/2 so just need factor of lambda
    model.hessian_.value_ = lam*sigma.data

    # add Mx = m to the model using CSR format. for M it's less efficient than
    # if it were stored densely, but HiGHS requires CSR for input
    model.lp_.row_lower_ = model.lp_.row_upper_ = np.full(2, 0.5)
    model.lp_.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    model.lp_.a_matrix_.start_ = [0, len(sires), dimension]
    model.lp_.a_matrix_.index_ = list(sires) + list(dams)
    model.lp_.a_matrix_.value_ = [1]*dimension

    # HiGHS spews all its output into the terminal by default, this restricts
    # that behaviour to only happen when the `debug` flag is used.
    if not debug:
        h.setOptionValue('output_flag', False)
        h.setOptionValue('log_to_console', False)

    # HiGHS' passModel returns a status indicating its success
    pass_status: highspy._core.HighsStatus = h.passModel(model)
    # model file must be saved between passModel and any error
    if model_output:
        h.writeModel(f"{model_output}.mps")
    # HiGHS will try to continue if it gets an error, so stop it
    if pass_status == highspy.HighsStatus.kError:
        raise ValueError(f"h.passModel failed with status "
                         f"{h.getModelStatus()}")

    # optional controls to stop HiGHS taking too long (see issue #16)
    if time_limit:
        h.setOptionValue('time_limit', h.getRunTime() + time_limit)

    # HiGHS' run returns a status indicating its success
    run_status: highspy._core.HighsStatus = h.run()
    # solution (with dual info) must be printed between run and any error
    if debug:
        h.writeSolution("", 1)
    mod_status: highspy._core.HighsModelStatus = h.getModelStatus()
    # HiGHS will try to continue if it gets an error, so stop it
    if run_status == highspy.HighsStatus.kError:
        raise ValueError(f"h.run failed with status {mod_status}")
    elif mod_status != highspy.HighsModelStatus.kOptimal:
        raise RuntimeError(f"h.run did not achieve optimality, status "
                           f"{mod_status}")

    # by default, col_value is a stock-Python list
    solution: npt.NDArray[np.floating] = np.array(h.getSolution().col_value)
    # we negated the objective function, so negate it back
    objective_value: float = -h.getInfo().objective_function_value

    return solution, objective_value


def highs_robust_genetics_sqp(
    sigma: sparse.spmatrix,
    mubar: npt.NDArray[np.floating],
    omega: npt.NDArray[np.floating] | sparse.spmatrix,
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,  # cannot be called `lambda`, that's reserved in Python
    kappa: float,
    dimension: int,
    upper_bound: npt.NDArray[np.floating] | list[float] | float = 1.0,
    lower_bound: npt.NDArray[np.floating] | list[float] | float = 0.0,
    time_limit: float | None = None,
    max_iterations: int = 1000,
    robust_gap_tol: float = 1e-7,
    model_output: str = '',
    debug: bool = False
) -> tuple[npt.NDArray[np.floating], float, float]:
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

    Parameters
    ----------
    sigma : spmatrix
        Covariance matrix of the candidates in the cohorts for selection.
    mubar : ndarray
        Vector of expected values of the expected returns for candidates in the
        cohort for selection.
    omega : ndarray or spmatrix   # TODO this doesn't *have* to be an spmatrix
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
    upper_bound : ndarray, list, or float, optional
        Upper bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `1.0`.
    lower_bound : ndarray, list, or float, optional
        Lower bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `0.0`.
    time_limit : float or None, optional
        Maximum amount of time in seconds to give HiGHS to solve the problem.
        Default value is `None`, i.e. no time limit.
    max_iterations : int, optional
        Maximum number of iterations that can be taken in solving the problem,
        i.e. the maximum number of constraints to use to approximate the conic
        constraint. Default value is `1000`.
    robust_gap_tol : float, optional
        Tolerance when checking whether an approximating constraint is active
        and whether the SQP overall has converged. Default value is 10^-7.
    model_output : str, optional
        Flag which controls whether Gurobi saves the model file to the working
        directory. If given, the string is used as the file name, 'str.mps',
        Default value is the empty string, i.e. the file isn't saved.
    debug : bool, optional
        Flag which controls whether Gurobi prints its output to terminal.
        Default value is `False`.

    Returns
    -------
    ndarray
        Portfolio vector which Gurobi has determined is a solution.
    float
        Auxiliary variable corresponding to uncertainty associated with the
        portfolio vector which Gurobi has determined is a solution.
    float
        Value of the objective function for returned solution vector.
    """

    # initialise an empty model
    h = highspy.Highs()
    model = highspy.HighsModel()

    # use value for infinity from HiGHS
    inf = highspy.kHighsInf

    # NOTE HiGHS doesn't support typing for model parameters
    model.lp_.model_name_ = "robust-genetics"
    model.lp_.num_col_ = dimension
    model.lp_.num_row_ = 2

    # HiGHS does minimization so negate objective
    model.lp_.col_cost_ = -mubar

    # bounds on w using a helper function
    model.lp_.col_lower_ = highs_bound_like(dimension, lower_bound)
    model.lp_.col_upper_ = highs_bound_like(dimension, upper_bound)

    # define the quadratic term in the objective
    model.hessian_.dim_ = dimension
    model.hessian_.start_ = sigma.indptr
    model.hessian_.index_ = sigma.indices
    # # # HiGHS multiplies Hessian by 1/2 so just need factor of lambda
    model.hessian_.value_ = lam*sigma.data

    # add Mx = m to the model using CSR format
    model.lp_.row_lower_ = model.lp_.row_upper_ = np.full(2, 0.5)
    model.lp_.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    model.lp_.a_matrix_.start_ = [0, len(sires), dimension]
    model.lp_.a_matrix_.index_ = list(sires) + list(dams)
    model.lp_.a_matrix_.value_ = [1]*dimension

    # HiGHS' passModel returns a status indicating its success
    pass_status: highspy._core.HighsStatus = h.passModel(model)
    # model file must be saved between passModel and any error
    if model_output:
        h.writeModel(f"{model_output}.mps")
    # HiGHS will try to continue if it gets an error, so stop it
    if pass_status == highspy.HighsStatus.kError:
        raise ValueError(f"h.passModel failed with status "
                         f"{h.getModelStatus()}")

    # add z variable with bound 0 < z < inf and cost kappa
    h.addVar(0, highspy.kHighsInf)
    h.changeColCost(dimension, kappa)

    # HiGHS spews all its output into the terminal by default, this restricts
    # that behaviour to only happen when the `debug` flag is used.
    if not debug:
        h.setOptionValue('output_flag', False)
        h.setOptionValue('log_to_console', False)

    # optional controls to stop HiGHS taking too long
    if time_limit:
        time_remaining: float = time_limit

    for i in range(max_iterations):
        # use at most the remaining unused time (see issue #16)
        if time_limit:
            h.setOptionValue('time_limit', h.getRunTime() + time_remaining)
            start_time: float = time()

        try:
            run_status: highspy._core.HighsStatus = h.run()
        except RuntimeError as e:
            raise RuntimeError(f"HiGHS failed with error:\n{e}")

        # subtract time taken from time remaining
        if time_limit:
            time_remaining -= time() - start_time
            if time_remaining < 0:
                raise RuntimeError(f"HiGHS hit time limit of {time_limit} "
                                   f"seconds without reaching optimality")

        # return model and solution at every approximation to help debug
        if model_output:
            h.writeModel(f"{model_output}.mps")
        if debug:
            h.writeSolution("", 1)

        # evaluate HiGHS' return value from h.run and attempt to solve
        model_status: highspy._core.HighsModelStatus = h.getModelStatus()
        # HiGHS will try to continue if it gets an error, so stop it
        if run_status == highspy.HighsStatus.kError:
            raise RuntimeError(f"HiGHS at approximation #{i} failed with "
                               f"status {model_status}")
        elif model_status != highspy.HighsModelStatus.kOptimal:
            raise RuntimeError(f"HiGHS did not achieve optimality at "
                               f"approximation #{i}, status {model_status}")

        # by default, col_value is a stock-Python list
        solution: list[float] = h.getSolution().col_value
        w_star: npt.NDArray[np.floating] = np.array(solution[:-1])
        z_star: float = solution[-1]

        # we negated the objective function, so negate it back
        objective_value: float = -h.getInfo().objective_function_value

        if debug:
            print(f"{i}: {w_star}, {objective_value:g}")

        # assess which constraints are currently active
        active_const: bool = False
        constraints = h.getBasis().row_status
        for c in range(len(constraints)-2):  # first two are sum-to-half
            if constraints[c+2] == highspy.HighsBasisStatus.kBasic:
                active_const = True
                if debug:
                    print(f"P{c} active")  # don't have slack values
        if debug and not active_const:
            print("No active constraints!")

        # z coefficient for the new constraint
        alpha: float = sqrt(w_star.transpose()@omega@w_star)

        # if gap between z and w'Omega w has converged, done
        if abs(z_star - alpha) < robust_gap_tol:
            break

        # add a new plane to the approximation of the uncertainty cone
        num_nz: int = dimension + 1  # HACK assuming entirely dense
        index: range = range(dimension + 1)
        value: npt.NDArray[np.floating] = np.append(-omega@w_star, alpha)
        h.addRow(0, inf, num_nz, index, value)

    # final value of solution is the z value, return separately
    return w_star, z_star, objective_value


# make highs_robust_genetics(...) an alias of the fastest method
highs_robust_genetics = highs_robust_genetics_sqp
