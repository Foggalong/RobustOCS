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
    "gurobi_robust_genetics_sqp",
    "highs_standard_genetics",
    "highs_robust_genetics_sqp"
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
    debug: str | bool = False
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
    debug : str or bool, optional
        Flag which controls both whether Gurobi prints its output to terminal
        and whether it saves the model file to the working directory. If given
        as a string, that string is used as the model output name, 'str.mps',
        or if boolean `True` then `grb-std-opt.mps`. Default value is `False`.

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
        if type(debug) is str:
            model.write(f"{debug}.mps")
        else:
            model.write("grb-std-opt.mps")

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
    debug: str | bool = False
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
    debug : str or bool, optional
        Flag which controls both whether Gurobi prints its output to terminal
        and whether it saves the model file to the working directory. If given
        as a string, that string is used as the model output name, 'str.mps',
        or if boolean `True` then `grb-rob-opt.mps`. Default value is `False`.

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
        if type(debug) is str:
            model.write(f"{debug}.mps")
        else:
            model.write("grb-rob-opt.mps")

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
    robust_gap_tol: float = 1e-7,
    debug: str | bool = False
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
        and whether the SQP overall has converged. Default value is 10^-7.
    debug : str or bool, optional
        Flag which controls both whether Gurobi prints its output to terminal
        and whether it saves the model file to the working directory. If given
        as a string, that string is used as the model output name, 'str.mps',
        or if boolean `True` then `grb-rob-sqp.mps`. Default value is `False`.

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
        if type(debug) is str:
            model.write(f"{debug}.mps")
        else:
            model.write("grb-rob-sqp.mps")

    return np.array(w.X), z.X, model.ObjVal  # HACK np.array avoids issue #9


def highs_bound_like(vector: npt.NDArray[np.float64],
                     value: float | npt.NDArray[np.float64]
                     ) -> npt.NDArray[np.float64]:
    """
    Helper function which allows HiGHS to interpret variable bounds specified
    either as a vector or a single floating point value. If `value` is an array
    will just return that array. If `value` is a float, it'll return a NumPy
    array in the shape of `vector` with every entry being `value`.
    """
    return np.full_like(vector, value) if type(value) is float else value


def highs_standard_genetics(
    sigma: sparse.spmatrix,
    mu: npt.NDArray[np.float64],
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,  # cannot be called `lambda`, that's reserved in Python
    dimension: int,
    upper_bound: npt.NDArray[np.float64] | float = 1.0,
    lower_bound: npt.NDArray[np.float64] | float = 0.0,
    time_limit: float | None = None,
    max_duality_gap: float | None = None,
    debug: str | bool = False
) -> tuple[npt.NDArray[np.float64], float]:
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
    upper_bound : ndarray or float, optional
        Upper bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `1.0`.
    lower_bound : ndarray or float, optional
        Lower bound on how much each candidate can contribute. Can be an array
        of differing bounds for each candidate, or a float which applies to all
        candidates. Default value is `0.0`.
    time_limit : float or None, optional
        Maximum amount of time in seconds to give HiGHS to solve the problem.
        Default value is `None`, i.e. no time limit.
    max_duality_gap : float or None, optional
        HiGHS does not support a tolerance on duality gap for this type of
        problem, so regardless whether specified the value will be ignored.
    debug : str or bool, optional
        Flag which controls both whether Gurobi prints its output to terminal
        and whether it saves the model file to the working directory. If given
        as a string, that string is used as the model output name, 'str.mps',
        or if boolean `True` then `hgs-std-opt.mps`. Default value is `False`.

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

    model.lp_.model_name_ = "standard-genetics"
    model.lp_.num_col_ = dimension
    model.lp_.num_row_ = 2

    # HiGHS does minimization so negate objective
    model.lp_.col_cost_ = -mu

    # bounds on w using a helper function
    model.lp_.col_lower_ = highs_bound_like(mu, lower_bound)
    model.lp_.col_upper_ = highs_bound_like(mu, upper_bound)

    # define the quadratic term in the objective
    sigma = sparse.csc_matrix(sigma)  # BUG is sigma in CSR or CSC format?
    model.hessian_.dim_ = dimension
    model.hessian_.start_ = sigma.indptr
    model.hessian_.index_ = sigma.indices
    # HiGHS multiplies Hessian by 1/2 so just need factor of lambda
    model.hessian_.value_ = lam*sigma.data

    # add Mx = m to the model using CSR format. for M it's less efficient than
    # if it were stored densely, but HiGHS requires CSR for input
    model.lp_.row_lower_ = model.lp_.row_upper_ = np.full(2, 0.5)
    model.lp_.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    model.lp_.a_matrix_.start_ = np.array([0, len(sires), dimension])
    model.lp_.a_matrix_.index_ = np.array(list(sires) + list(dams))
    model.lp_.a_matrix_.value_ = np.ones(dimension, dtype=int)

    # HiGHS spews all its output into the terminal by default, this restricts
    # that behaviour to only happen when the `debug` flag is used.
    if not debug:
        h.setOptionValue('output_flag', False)
        h.setOptionValue('log_to_console', False)

    # optional controls to stop HiGHS taking too long
    if time_limit:
        h.setOptionValue('time_limit', time_limit)
    if max_duality_gap:
        pass  # NOTE HiGHS doesn't support duality gap, skip

    h.passModel(model)
    h.run()

    # model file can be used externally for verification
    if debug:
        if type(debug) is str:
            h.writeModel(f"{debug}.mps")
        else:
            h.writeModel("hgs-std-opt.mps")

    # prints the solution with info about dual values
    if debug:
        h.writeSolution("", 1)

    # by default, col_value is a stock-Python list
    solution = np.array(h.getSolution().col_value)
    # we negated the objective function, so negate it back
    objective_value = -h.getInfo().objective_function_value

    return solution, objective_value


def highs_robust_genetics_sqp(
    sigma: sparse.spmatrix,
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
    robust_gap_tol: float = 1e-7,
    debug: str | bool = False
) -> tuple[npt.NDArray[np.float64], float, float]:
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
        and whether the SQP overall has converged. Default value is 10^-7.
    debug : str or bool, optional
        Flag which controls both whether Gurobi prints its output to terminal
        and whether it saves the model file to the working directory. If given
        as a string, that string is used as the model output name, 'str.mps',
        or if boolean `True` then `grb-rob-sqp.mps`. Default value is `False`.

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
    h = highspy.Highs()
    model = highspy.HighsModel()

    # use value for infinity from HiGHS
    inf = highspy.kHighsInf

    model.lp_.model_name_ = "robust-genetics"
    model.lp_.num_col_ = dimension + 1  # additional column for z
    model.lp_.num_row_ = 2

    # HiGHS does minimization so negate objective
    model.lp_.col_cost_ = np.append(-mubar, kappa)

    # bounds on w using a helper function, plus 0 <= z <= inf
    model.lp_.col_lower_ = np.append(highs_bound_like(mubar, lower_bound), 0)
    model.lp_.col_upper_ = np.append(highs_bound_like(mubar, upper_bound), inf)

    # define the quadratic term in the objective
    sigma = sparse.csc_matrix(sigma)  # BUG is sigma in CSR or CSC format?
    model.hessian_.dim_ = dimension + 1
    model.hessian_.start_ = np.append(sigma.indptr, dimension*dimension)
    model.hessian_.index_ = sigma.indices
    # HiGHS multiplies Hessian by 1/2 so just need factor of lambda
    model.hessian_.value_ = lam*sigma.data

    # add Mx = m to the model using CSR format. Note the additional column
    model.lp_.row_lower_ = model.lp_.row_upper_ = np.array([0.5, 0.5, 0])
    model.lp_.a_matrix_.format_ = highspy.MatrixFormat.kRowwise
    model.lp_.a_matrix_.start_ = np.array([0, len(sires), dimension + 1])
    model.lp_.a_matrix_.index_ = np.array(list(sires) + list(dams) + [dimension])
    model.lp_.a_matrix_.value_ = np.append(np.ones(dimension, dtype=int), 0)

    # BUG should be able to use this to add `z` to the model, but isn't working
    # h.addVar(0, highspy.kHighsInf)
    # h.changeColCost(dimension, kappa)

    # HiGHS spews all its output into the terminal by default, this restricts
    # that behaviour to only happen when the `debug` flag is used.
    if not debug:
        h.setOptionValue('output_flag', False)
        h.setOptionValue('log_to_console', False)

    # optional controls to stop HiGHS taking too long
    if time_limit:
        h.setOptionValue('time_limit', time_limit)
    if max_duality_gap:
        pass  # NOTE HiGHS doesn't support duality gap, skip

    h.passModel(model)

    for i in range(max_iterations):
        # optimization of the model, print weights and objective
        h.run()

        # by default, col_value is a stock-Python list
        solution: npt.NDArray[np.float64] = np.array(h.getSolution().col_value)
        w_star = solution[:-1]
        z_star = solution[-1]

        # we negated the objective function, so negate it back
        objective_value: float = -h.getInfo().objective_function_value

        if debug:
            print(f"{i}: {w_star}, {objective_value:g}")

        # assess which constraints are currently active
        # TODO see if HiGHS can explore active constraints mid-run

        # z coefficient for the new constraint
        alpha: float = sqrt(w_star.transpose()@omega@w_star)

        # if gap between z and w'Omega w has converged, done
        if abs(z_star - alpha) < robust_gap_tol:
            break

        # add a new plane to the approximation of the uncertainty cone
        num_nz = dimension + 1  # HACK assuming entirely dense
        index = np.array(range(dimension + 1))
        value = np.append(-omega@w_star, alpha)
        h.addRow(0, inf, num_nz, index, value)

    # model file can be used externally for verification
    if debug:
        if type(debug) is str:
            h.writeModel(f"{debug}.mps")
        else:
            h.writeModel("hgs-rob-sqp.mps")

    # prints the solution with info about dual values
    if debug:
        h.writeSolution("", 1)

    # final value of solution is the z value, return separately
    return solution[:-1], solution[-1], objective_value
