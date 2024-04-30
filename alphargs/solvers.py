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
    lam: float,
    dimension: int,
    upper_bound: npt.NDArray[np.float64] | float = 1.0,
    lower_bound: npt.NDArray[np.float64] | float = 0.0,
    time_limit: float | None = None,
    max_duality_gap: float | None = None,
    debug: bool = False
) -> tuple[npt.NDArray[np.float64], float]:
    """
    Takes a sigma, mu, sire list, dam list, and dimension as inputs and solves
    the standard genetic selection problem with Gurobi for a given value of
    lambda (lam), returning the portfolio and objective value as outputs. The
    default lower and upper bounds of 0 and 1 can be changed also.

    Optional arguments time_limit and max_duality_gap respectively control how
    long Gurobi will spend on the problem and the maximum allowable duality
    gap. Optional argument debug sets whether Gurobi prints output to terminal.
    """

    # create models for standard and robust genetic selection
    model = gp.Model("standard-genetics")

    # Gurobi spews all its output into the terminal by default, this restricts
    # that behaviour to only happen when the `debug` flag is used.
    if not debug: model.setParam('OutputFlag', 0)

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
    if time_limit: model.setParam(GRB.Param.TimeLimit, time_limit)
    if max_duality_gap: model.setParam('MIPGap', max_duality_gap)

    # model file can be used externally for verification
    if debug: model.write("standard-opt.mps")

    model.optimize()
    return w.X, model.ObjVal


def gurobi_robust_genetics(
    sigma: npt.NDArray[np.float64],
    mubar: npt.NDArray[np.float64],
    omega: npt.NDArray[np.float64],
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,
    kappa: float,
    dimension: int,
    upper_bound: npt.NDArray[np.float64] | float = 1.0,
    lower_bound: npt.NDArray[np.float64] | float = 0.0,
    time_limit: float | None = None,
    max_duality_gap: float | None = None,
    debug: bool = False
) -> tuple[npt.NDArray[np.float64], float, float]:
    """
    Takes a sigma, mu-bar, omega, sire list, dam list, and dimension as inputs
    and solves the robust genetic selection problem using Gurobi for given
    values of lambda (lam) and kappa. It returns the portfolio and objective
    value as outputs. The default lower and upper bounds of 0 and 1 can be
    changed also.

    Optional arguments time_limit and max_duality_gap respectively control how
    long Gurobi will spend on the problem and the maximum allowable duality
    gap. Optional argument debug sets whether Gurobi prints output to terminal.
    """

    # create models for standard and robust genetic selection
    model = gp.Model("robust-genetics")

    # Gurobi spews all its output into the terminal by default, this restricts
    # that behaviour to only happen when the `debug` flag is used.
    if not debug: model.setParam('OutputFlag', 0)

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
    if time_limit: model.setParam(GRB.Param.TimeLimit, time_limit)
    if max_duality_gap: model.setParam('MIPGap', max_duality_gap)

    # model file can be used externally for verification
    if debug: model.write("robust-opt.mps")

    model.optimize()
    return w.X, z.X, model.ObjVal
