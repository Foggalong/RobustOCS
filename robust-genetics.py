#!/usr/bin/env python3

import numpy as np          # defines matrix structures
import numpy.typing as npt  # variable typing definitions for NumPy
import gurobipy as gp       # Gurobi optimization interface (1)
from gurobipy import GRB    # Gurobi optimization interface (2)
from sys import maxsize     # maximum precision for specific system


# OUTPUT SETTINGS
# Numpy makes some odd choices with its default output formats, this adjusts
# the most intrusive of those for anything which uses this utility.

np.set_printoptions(
    formatter={'float_kind': "{:.8f}".format},  # only show to 8 d.p.
    threshold=maxsize  # want to round rather than truncate when printing
)


# READING GENETICS DATA
# Genetics data could be presented to our solvers in multiple formats, these
# functions define the appropriate methods for loading those in correctly.

def load_ped(filename: str) -> dict:
    """
    Function for reading *.ped files to a dictionary. Takes the file name
    as a string input and returns the pedigree structure as a dictionary.
    """
    with open(filename, "r") as file:
        # first line of *.ped lists the headers; skip
        file.readline()
        # create a list of int lists from each line (dropping optional labels)
        data = [[int(x) for x in line.split(",")[0:3]] for line in file]
    # convert this list of lists into a dictionary
    ped = {entry[0]: entry[1:3] for entry in data}
    return ped


def makeA(pedigree: dict) -> npt.NDArray[np.float64]:
    """
    Construct Wright's Numerator Relationship Matrix from a given pedigree
    structure. Takes the pedigree as a dictionary input and returns the
    matrix as output.
    """
    m = len(pedigree)
    # preallocate memory for A
    A = np.zeros((m, m), dtype=float)

    # iterate over rows
    for i in range(0, m):
        # save parent indexes: pedigrees indexed from 1, Python from 0
        p = pedigree[i+1][0]-1
        q = pedigree[i+1][1]-1
        # iterate over columns sub-diagonal
        for j in range(0, i):
            # calculate sub-diagonal entries
            A[i, j] = 0.5*(A[j, p] + A[j, q])
            # populate sup-diagonal (symmetric)
            A[j, i] = A[i, j]
        # calculate diagonal entries
        A[i, i] = 1 + 0.5*A[p, q]

    return A


def load_problem(A_filename: str, E_filename: str, S_filename: str,
                 dimension: int | None = None, pedigree: bool = False
                 ) -> tuple[npt.NDArray[np.float64], npt.NDArray[np.float64],
                            npt.NDArray[np.float64], int]:
    """
    Used to load genetic selection problems into NumPy. It takes three
    string inputs for filenames where Sigma, Mu, and Omega are stored,
    as well as an optional integer input for problem dimension if this
    is known. If it's know know, it's worked out based on E_filename.

    As output, it returns (A, E, S, n), where A and S are n-by-n NumPy
    arrays, E is a length n NumPy array, and n is an integer.
    """

    def load_symmetric_matrix(filename: str, dimension: int
                              ) -> npt.NDArray[np.float64]:
        """
        Since NumPy doesn't have a stock way to load matrices
        stored in coordinate format format, this adds one.
        """

        matrix = np.zeros([dimension, dimension], dtype=float)

        with open(filename, 'r') as file:
            for line in file:
                i, j, entry = line.split(" ")
                # data files indexed from 1, not 0
                matrix[int(i)-1, int(j)-1] = entry
                matrix[int(j)-1, int(i)-1] = entry

        return matrix

    E = np.loadtxt(E_filename, dtype=float)
    # if dimension not specified, use `E` which doesn't need preallocation
    if not dimension:
        assert isinstance(E.size, int)  # catches E being empty
        dimension = E.size

    # S is stored by coordinates so need special loader
    S = load_symmetric_matrix(S_filename, dimension)
    # A can be stored as a pedigree or by coordinates
    if pedigree:
        A = makeA(load_ped(A_filename))
    else:
        A = load_symmetric_matrix(A_filename, dimension)

    return A, E, S, dimension


# DEFINING SOLVER
# With the problems properly loaded into Numpy, this section contains functions
# for solving those under various formulations and methods, as well as printing
# and comparing outputted solutions.

def gurobi_standard_genetics(
    sigma: np.ndarray,
    mu: np.ndarray,
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc
    lam: float,
    dimension: int,
    upper_bound: npt.NDArray[np.float64] | float = 1.0,
    lower_bound: npt.NDArray[np.float64] | float = 0.0,
    time_limit: float | None = None,
    max_duality_gap: float | None = None,
    debug: bool = False
):  # -> tuple[npt.NDArray[np.float64], float]:  # BUG Gurobi typing broken
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
):  # -> tuple[npt.NDArray, float, float]:  # BUG Gurobi typing broken
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


def print_compare_solutions(
    portfolio1,  # : npt.NDArray[np.float64],  # BUG Gurobi typing broken
    portfolio2,  # : npt.NDArray[np.float64],  # BUG Gurobi typing broken
    objective1: float,
    objective2: float,
    precision: int = 5,
    z1: float | None = None,
    z2: float | None = None,
    name1: str = "First",
    name2: str = "Second"
) -> None:
    """
    Takes two solutions (comprised of at least a portfolio and objective value,
    plus an optional z-value and/or solution name) as inputs, and prints a
    comparison of the two solutions to the terminal. The number of decimals
    values are displayed to defaults to 5, but can be changed through the
    precision argument.
    """

    dimension = portfolio1.size
    order = len(str(dimension))

    # HACK header breaks if precision < 3 or len(problem1) != 5
    print(f"i{' '*(order-1)}  {name1}  {' '*(precision-3)}{name2}")
    for candidate in range(dimension):
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


# MAIN
# A temporary section which includes an example problem. This will live here
# while porting to module, but will be removed once done.

# key problem variables
sigma, mubar, omega, n = load_problem(
    "Example/04/A04.txt",
    "Example/04/EBV04.txt",
    "Example/04/S04.txt"
)

# NOTE this trick of handling sex data is specific to the initial simulation
# data which is structured so that candidates alternate between sires (which
# have even indices) and dams (which have odd indices).
sires = range(0, n, 2)
dams = range(1, n, 2)

lam = 0.5
kap = 1

# computes the standard and robust genetic selection solutions
w_std, obj_std = gurobi_standard_genetics(sigma, mubar, sires, dams, lam, n)
w_rbs, z_rbs, obj_rbs = gurobi_robust_genetics(sigma, mubar, omega, sires,
                                               dams, lam, kap, n)

print_compare_solutions(w_std, w_rbs, obj_std, obj_rbs,
                        z2=z_rbs, name1="w_std", name2="w_rbs")

# DONE
pass
