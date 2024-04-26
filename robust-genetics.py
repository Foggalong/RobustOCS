#!/usr/bin/env python3

import numpy as np          # defines matrix structures
import gurobipy as gp       # Gurobi optimization interface (1)
from gurobipy import GRB    # Gurobi optimization interface (2)
from typing import Union    # TODO replace with stock `|` from 3.10+ 


# OUTPUT SETTINGS
# Numpy makes some odd choices with its default output formats,
# this adjusts the most intrusive of those for anything which
# uses this utility.

# want to round rather than truncate when printing
np.set_printoptions(threshold=np.inf)

# only show numpy output to eight decimal places
np.set_printoptions(formatter={'float_kind':"{:.8f}".format})



# READING GENETICS DATA
# TODO write a more descriptive section heading

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


def makeA(pedigree: dict) -> np.ndarray:
    """
    Construct Wright's Numerator Relationship Matrix from a given pedigree
    structure. Takes the pedigree as a dictionary input and returns the
    matrix as output.
    """
    m = len(pedigree)
    # preallocate memory for A 
    A = np.zeros((m, m))

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
                 dimension: bool = False, pedigree: bool = False
                 ) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """
    Used to load genetic selection problems into NumPy. It takes three
    string inputs for filenames where Sigma, Mu, and Omega are stored,
    as well as an optional integer input for problem dimension if this
    is known. If it's know know, it's worked out based on E_filename.

    As output, it returns (A, E, S, n), where A and S are n-by-n NumPy
    arrays, E is a length n NumPy array, and n is an integer.
    """

    def load_symmetric_matrix(filename: str, dimension: int) -> np.ndarray:
        """
        Since NumPy doesn't have a stock way to load matrices
        stored in coordinate format format, this adds one.
        """

        matrix = np.zeros([dimension, dimension])

        with open(filename, 'r') as file:
            for line in file:
                i, j, entry = line.split(" ")
                # data files indexed from 1, not 0
                matrix[int(i)-1, int(j)-1] = entry
                matrix[int(j)-1, int(i)-1] = entry

        return matrix

    # if dimension wasn't supplied, need to find that
    if not dimension:
        # get dimension from EBV, since it's the smallest file
        with open(E_filename, 'r') as file:
            dimension = sum(1 for _ in file)

    # TODO test whether it's better to handle dimension separately as 
    # above or to take the hit of doing `np.loadtxt`s work explicitly
    # but then getting the dimension for free. 
    # EBV isn't in coordinate format so can be loaded directly
    E = np.loadtxt(E_filename)  
    # S is stored by coordinates so need special loader
    S = load_symmetric_matrix(S_filename, dimension)
    # A can be stored as a pedigree or by coordinates
    if pedigree:
        A = makeA(load_ped(A_filename))
    else:
        A = load_symmetric_matrix(A_filename, dimension)

    return A, E, S, dimension




# DEFINING SOLVER

def gurobi_standard_genetics(
    sigma: np.ndarray,
    mu: np.ndarray,
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc 
    lam: float,
    dimension: int,
    upper_bound: Union[np.ndarray, float] = 1.0,
    lower_bound: Union[np.ndarray, float] = 0.0,
    time_limit: Union[float, None] = None,
    min_duality_gap: Union[float, None] = None,
    debug: bool = False
) -> tuple[float, float]:
    """
    TODO write a docstring for solving a robust genetics problem
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
        w.transpose()@mubar - (lam/2)*w.transpose()@(sigma@w),
    GRB.MAXIMIZE)

    # set up the two sum-to-half constraints
    M = np.zeros((2, dimension))
    # define the M so that column i is [1;0] if i is a sire and [0;1] otherwise 
    M[0, sires] = 1
    M[1, dams] = 1
    # define the right hand side of the constraint Mx = m
    m = np.full(2, 0.5)
    model.addConstr(M@w == m, name="sum-to-half")

    # optional controls to stop Gurobi taking too long
    if time_limit: model.setParam(GRB.Param.TimeLimit, time_limit)
    if min_duality_gap: model.setParam('MIPGap', min_duality_gap)

    # model file can be used externally for verification
    if debug: model.write("standard-opt.mps")

    model.optimize()
    return w.X, model.ObjVal

def gurobi_robust_genetics(
    sigma: np.ndarray,
    mubar: np.ndarray,
    omega: np.ndarray,
    sires,  # type could be np.ndarray, sets[ints], lists[int], range, etc
    dams,   # type could be np.ndarray, sets[ints], lists[int], range, etc 
    lam: float,
    dimension: int,
    upper_bound: Union[np.ndarray, float] = 1.0,
    lower_bound: Union[np.ndarray, float] = 0.0,
    kappa: float = 0.0,
    time_limit: Union[float, None] = None,
    min_duality_gap: Union[float, None] = None,
    debug: bool = False
) -> tuple[float, float, float]:
    """
    TODO write a docstring for solving a robust genetics problem
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
    GRB.MAXIMIZE)

    # set up the two sum-to-half constraints
    M = np.zeros((2, dimension))
    # define the M so that column i is [1;0] if i is a sire and [0;1] otherwise 
    M[0, sires] = 1
    M[1, dams] = 1
    # define the right hand side of the constraint Mx = m
    m = np.full(2, 0.5)
    model.addConstr(M@w == m, name="sum-to-half")

    # conic constraint which comes from robust optimization
    model.addConstr(z**2 >= w.transpose()@omega@w, name="uncertainty")

    # optional controls to stop Gurobi taking too long
    if time_limit: model.setParam(GRB.Param.TimeLimit, time_limit)
    if min_duality_gap: model.setParam('MIPGap', min_duality_gap)

    # model file can be used externally for verification
    if debug: model.write("robust-opt.mps")

    model.optimize()
    return w.X, z.X, model.ObjVal


def print_compare_solutions(
    w_std: np.ndarray,
    objective_std: float,
    w_rbs: np.ndarray,
    z_rbs: float,
    objective_rbs: float,
    dimension: int,
    precision: int = 5
) -> None:
    """
    TODO write a docstring for comparing two robust genetics portfolios
    """

    order = len(str(dimension))
    print(" "*(order-1) + "i   w_std    w_rbs")
    for candidate in range(dimension):
        print(
            f"{candidate+1:0{order}d}  " \
            f"{w_std[candidate]:.{precision}f}  " \
            f"{w_rbs[candidate]:.{precision}f}" 
        )

    print(
        f"\nStandard Obj.:  {objective_std:.{precision}f}" \
        f"\nRobust Obj:     {objective_rbs:.{precision}f} (z = {z_rbs:.{precision}f})" \
        f"\nMaximum change: {max(np.abs(w_std-w_rbs)):.{precision}f}" \
        f"\nAverage change: {np.mean(np.abs(w_std-w_rbs)):.{precision}f}" \
        f"\nMinimum change: {min(np.abs(w_std-w_rbs)):.{precision}f}"
    )


# MAIN

# key problem variables
sigma, mubar, omega, n = load_problem(
    "Example/04/A04.txt",
    "Example/04/EBV04.txt",
    "Example/04/S04.txt"
)

# NOTE this trick of handling sex data is specific to the initial simulation data
# which is structured so that candidates alternate between sires (which have even
# indices) and dams (which have odd indices).
sires = range(0, n, 2)
dams  = range(1, n, 2)

lam = 0.5
kap = 1

# actually finding the standard genetics solution
w_std, obj_std = gurobi_standard_genetics(sigma, mubar, sires, dams, lam, n)
z_std = -1  # HACK placeholder for missing value

# not specifying the kappa argument, so will solve the non-robust problem
# w_std, z_std, obj_std = gurobi_robust_genetics(sigma, mubar, omega, sires, dams, lam, n)
w_rbs, z_rbs, obj_rbs = gurobi_robust_genetics(sigma, mubar, omega, sires, dams, lam, n, kappa=kap)

print_compare_solutions(w_std, obj_std, w_rbs, z_rbs, obj_rbs, n)
