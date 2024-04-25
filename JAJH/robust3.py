import numpy as np                  # defines matrix structures
import gurobipy as gp               # Gurobi optimization interface (1)
from gurobipy import GRB            # Gurobi optimization interface (2)

problem_size = 3
expected_breeding_values = np.array([
    1.0,
    5.0,
    2.0
])
relationship_matrix = np.array([
    [1, 0, 0],
    [0, 5, 0],
    [0, 0, 3]
])
sire_indices = [0]
dam_indices  = [1,2]
lower_bound = np.full((problem_size, 1), 0.0)
upper_bound = np.full((problem_size, 1), 1.0)

# OPTIMIZATION SETUP VARIABLES
lam = 2
# define the M so that column i is [1;0] if i is a sire and [0;1] otherwise 
M = np.zeros((2, problem_size))
M[0, sire_indices] = 1
M[1, dam_indices] = 1
# define the right hand side of the constraint Mx = m
m = np.array([[0.5], [0.5]])

# create a model for standard genetic selection
model = gp.Model("standardGS")

# define variable of interest as a continuous 
w = model.addMVar(shape=problem_size, lb=0, vtype=GRB.CONTINUOUS, name="w")

# set the objective function
model.setObjective(
    w.transpose()@expected_breeding_values - (lam/2)*w@(relationship_matrix@w),
GRB.MAXIMIZE)

# add sum-to-half constraints
model.addConstr(M @ w == m, name="sum-to-half")

# solve the problem with Gurobi
model.optimize()
print("\n****\nMean variance model has solution", f"w = {w.X}\n****")
model.write("MeanVar.mps")

omega = np.array([
    [1, 0, 0],
    [0, 4, 0],
    [0, 0, 1/8]
])

kappa = 0.5

# create a new model for robust genetic selection
model = gp.Model("robustGS")

# define variables of interest as a continuous
w = model.addMVar(shape=problem_size, lb=0, vtype=GRB.CONTINUOUS, name="w")
z = model.addVar(name="z")

# setup the robust objective function
model.setObjective(
    w.transpose()@expected_breeding_values - (lam/2)*w@(relationship_matrix@w) - kappa*z,
GRB.MAXIMIZE)

# add quadratic uncertainty constraint
#model.addConstr(z**2 >= w.transpose()@omega@w, name="uncertainty")
model.addConstr(z**2 >= np.inner(w, omega@w), name="uncertainty")

# add sum-to-half constraints
model.addConstr(M @ w == 0.5, name="sum-to-half")

# solve the problem with Gurobi
model.optimize()
print("\n****\nRobust optimization model has solution", f"w = {w.X}\n****")
model.write("Robust.mps")

