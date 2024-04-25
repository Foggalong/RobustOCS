import numpy as np                  # defines matrix structures
import gurobipy as gp               # Gurobi optimization interface (1)
from gurobipy import GRB            # Gurobi optimization interface (2)

problem_size = 2
expected_breeding_values = np.array([
    1.0,
    2.0
])
relationship_matrix = np.array([
    [1, 0],
    [0, 1]
])
#lower_bound = np.full((problem_size, 1), 0.0)
#upper_bound = np.full((problem_size, 1), 1.0)

omega = np.array([
    [1/9, 0],
    [0, 4]
])

kappa = 0.5

# OPTIMIZATION SETUP VARIABLES
lam = 0.1
# define the M so that column i is [1;0] if i is a sire and [0;1] otherwise 
M = np.ones(problem_size)
# define the right hand side of the constraint Mx = m
m = 1

# create a model for standard genetic selection
model = gp.Model("standardGS")

# define variable of interest as a continuous
#
#JAJH No need to define the lower bounds separately, and no need to define the
#upper bounds of 1 at all, since the sum of w's is to at most 1
w = model.addMVar(shape=problem_size, lb=0.0, vtype=GRB.CONTINUOUS, name="w")

# set the objective function
model.setObjective(
    w.transpose()@expected_breeding_values - (lam/2)*w@(relationship_matrix@w),
GRB.MAXIMIZE)

# add sum-to-one constraint
model.addConstr(M @ w == 1, name="sum-to-one")

# add weight-bound constraints
#
#JAJH No longer necessary, but avoid giving names with spaces, as they
#aren't allowed if you try to write out a model as an MPS file
#
#model.addConstr(w >= lower_bound, name="lower bound")
#model.addConstr(w <= upper_bound, name="upper bound")

# solve the problem with Gurobi
model.optimize()
print("\n****\nMean variance model has solution", f"w = {w.X}\n****")
model.write("MeanVar.mps")

# create a new model for robust genetic selection
model = gp.Model("robustGS")

# define variables of interest as a continuous
#
#JAJH Ditto add lower bound here
w = model.addMVar(shape=problem_size, lb=0, vtype=GRB.CONTINUOUS, name="w")
z = model.addVar(name="z")

# setup the robust objective function
model.setObjective(
    w.transpose()@expected_breeding_values - (lam/2)*w@(relationship_matrix@w) - kappa*z,
GRB.MAXIMIZE)

# add quadratic uncertainty constraint

#JAJH This was the crux.
#
#JAJH Gurobi doesn't interpret np.inner(w, omega@w) as
#w^T*Omega*w. Observe this by flipping the definition and seeing the
#ratio between the optimal value of z**2 and the optimal value of
#np.inner(w, omega@w) change. It's more dramatic for the value kappa=1
#(that I was using) since the ratio flips between 1 and 0.5.
#
#JAJH At the very least this is odd behaviour that I should report to
#Gurobi

model.addConstr(z**2 >= w.transpose()@omega@w, name="uncertainty")
#model.addConstr(z**2 >= np.inner(w, omega@w), name="uncertainty")


#JAJH z >= 0 isn't needed - and is better defined in model.addVar(lb=0, name="z")
#
#model.addConstr(z >= 0, name="z positive")

#JAJH Same mods as above
#
# add sum-to-one constraint
model.addConstr(M @ w == 1, name="sum-to-one")
# add weight-bound constraints~
#model.addConstr(w >= lower_bound, name="lower bound")
#model.addConstr(w <= upper_bound, name="upper bound")

# solve the problem with Gurobi
model.optimize()

#JAJH To try to identify where the differences were, I computed the
#optimal values of the linear_term, cone_term and quad_term in my C++,
#and compared their sum with the optimal objective from HiGHS. Then I
#did the same with Gurobi, and spotted that the difference was the
#cone_term, and that computed_objective wasn't the same as
#gurobi_objective.

#JAJH All this was due to the cone terms computed from z and w
#being different, when they should be the same, given the defined
#constraint, so I tried the "simpler" constraint definition z**2 >=
#w.transpose()@omega@w
#
#JAJH I'd previously tried z*z instead of z**2, just in case :D

linear_term = w.transpose().X @expected_breeding_values;
cone_term = -kappa * (np.inner(w.X, omega@w.X))**(0.5)
gurobi_cone_term = -kappa * z.X
quad_term = -(lam/2) * np.inner(w.X, relationship_matrix@w.X)
gurobi_objective = model.getObjective().getValue()
computed_objective = linear_term + cone_term + quad_term
objective_error = abs(gurobi_objective-computed_objective)
print("Gurobi cone term =", gurobi_cone_term, "; computed cone term =", cone_term)
print("Gurobi z**2 =", (z.X)**2, "; w.Omega.w =", np.inner(w.X, omega@w.X), "; ratio = ", (z.X)**2 / np.inner(w.X, omega@w.X))
print("Objective is", linear_term, "+", cone_term, "+", quad_term, "=", computed_objective, "Gurobi objective =", gurobi_objective, "with error =", objective_error)
print("\n****\nRobust optimization model has solution", f"w = {w.X}\n****")
print(f"w = {w.X},\nz = {z.X:.5f}.")
model.write("Robust.mps")

#JAJH Comparing the MPS files helped me, too. but more of that tomorrow

