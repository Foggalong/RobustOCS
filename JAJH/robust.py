import numpy as np                  # defines matrix structures
import gurobipy as gp               # Gurobi optimization interface (1)
from gurobipy import GRB            # Gurobi optimization interface (2)

def load_problem(A_filename, E_filename, S_filename, dimension=False):
    """
    Used to load genetic selection problems into NumPy. It takes three
    string inputs for filenames where Sigma, Mu, and Omega are stored,
    as well as an optional integer input for problem dimension if this
    is known. If it's know know, it's worked out based on E_filename.

    As output, it returns (A, E, S, n), where A and S are n-by-n NumPy
    arrays, E is a length n NumPy array, and n is an integer.
    """

    def load_symmetric_matrix(filename, dimension):
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
                if int(i) == dimension:
                    break

        return matrix

    # EBV isn't in coordinate format so can be loaded directly
    FullE = np.loadtxt(E_filename)
    E = np.zeros([dimension])
    #JAJH I tried to write a method to load just the first "dimension"
    #values from the file, but failed - Python is a horrible language!
    for i in range(dimension):
        E[i] = FullE[i]
    # A and S are stored by coordinates so need special loader
    A = load_symmetric_matrix(A_filename, dimension)
    S = load_symmetric_matrix(S_filename, dimension)

    return A, E, S, dimension

dimension = 1000
if (dimension <= 50):
    A_file = "Example/A50.txt"
    EBV_file = "Example/EBV50.txt"
    S_file = "Example/S50.txt"
else:
    A_file = "Example/A1000.txt"
    EBV_file = "Example/EBV1000.txt"
    S_file = "Example/S1000.txt"

sigma, mubar, omega, n = load_problem(A_file, EBV_file, S_file, dimension)

# define the M so that column i is [1;0] if i is a sire (so even) and [0;1] otherwise 
M = np.zeros((2, n))
M[0, range(0,n,2)] = 1
M[1, range(1,n,2)] = 1

lam=2
kappa=0.5

model_std = gp.Model("standard")
model_rbs = gp.Model("robust")

# initialise w for both models, z for robust model
w_std = model_std.addMVar(shape=n, lb=0, vtype=GRB.CONTINUOUS, name="w") 
w_rbs = model_rbs.addMVar(shape=n, lb=0, vtype=GRB.CONTINUOUS, name="w")
z_rbs = model_rbs.addVar(name="z")

# define the objective functions for both models
model_std.setObjective(
    w_std.transpose() @ mubar - (lam/2)*w_std.transpose()@(sigma@w_std),
GRB.MAXIMIZE)

model_rbs.setObjective(
    # Gurobi does offer a way to set one objective in terms of another, i.e.
    # we could use `model_std.getObjective() - lam*kappa*z_rbs` to define this
    # robust objective, but it results in a significant slowdown in code.
    w_rbs.transpose() @ mubar - (lam/2)*w_rbs.transpose()@(sigma@w_rbs) - kappa*z_rbs,
GRB.MAXIMIZE)
# JAJH This is the definition in the Jupyter notebook
#
#   w_rbs.transpose() @ mubar - (lam/2)*w_rbs.transpose()@(sigma@w_rbs) - kappa*z,
#
# JAJH However, z "isn't known" when I cut the Python from your
# Jupyter notebook to produce the first version of robust.py - but
# inside it seems to be interpreted. Presumably the notebook has some
# historical definition of z. Another scary feature of Jupyter.
#
# JAJH I insist that you stop using Jupyter notebooks for this work:
# they introduce too much scope for error.
#
#model_rbs.addConstr(z**2 >= np.inner(w_rbs, omega@w_rbs), name="uncertainty")

# add sum-to-half constraints to both models
model_std.addConstr(M @ w_std == 0.5, name="sum-to-half")
model_rbs.addConstr(M @ w_rbs == 0.5, name="sum-to-half")

# add quadratic uncertainty constraint to the robust model
#
# JAJH This definition
#
# model_rbs.addConstr(z_rbs**2 >= np.inner(w_rbs, omega@w_rbs), name="uncertainty")
#
# yields n separate quadratic constraints of the form
# -w^T.Q_i.w + z**2 \ge 0, but each Q_i is zero except for row and
# column i. Hence it's not positive definite so Gurobi treats the
# problem as being non-convex. Calling
# model_rbs.setParam(GRB.Param.NonConvex, 2) is necessary, but then
# Gurobi takes forever because it handles the non-convexity by
# linearizing using integer variables, and solving a MIP. See MPS
# files for n=2
#
# JAJH In the toy n=2 and n=3 problems, Omega is diagonal, so the
# corresponding quadratic constraints only involve w_i and z, so don't
# lead to the non-convexity

model_rbs.addConstr(z_rbs**2 >= w_rbs.transpose()@omega@w_rbs, name="uncertainty")

model_std.write("MeanVar.mps")
model_rbs.write("Robust.mps")

# since working with non-trivial size, set a time limit
#time_limit = 60*5  # 5 minutes
#model_std.setParam(GRB.Param.TimeLimit, time_limit)
#model_rbs.setParam(GRB.Param.TimeLimit, time_limit)

# solve both problems with Gurobi
model_std.optimize()
model_rbs.optimize()

print("\n****\nFor n =", n, ", lambda =", lam, " and kappa =", kappa)
print("Mean variance       model objective", model_std.getObjective().getValue())
print("Robust optimization model objective", model_rbs.getObjective().getValue(), "\n****")

# HACK code which prints the results for comparison in a nice format - JAJH but not if n is too large
limit = int(n/2)
if limit < 100:
    print("\nSIRE WEIGHTS\t\t\t DAM WEIGHTS")
    print("-"*20 + "\t\t " + "-"*20)
    print(" i   w_std    w_rbs\t\t  i   w_std    w_rbs")
    for candidate in range(limit):
        print(f"{candidate*2:02d}  {w_std.X[candidate*2]:.5f}  {w_rbs.X[candidate*2]:.5f} \
        {candidate*2+1:02d}  {w_std.X[candidate*2+1]:.5f}  {w_rbs.X[candidate*2+1]:.5f}")
    
print(f"\nMaximum change: {max(np.abs(w_std.X-w_rbs.X)):.5f}")
print(f"Average change: {np.mean(np.abs(w_std.X-w_rbs.X)):.5f}")
print(f"Minimum change: {min(np.abs(w_std.X-w_rbs.X)):.5f}")
