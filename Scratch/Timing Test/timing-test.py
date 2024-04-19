import numpy as np                  # defines matrix structures
from qpsolvers import solve_qp      # used for quadratic optimization
import gurobipy as gp               # Gurobi optimization interface (1)
from gurobipy import GRB            # Gurobi optimization interface (2)

# want to round rather than truncate when printing
np.set_printoptions(threshold=np.inf)

# only show numpy output to five decimal places
np.set_printoptions(formatter={'float_kind':"{:.5f}".format})


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

        return matrix


    # if dimension wasn't supplied, need to find that
    if not dimension:
        # get dimension from EBV, since it's the smallest file
        with open(E_filename, 'r') as file:
            dimension = sum(1 for _ in file)

    # EBV isn't in coordinate format so can be loaded directly
    E = np.loadtxt(E_filename)  
    # A and S are stored by coordinates so need special loader
    A = load_symmetric_matrix(A_filename, dimension)
    S = load_symmetric_matrix(S_filename, dimension)

    return A, E, S, dimension


sigma, mubar, omega, n = load_problem(
    "../Example/A50.txt",
    "../Example/EBV50.txt",
    "../Example/S50.txt",
    50)

lam = 0.5
kappa = 2

# define the M so that column i is [1;0] if i is a sire (so even) and [0;1] otherwise 
M = np.zeros((2, n))
M[0, range(0,50,2)] = 1
M[1, range(1,50,2)] = 1
# define the right hand side of the constraint Mx = m
m = np.array([[0.5], [0.5]])

# create models for standard and robust genetic selection
model_std = gp.Model("n50standard")
model_rbs = gp.Model("n50robust")

# initialise w for both models, z for robust model
w_std = model_std.addMVar(shape=n, vtype=GRB.CONTINUOUS, name="w") 
w_rbs = model_rbs.addMVar(shape=n, vtype=GRB.CONTINUOUS, name="w")
z_rbs = model_rbs.addVar(name="z")

# define the objective functions for both models
model_std.setObjective(
    0.5*w_std@(sigma@w_std) - lam*w_std.transpose()@mubar,
GRB.MINIMIZE)

model_rbs.setObjective(
    # Gurobi does offer a way to set one objective in terms of another, i.e.
    # we could use `model_std.getObjective() - lam*kappa*z_rbs` to define this
    # robust objective, but it results in a significant slowdown in code.
    0.5*w_rbs@(sigma@w_rbs) - lam*w_rbs.transpose()@mubar - lam*kappa*z_rbs,
GRB.MINIMIZE)

# add sum-to-half constraints to both models
model_std.addConstr(M @ w_std == m, name="sum-to-half")
model_rbs.addConstr(M @ w_std == m, name="sum-to-half")

# add quadratic uncertainty constraint to the robust model
model_rbs.addConstr(z_rbs**2 <= np.inner(w_rbs, omega@w_rbs), name="uncertainty")
model_rbs.addConstr(z_rbs >= 0, name="z positive")

# since working with non-trivial size, set a time limit
time_limit = 60*5  # 5 minutes
model_std.setParam(GRB.Param.TimeLimit, time_limit)
model_std.setParam(GRB.Param.TimeLimit, time_limit)

# for the same reason, also set a duality gap tolerance
duality_gap = 0.009
model_std.setParam('MIPGap', duality_gap)
model_rbs.setParam('MIPGap', duality_gap)

# solve both problems with Gurobi
model_std.optimize()
model_rbs.optimize()

# HACK code which prints the results for comparison in a nice format
print("\nSIRE WEIGHTS\t\t\t DAM WEIGHTS")
print("-"*20 + "\t\t " + "-"*20)
print(" i   w_std    w_rbs\t\t  i   w_std    w_rbs")
for candidate in range(25):
    print(f"{candidate*2:02d}  {w_std.X[candidate*2]:.5f}  {w_rbs.X[candidate*2]:.5f} \
            {candidate*2+1:02d}  {w_std.X[candidate*2+1]:.5f}  {w_rbs.X[candidate*2+1]:.5f}")