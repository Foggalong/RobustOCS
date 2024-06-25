# Timings

This is how long to solve the $n = 1000$ example with each method at various commits. Timings in seconds (all shown to three significant figures) are the average of three runs, repeated five times with the best average taken. The first two solvers are standard non-robust optimization, and the last three are all robust optimization (conic or SQP).

| Commit  | Gurobi (standard) | HiGHS (standard) | Gurobi (conic) | Gurobi (SQP) | HiGHS (SQP) | Total |
| :-----: | ----------------: | ---------------: | -------------: | -----------: | ----------: | ----: |
| 7d866d8 |             0.688 |            0.197 |          2.690 |       24.400 |       1.700 | 492.8 |
| 90a6040 |             0.690 |            0.204 |          2.770 |       24.600 |       1.700 | 499.9 |
| 92ae275 |             0.676 |            0.204 |          2.750 |       24.600 |       1.680 | 507.3 |
| 48e645f |             0.675 |            0.205 |          2.750 |       24.600 |       1.680 | 508.7 |
