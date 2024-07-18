# Examples

- [04/](04/) A scaled up version of the small $n = 2$ example created by Julian, replicating the same problem across sires and dams.
- [50/](50/) The 50 youngest individuals from a generated cohort of 12,000. Implications of this non-random example are unknown.

## File Description

In the original simulation, we had 12k individuals and 1k samples of their estimated breeding values (EBV) – our selection criterion.

Based on that simulated data:

- I constructed the A-matrix (12k x 12k), this is measure of co-ancestry between individuals based on the pedigree data. This is the standard used in optimal contributions, but it can be constructed in different ways (we can later test the DNA method vs pedigree method, etc.). [A50]
- I constructed the S-matrix (12k x 12k), which is the covariance matrix between 1k EBV samples for 12k individuals. [S50]
- The EBV vector (12k), which is posterior mean over 1k samples of EBV. [EBV50]

The matrices are saved in row, column, value format like we talked about before.

_Note: this is an unedited version of Gregor's description of the files._

## Times

To give an idea of how the methods available scale with dimension, the table below times each for four example problems of increasing size.

|   $n$ | Gurobi (standard) | HiGHS (standard) | Gurobi (conic) | Gurobi (SQP) | HiGHS (SQP) |
| ----: | ----------------: | ---------------: | -------------: | -----------: | ----------: |
|     4 |           2.95e-3 |          4.78e-4 |        4.71e-3 |      1.74e-2 |     5.98e-3 |
|    50 |           4.46e-3 |          1.02e-3 |        1.02e-2 |      5.52e-2 |     1.84e-2 |
|  1000 |           6.76e-1 |          2.04e-1 |        2.75e+0 |      2.64e+1 |     1.68e+0 |
| 10000¹ |           8.63e+1 |          2.58e+1 |           DNF² |      1.56e+3 |     1.06e+2 |

_1: This repository doesn't contain the data files due to storage limitations. Contact the authors for access._

_2: Gurobi crashed without displaying an error message when attempting to solve using conic programming for the largest problem._
