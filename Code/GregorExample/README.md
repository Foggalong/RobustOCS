# File Description

In the original simulation, we had 12k individuals and 1k samples of their estimated breeding values (EBV) – our selection criterion.

Based on that simulated data:

- I constructed the A-matrix (12k x 12k), this is measure of co-ancestry between individuals based on the pedigree data. This is the standard used in optimal contributions, but it can be constructed in different ways (we can later test the DNA method vs pedigree method, etc.). [A50]
- I constructed the S-matrix (12k x 12k), which is the covariance matrix between 1k EBV samples for 12k individuals. [S50]
- The EBV vector (12k), which is posterior mean over 1k samples of EBV. [EBV50]
 
Since 12k might be a lot, I have extracted last 50 rows/columns from above 12k set, for your test. This corresponds to the youngest 50 individuals (I don’t know the implications of choosing this non-random sample ).

The matrices are saved in row, column, value format like we talked about before.

_Note: this is an unedited version of Gregor's description of the files._
