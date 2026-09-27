An institutional-grade Quantitative Option Pricing Engine written in Python, accelerated for Apple Silicon M2 execution.

## Features
- **Black-Scholes-Merton Analytical Model**: European option benchmarks and closed-form Greeks.
- **Cox-Ross-Rubinstein (CRR) Binomial Tree**: 20-step trees for standard European/American options and Q2 European compound options.
- **Monte Carlo Simulation Engine**: Parallel path generation with Owen-scrambled Sobol Sequences (QMC), Antithetic Variates, and Black-Scholes Control Variates.
- **Longstaff-Schwartz (LSM) Engine**: American option pricing using backward induction regression on orthogonal Laguerre basis polynomials ($L_0, L_1, L_2, L_3$).
- **Greeks Estimation**: Bump-and-revalue central finite differences for Delta, Gamma, Vega, Theta, and Rho.```bash

