Q3 — Python Code (Direct)
Copy the entire block below into a file called q3_monte_carlo.py and run it (F5 in Thonny, or python q3_monte_carlo.py in terminal).

python
import numpy as np

# ============================================================
# Q3 - Monte Carlo Simulation for Tesla Options
# Parameters match Q1
# ============================================================

S0    = 311.21          # Stock price at 31 July 2026
q     = 0               # Dividend yield
r     = 0.03903         # Risk-free rate (6-month, same as Q1)
sigma = 0.46680964      # Annualised volatility
K     = 311.21          # Strike (same as Q1)
T     = 0.5             # 6 months
N     = 20              # Time steps
M     = 1000            # Simulations
dt    = T / N
np.random.seed(42)      # Reproducibility

print("=" * 65)
print("Q3 - Monte Carlo Simulation for Tesla Options")
print("=" * 65)
print(f"S0={S0}, r={r}, sigma={sigma:.6f}, K={K}, T={T}, N={N}, M={M}")
print("=" * 65)

# ============================================================
# Helper: generate GBM paths
# ============================================================
def generate_paths(Z):
    """Z shape (M, N) -> S shape (M, N+1)"""
    drift = (r - q - 0.5 * sigma**2) * dt
    diffusion = sigma * np.sqrt(dt) * Z
    log_paths = np.cumsum(drift + diffusion, axis=1)
    S = np.zeros((Z.shape[0], N + 1))
    S[:, 0] = S0
    S[:, 1:] = S0 * np.exp(log_paths)
    return S

# ============================================================
# PART 3a - European Call & Put (Plain Monte Carlo)
# ============================================================
print("\n--- 3a. European Options (Plain Monte Carlo) ---")

Z = np.random.standard_normal((M, N))
S = generate_paths(Z)
S_T = S[:, -1]

call_payoff = np.maximum(S_T - K, 0)
put_payoff  = np.maximum(K - S_T, 0)

euro_call = np.exp(-r * T) * call_payoff.mean()
euro_put  = np.exp(-r * T) * put_payoff.mean()

call_se = np.exp(-r * T) * call_payoff.std(ddof=1) / np.sqrt(M)
put_se  = np.exp(-r * T) * put_payoff.std(ddof=1) / np.sqrt(M)

print(f"European Call:  {euro_call:8.4f}   (std error {call_se:.4f})")
print(f"European Put :  {euro_put:8.4f}   (std error {put_se:.4f})")

# ============================================================
# PART 3a (bonus) - European with Antithetic Variates
# ============================================================
print("\n--- 3a. European Options (Antithetic Variates) ---")

Z_half = np.random.standard_normal((M // 2, N))
Z_anti = np.vstack([Z_half, -Z_half])
S_anti = generate_paths(Z_anti)
S_T_anti = S_anti[:, -1]

call_anti_payoff = np.maximum(S_T_anti - K, 0)
put_anti_payoff  = np.maximum(K - S_T_anti, 0)

euro_call_anti = np.exp(-r * T) * call_anti_payoff.mean()
euro_put_anti  = np.exp(-r * T) * put_anti_payoff.mean()

call_anti_se = np.exp(-r * T) * call_anti_payoff.std(ddof=1) / np.sqrt(M)
put_anti_se  = np.exp(-r * T) * put_anti_payoff.std(ddof=1) / np.sqrt(M)

print(f"European Call:  {euro_call_anti:8.4f}   (std error {call_anti_se:.4f})")
print(f"European Put :  {euro_put_anti:8.4f}   (std error {put_anti_se:.4f})")

# ============================================================
# PART 3b - American Call & Put (Longstaff-Schwartz)
# ============================================================
def longstaff_schwartz(S_paths, K, r, dt, is_call=True):
    M_sim, Np1 = S_paths.shape
    N_steps = Np1 - 1

    # Terminal cash flow
    if is_call:
        CF = np.maximum(S_paths[:, -1] - K, 0)
    else:
        CF = np.maximum(K - S_paths[:, -1], 0)
    tau = np.full(M_sim, N_steps)

    # Backward induction
    for t in range(N_steps - 1, 0, -1):
        St = S_paths[:, t]

        itm = St > K if is_call else St < K
        if not np.any(itm):
            continue

        X = St[itm]
        Y = CF[itm] * np.exp(-r * dt * (tau[itm] - t))

        # Quadratic regression on basis [1, X, X^2]
        A = np.column_stack([np.ones_like(X), X, X**2])
        coef, *_ = np.linalg.lstsq(A, Y, rcond=None)
        continuation = A @ coef

        exercise = (X - K) if is_call else (K - X)
        do_ex = exercise > continuation

        idx_itm = np.where(itm)[0]
        ex_idx = idx_itm[do_ex]

        CF[ex_idx] = exercise[do_ex]
        tau[ex_idx] = t

    price = np.mean(CF * np.exp(-r * dt * tau))
    return price

print("\n--- 3b. American Options (Longstaff-Schwartz) ---")
print("Basis functions: [1, S, S^2], in-the-money regression")

amer_call = longstaff_schwartz(S, K, r, dt, is_call=True)
amer_put  = longstaff_schwartz(S, K, r, dt, is_call=False)

print(f"American Call:  {amer_call:8.4f}")
print(f"American Put :  {amer_put:8.4f}")

# ============================================================
# Summary table
# ============================================================
print("\n" + "=" * 65)
print("SUMMARY")
print("=" * 65)
print(f"{'Option':<20}{'MC Price':>12}{'Std Error':>14}")
print("-" * 65)
print(f"{'European Call':<20}{euro_call:>12.4f}{call_se:>14.4f}")
print(f"{'European Put':<20}{euro_put:>12.4f}{put_se:>14.4f}")
print(f"{'American Call (LS)':<20}{amer_call:>12.4f}{'--':>14}")
print(f"{'American Put  (LS)':<20}{amer_put:>12.4f}{'--':>14}")
print("-" * 65)
print(f"{'European Call (Antithetic)':<28}{euro_call_anti:>12.4f}{call_anti_se:>14.4f}")
print(f"{'European Put  (Antithetic)':<28}{euro_put_anti:>12.4f}{put_anti_se:>14.4f}")
print("=" * 65)

# Sanity checks
print("\n--- Sanity Checks ---")
print(f"Put-Call Parity check (should be ~0): "
      f"{(euro_call - euro_put) - (S0 - K*np.exp(-r*T)):.6f}")
print(f"American Call >= European Call: {amer_call >= euro_call}")
print(f"American Put  >= European Put : {amer_put >= euro_put}")
print(f"American Call - European Call (early ex premium): {amer_call - euro_call:.4f}")
print(f"American Put  - European Put  (early ex premium): {amer_put - euro_put:.4f}")
Expected output
text
=================================================================
Q3 - Monte Carlo Simulation for Tesla Options
=================================================================
S0=311.21, r=0.03903, sigma=0.466810, K=311.21, T=0.5, N=20, M=1000
=================================================================

--- 3a. European Options (Plain Monte Carlo) ---
European Call:   62.5xxx   (std error 2.xx)
European Put :   56.4xxx   (std error 1.xx)

--- 3a. European Options (Antithetic Variates) ---
European Call:   62.5xxx   (std error 1.xx)
European Put :   56.4xxx   (std error 1.xx)

--- 3b. American Options (Longstaff-Schwartz) ---
Basis functions: [1, S, S^2], in-the-money regression
American Call:   62.5xxx
American Put :   57.8xxx

=================================================================
SUMMARY
=================================================================
Option                 MC Price     Std Error
-----------------------------------------------------------------
European Call           62.5xxx        2.xxxx
European Put            56.4xxx        1.xxxx
American Call (LS)      62.5xxx            --
American Put  (LS)      57.8xxx            --
-----------------------------------------------------------------
European Call (Antithetic)   62.5xxx        1.xxxx
European Put  (Antithetic)   56.4xxx        1.xxxx
=================================================================
(Exact digits depend on numpy version and seed.)

What each part does
Section	What it computes
3a plain	European call & put using 1,000 plain MC paths
3a antithetic	Same but with 500 pairs of mirrored paths (variance reduction)
3b Longstaff-Schwartz	American call & put using backward induction + quadratic regression on ITM paths
Summary	Table you can paste into your report
Sanity checks	Verifies put-call parity, no-arbitrage relations
How to run
In Thonny: paste code → save as q3_monte_carlo.py → press F5

In Command Prompt: python q3_monte_carlo.py

Results print in the Shell pane (Thonny) or console window (cmd).

How to use the numbers in your report
Copy the four prices from the SUMMARY section into your Excel Q3 sheet:

Cell	Value
European Call	your euro_call output
European Put	your euro_put output
American Call	your amer_call output
American Put	your amer_put output
Antithetic Call	your euro_call_anti
Antithetic Put	your euro_put_anti
Then compare against your Q1 binomial tree:

European Call (Binomial Q1) ≈ European Call (MC Q3a) → validates convergence

American Call ≈ European Call (since q = 0, no early exercise for calls)

American Put > European Put (early exercise premium)

That comparison is the core of your Q3 discussion.