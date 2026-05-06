"""
Cobb-Douglas Market Equilibrium: Numerical Root-Finding (Task 2)
================================================================
Pure-exchange economy with N consumers and K = 2 goods.

Each consumer j has:
  * Cobb-Douglas preference weights alpha[j, :], with rows summing to 1
    (alpha_jk is the spending share consumer j devotes to good k);
  * exogenous wealth w_j (dollars).

The economy has a fixed total endowment E_k of each good. Goal: find
prices p such that aggregate demand equals total supply for each good,

        D_k(p)  =  sum_j (alpha_jk * w_j) / p_k  =  E_k.

Two solvers:
  1. scipy.optimize.fsolve on the excess-demand vector
     (numerical, the Task 2 deliverable).
  2. Closed-form  p_k* = sum_j (alpha_jk * w_j) / E_k
     (analytical, derived in Task 1, used here as ground truth).

Both methods are run on the same parameters and the results are
compared. Prices are reported in raw form and after normalising onto
the simplex p_1 + p_2 = 1 (the convention used elsewhere in the
project for comparison with the social planner's dual variables).
"""

import warnings

import numpy as np
from scipy.optimize import fsolve


# ── Parameters: project's standard test economy ───────────────────────────────

# alpha[j, k] = consumer j's Cobb-Douglas spending share on good k.
# Rows sum to 1.
alpha = np.array([
    [0.6, 0.4],   # consumer A
    [0.3, 0.7],   # consumer B
    [0.5, 0.5],   # consumer C
])
w = np.array([100.0, 80.0, 60.0])     # consumer wealths (dollars)
E = np.array([30.0,  40.0])           # total endowment per good

N, K = alpha.shape


# ── Demand and Excess Demand ──────────────────────────────────────────────────

def aggregate_demand(p):
    """Aggregate Cobb-Douglas demand at prices p.

    D_k(p) = sum_j (alpha_jk * w_j) / p_k.
    """
    return (alpha * w[:, None]).sum(axis=0) / p


def excess_demand(p):
    """Aggregate demand minus total supply, per good."""
    return aggregate_demand(p) - E


# ── Numerical Equilibrium via Root-Finding ────────────────────────────────────

def find_equilibrium_numerical(p_init=None):
    """Solve excess_demand(p) = 0 using scipy.optimize.fsolve."""
    if p_init is None:
        p_init = np.ones(K)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        sol, _, ier, _ = fsolve(excess_demand, p_init, full_output=True)
    converged = ier == 1
    return sol, converged


# ── Closed-Form Equilibrium (Task 1 ground truth) ─────────────────────────────

def find_equilibrium_analytical():
    """Closed-form market-clearing prices: p_k* = sum_j alpha_jk w_j / E_k."""
    return (alpha * w[:, None]).sum(axis=0) / E


def normalise(p):
    """Project prices onto the simplex p_1 + p_2 + ... = 1."""
    return p / p.sum()


# ── Reporting ─────────────────────────────────────────────────────────────────

def report():
    print("=" * 64)
    print("  Cobb-Douglas Market Equilibrium  -  Task 2")
    print("=" * 64)
    print(f"  Consumers : {N}        Goods : {K}")
    print(f"  alpha (rows = consumers, cols = goods):")
    for j in range(N):
        print(f"      consumer {chr(65+j)}: {alpha[j]}")
    print(f"  wealths    : {w}")
    print(f"  endowments : {E}")

    p_num, ok = find_equilibrium_numerical()
    print(f"\n  Numerical (fsolve)")
    print(f"    converged      : {ok}")
    print(f"    raw prices     : {p_num}")
    print(f"    normalised     : {normalise(p_num)}")
    print(f"    excess demand  : {excess_demand(p_num)}")

    p_ana = find_equilibrium_analytical()
    print(f"\n  Closed-form (Task 1 aggregation)")
    print(f"    raw prices     : {p_ana}")
    print(f"    normalised     : {normalise(p_ana)}")

    diff = float(np.max(np.abs(p_num - p_ana)))
    print(f"\n  Validation")
    print(f"    |p_numerical - p_closed_form|  =  {diff:.2e}")
    flag = "Solutions agree." if diff < 1e-6 else "Discrepancy detected."
    print(f"    {flag}")

    print(f"\n  Individual allocations at p*")
    print(f"  {'Consumer':>9} {'alpha_1':>9} {'alpha_2':>9}"
          f" {'wealth':>9} {'x_1':>10} {'x_2':>10}")
    print(f"  {'-' * 60}")
    for j in range(N):
        x = alpha[j] * w[j] / p_num
        print(f"  {chr(65+j):>9} {alpha[j, 0]:>9.2f} {alpha[j, 1]:>9.2f}"
              f" {w[j]:>9.1f} {x[0]:>10.4f} {x[1]:>10.4f}")
    print(f"  totals  =  {(alpha * w[:, None] / p_num).sum(axis=0)}"
          f"   (should equal E = {E})")
    print("=" * 64)


if __name__ == "__main__":
    report()
