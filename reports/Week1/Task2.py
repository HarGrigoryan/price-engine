"""
Cobb-Douglas General Equilibrium: Market Clearing via Root-Finding
==================================================================
Finds prices p1, p2 such that total demand = total supply for each good.
Normalization: p1 + p2 = 1 (only relative prices matter).
"""

import numpy as np
from scipy.optimize import fsolve
import warnings

# ── Parameters ────────────────────────────────────────────────────────────────

# Each consumer i has preference weight alpha_i and endowment (e1_i, e2_i)
# Feel free to change these values to experiment.
alphas = np.array([0.3, 0.5, 0.7, 0.6])       # Cobb-Douglas alpha per consumer
endowments_1 = np.array([1.0, 2.0, 1.5, 0.5]) # Endowment of good 1
endowments_2 = np.array([2.0, 1.0, 0.5, 1.5]) # Endowment of good 2

# Total supply (fixed)
S1 = endowments_1.sum()
S2 = endowments_2.sum()

# ── Demand Functions ───────────────────────────────────────────────────────────

def demand(p1: float, p2: float, alpha: float, e1: float, e2: float):
    """
    Optimal Cobb-Douglas demand for one consumer.
      x1* = alpha  * w / p1
      x2* = (1-alpha) * w / p2
    where wealth w = p1*e1 + p2*e2.
    """
    w = p1 * e1 + p2 * e2
    x1 = alpha * w / p1
    x2 = (1 - alpha) * w / p2
    return x1, x2

def total_demand(p1: float, p2: float):
    """Aggregate demand across all consumers."""
    D1 = sum(demand(p1, p2, a, e1, e2)[0]
             for a, e1, e2 in zip(alphas, endowments_1, endowments_2))
    D2 = sum(demand(p1, p2, a, e1, e2)[1]
             for a, e1, e2 in zip(alphas, endowments_1, endowments_2))
    return D1, D2

# ── Excess Demand (Walras' Law: one equation is redundant) ─────────────────────

def excess_demand_system(p1: float) -> float:
    """
    With normalization p2 = 1 - p1, return excess demand for good 1.
    At equilibrium this equals zero.
    """
    p2 = 1.0 - p1
    if p1 <= 0 or p2 <= 0:
        return np.inf
    D1, _ = total_demand(p1, p2)
    return D1 - S1

# ── Root-Finding ───────────────────────────────────────────────────────────────

def find_equilibrium_numerical():
    """Use scipy fsolve to find the equilibrium price p1* ∈ (0,1)."""
    p1_init = 0.5  # initial guess
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        p1_sol, info, ier, msg = fsolve(
            excess_demand_system, p1_init, full_output=True
        )
    p1_star = float(p1_sol[0])
    p2_star = 1.0 - p1_star
    converged = ier == 1
    return p1_star, p2_star, converged

# ── Analytical Solution ────────────────────────────────────────────────────────

def find_equilibrium_analytical():
    """
    Closed-form equilibrium price ratio.
    From market clearing for good 1:
      sum_i [ alpha_i * (p1*e1_i + p2*e2_i) / p1 ] = S1
    Rearranging with p2 = 1 - p1:
      p1* = (sum alpha_i * e1_i) / (S1 + sum alpha_i * (e2_i - e1_i)... )
    Solved directly:
      p1* = A / (A + B)   where A = sum(alpha_i * e1_i), B = sum((1-alpha_i)*e1_i) ... 
    
    General closed form via Walras market clearing:
      p1/p2 = [sum alpha_i * w_i / p1] = S1  → solve linearly.
    """
    # From Z1(p) = 0 with normalization p1+p2=1:
    # sum_i alpha_i*(p1*e1i + (1-p1)*e2i) / p1 = S1
    # sum_i alpha_i * e1i + (1-p1)/p1 * sum_i alpha_i*e2i = S1
    A = (alphas * endowments_1).sum()   # sum alpha_i * e1_i
    B = (alphas * endowments_2).sum()   # sum alpha_i * e2_i
    # A + B*(1-p1)/p1 = S1
    # A*p1 + B*(1-p1) = S1*p1
    # p1*(A - B - S1) = -B
    # p1 = B / (B + S1 - A)
    p1_star = B / (B + S1 - A)
    p2_star = 1.0 - p1_star
    return p1_star, p2_star

# ── Validation & Report ────────────────────────────────────────────────────────

def report():
    print("=" * 60)
    print("  Cobb-Douglas General Equilibrium — Market Clearing")
    print("=" * 60)
    print(f"\n  Consumers      : {len(alphas)}")
    print(f"  Total supply 1 : {S1:.4f}")
    print(f"  Total supply 2 : {S2:.4f}")

    # Numerical solution
    p1_num, p2_num, ok = find_equilibrium_numerical()
    D1_num, D2_num = total_demand(p1_num, p2_num)
    print(f"\n  ── Numerical Solution (fsolve) ──")
    print(f"  Converged      : {ok}")
    print(f"  p1*            : {p1_num:.6f}")
    print(f"  p2*            : {p2_num:.6f}")
    print(f"  Total demand 1 : {D1_num:.6f}  (supply = {S1:.6f})")
    print(f"  Total demand 2 : {D2_num:.6f}  (supply = {S2:.6f})")
    print(f"  Excess demand 1: {D1_num - S1:.2e}")
    print(f"  Excess demand 2: {D2_num - S2:.2e}")

    # Analytical solution
    p1_ana, p2_ana = find_equilibrium_analytical()
    D1_ana, D2_ana = total_demand(p1_ana, p2_ana)
    print(f"\n  ── Analytical Solution (closed-form) ──")
    print(f"  p1*            : {p1_ana:.6f}")
    print(f"  p2*            : {p2_ana:.6f}")
    print(f"  Total demand 1 : {D1_ana:.6f}  (supply = {S1:.6f})")
    print(f"  Total demand 2 : {D2_ana:.6f}  (supply = {S2:.6f})")

    # Comparison
    diff = abs(p1_num - p1_ana)
    print(f"\n  ── Validation ──")
    print(f"  |p1_num - p1_ana| = {diff:.2e}")
    print(f"  {'✓ Solutions match!' if diff < 1e-6 else '✗ Discrepancy detected.'}")

    # Individual allocations at equilibrium
    print(f"\n  ── Individual Allocations at p* ──")
    print(f"  {'Consumer':>10} {'alpha':>7} {'x1*':>10} {'x2*':>10}")
    print(f"  {'-'*40}")
    for i, (a, e1, e2) in enumerate(zip(alphas, endowments_1, endowments_2)):
        x1, x2 = demand(p1_ana, p2_ana, a, e1, e2)
        print(f"  {i+1:>10} {a:>7.2f} {x1:>10.4f} {x2:>10.4f}")
    print("=" * 60)

if __name__ == "__main__":
    report()