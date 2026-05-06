"""
Cobb-Douglas Market Equilibrium: Tatonnement (Task 4)
=====================================================
Pure-exchange economy with N consumers and K = 2 goods, in the same
exogenous-wealth setup used by Task 2 and Task 3.

Tatonnement is the classical price-adjustment dynamic: at each step,
raise the price of any good with excess demand and lower the price
of any good with excess supply. Iterate until prices stabilise.

Update rule (discrete-time, proportional-to-excess-demand):
        p^(t+1)  =  p^(t)  +  eta * Z(p^(t)),
where Z(p) = D(p) - E is the excess-demand vector and eta is the
step size.

Note on re-normalisation. In a pure Arrow-Debreu exchange economy
(wealth = p . endowment, price-dependent), demand is homogeneous of
degree zero in prices, so a re-normalisation step p / sum(p) is
harmless and is the standard convention. In the project's exogenous-
wealth setup, demand is not homogeneous of degree zero (there is an
absolute price level), so we run the iteration on raw prices and
normalise only for display, after convergence. The dynamic limit
(3.800, 3.150) coincides with the closed-form market price from
Task 1 and the Negishi-weighted planner duals from Task 3.

Outputs: a console summary and a three-panel figure saved alongside
this script as task4_convergence.png:
  1. Price trajectory p_k(t) with the closed-form equilibrium overlaid.
  2. Excess-demand norm |Z(p)| over iterations on a log scale.
  3. p_1 trajectories under several step sizes eta.
"""

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt


# ── Parameters: project's standard test economy ──────────────────────────────

alpha = np.array([
    [0.6, 0.4],   # consumer A
    [0.3, 0.7],   # consumer B
    [0.5, 0.5],   # consumer C
])
w = np.array([100.0, 80.0, 60.0])     # consumer wealths (dollars)
E = np.array([30.0,  40.0])           # total endowment per good

N, K = alpha.shape
S = (alpha * w[:, None]).sum(axis=0)  # S_k = sum_j alpha_jk * w_j


# ── Demand and excess demand ─────────────────────────────────────────────────

def aggregate_demand(p):
    """D_k(p) = sum_j alpha_jk * w_j / p_k = S_k / p_k."""
    return S / p


def excess_demand(p):
    """Z_k(p) = D_k(p) - E_k."""
    return aggregate_demand(p) - E


# ── Tatonnement ──────────────────────────────────────────────────────────────

def tatonnement(p_init, eta=0.01, max_steps=2000, tol=1e-10):
    """Discrete-time, proportional-to-excess-demand price adjustment.

    Returns the price trajectory (shape (T+1, K)) and the sequence of
    excess-demand norms (length T+1).
    """
    p = np.asarray(p_init, dtype=float).copy()
    trajectory = [p.copy()]
    z_norms = [float(np.linalg.norm(excess_demand(p)))]
    for _ in range(max_steps):
        p = p + eta * excess_demand(p)
        p = np.maximum(p, 1e-6)               # keep strictly positive
        trajectory.append(p.copy())
        z_norms.append(float(np.linalg.norm(excess_demand(p))))
        if z_norms[-1] < tol:
            break
    return np.array(trajectory), np.array(z_norms)


def normalise(p):
    """Project prices onto the simplex p_1 + p_2 + ... = 1."""
    return p / p.sum()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    print("=" * 64)
    print("  Cobb-Douglas Tatonnement  -  Task 4")
    print("=" * 64)

    p_analytical = S / E
    print(f"  Consumers : {N}        Goods : {K}")
    print(f"  Closed-form equilibrium")
    print(f"    raw         : {p_analytical}")
    print(f"    normalised  : {normalise(p_analytical)}")

    p_init = np.array([1.0, 1.0])
    eta = 0.01

    print(f"\n  Initial prices : {p_init}")
    print(f"  Step size eta  : {eta}")

    trajectory, z_norms = tatonnement(p_init, eta=eta, max_steps=2000)
    p_final = trajectory[-1]
    converged = np.allclose(p_final, p_analytical, atol=1e-4)

    print(f"\n  Iterations to convergence : {len(trajectory) - 1}")
    print(f"  Final prices (raw)        : {p_final}")
    print(f"  Final prices (normalised) : {normalise(p_final)}")
    print(f"  Final |Z|                 : {z_norms[-1]:.2e}")
    print(f"  Matches closed-form?      : {converged}")

    # ── Figure ──────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 3, figsize=(15, 4.5))
    t_axis = np.arange(len(trajectory))

    # Panel 1: price trajectory
    axes[0].plot(t_axis, trajectory[:, 0], label=r"$p_1$", color="C0")
    axes[0].plot(t_axis, trajectory[:, 1], label=r"$p_2$", color="C1")
    axes[0].axhline(p_analytical[0], color="C0", linestyle="--", alpha=0.5,
                    label=fr"$p_1^* = {p_analytical[0]:.3f}$")
    axes[0].axhline(p_analytical[1], color="C1", linestyle="--", alpha=0.5,
                    label=fr"$p_2^* = {p_analytical[1]:.3f}$")
    axes[0].set_xlabel("iteration")
    axes[0].set_ylabel("price")
    axes[0].set_title(rf"Price trajectory  ($\eta = {eta}$)")
    axes[0].legend(loc="lower right", fontsize=9)
    axes[0].grid(True, alpha=0.3)

    # Panel 2: convergence rate (|Z| log scale)
    axes[1].semilogy(t_axis, z_norms, color="C2")
    axes[1].set_xlabel("iteration")
    axes[1].set_ylabel(r"$\|Z(p)\|$")
    axes[1].set_title("Excess-demand norm (log scale)")
    axes[1].grid(True, alpha=0.3, which="both")

    # Panel 3: step-size comparison (p_1 trajectory under several eta)
    eta_grid = [0.005, 0.01, 0.02, 0.05, 0.10]
    for eta_alt, color in zip(eta_grid, ["C3", "C0", "C4", "C5", "C6"]):
        traj_alt, _ = tatonnement(p_init, eta=eta_alt, max_steps=2000)
        axes[2].plot(np.arange(len(traj_alt)), traj_alt[:, 0],
                     label=rf"$\eta = {eta_alt}$", color=color, alpha=0.85)
    axes[2].axhline(p_analytical[0], color="black", linestyle="--", alpha=0.5,
                    label=fr"$p_1^* = {p_analytical[0]:.3f}$")
    axes[2].set_xlabel("iteration")
    axes[2].set_ylabel(r"$p_1$")
    axes[2].set_title(r"Effect of step size $\eta$ on $p_1$")
    axes[2].legend(loc="lower right", fontsize=9)
    axes[2].grid(True, alpha=0.3)
    axes[2].set_xlim(0, 600)

    plt.tight_layout()
    fig_path = Path(__file__).parent / "task4_convergence.png"
    fig.savefig(fig_path, dpi=120, bbox_inches="tight")
    print(f"\n  Figure saved to: {fig_path.name}")

    plt.show()


if __name__ == "__main__":
    main()
