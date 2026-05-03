import numpy as np
import matplotlib.pyplot as plt

# Parameters
alpha = [0.5, 0.3, 0.7]
w = [10, 8, 6]
E1 = 10

# Precompute income term (cleaner & faster)
income = sum(a * wi for a, wi in zip(alpha, w))
# Excess demand function
def excess_demand(p):
    return income / p - E1

# Price range
p_vals = np.linspace(0.1, 5, 200)
z_vals = excess_demand(p_vals)

# Plot
plt.figure(figsize=(8, 5))

plt.axhline(0, color="black", linewidth=1)  # equilibrium line
plt.plot(p_vals, z_vals, label="Z(p₁)")

plt.xlabel("Price p₁")
plt.ylabel("Excess Demand Z(p₁)")
plt.title("Excess Demand Curve (Task 2)")

plt.grid(True)
plt.legend()

plt.show()