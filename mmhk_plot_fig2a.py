"""
Plots Fig 2a: |E^n_MMHK - E^inf_Bethe| / |E^inf_Bethe| [%] vs U, for several n.

Usage:
    python3 mmhk_plot_fig2a.py

Requires mmhk_core.py in the same folder, plus numpy/scipy/matplotlib.
Edit n_values / U_values / n_k_map below to trade off speed vs coverage.
"""
import matplotlib.pyplot as plt
from mmhk_core import MMHKSolver, bethe_energy_per_site

def main():
    solver = MMHKSolver()
    n_values = [4, 6, 8, 10]
    n_k_map = {4: 20, 6: 16, 8: 12, 10: 8, 12: 6}
    U_values = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]

    data = {n: [] for n in n_values}
    for U in U_values:
        Ebethe = bethe_energy_per_site(U)
        for n in n_values:
            Emmhk = solver.mmhk_energy_per_site(n, U, n_k=n_k_map[n])
            dev = abs(Emmhk - Ebethe) / abs(Ebethe) * 100
            data[n].append(dev)
        print(f"U={U} done")

    fig, ax = plt.subplots(figsize=(5, 4))
    for n in n_values:
        ax.plot(U_values, data[n], marker='o', label=f"n={n}")
    ax.set_xlabel("U")
    ax.set_ylabel(r"$|E^n_{\rm MMHK}-E^\infty_{\rm HB}|/|E^\infty_{\rm HB}|$  [%]")
    ax.legend()
    ax.set_title("1D n-MMHK vs Bethe ansatz")
    fig.tight_layout()
    fig.savefig("fig2a_reproduction.png", dpi=150)
    print("Saved fig2a_reproduction.png")

if __name__ == "__main__":
    main()
