"""
Reproduces Fig 3b: AF spin susceptibility S(q=(pi,pi),omega=0) vs beta, for various U.
Usage: python3 mmhk_plot_fig3b.py
Requires mmhk_susceptibility.py, mmhk_heatcap.py, mmhk_2d_core.py in the same folder.
"""
import numpy as np
import matplotlib.pyplot as plt
from mmhk_susceptibility import chi_averaged

def main():
    beta_values = [1, 2, 4, 6, 8, 10, 14, 18, 22]
    U_values = [0, 2, 4, 6, 8, 10]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for U in U_values:
        chi = [chi_averaged(float(U), beta, n_k=8) for beta in beta_values]
        ax.plot(beta_values, chi, marker='o', label=f"U={U}")
        print(f"U={U} done")

    ax.set_xlabel(r"$\beta t$")
    ax.set_ylabel(r"$S(q=(\pi,\pi),\omega=0)$")
    ax.set_title("AF spin susceptibility, half-filled 4-MMHK")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig3b_reproduction.png", dpi=150)
    print("Saved fig3b_reproduction.png")

if __name__ == "__main__":
    main()
