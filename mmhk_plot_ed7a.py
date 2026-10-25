"""
Reproduces Extended Data Fig 7a: equal-time spin correlation S(q) vs beta,
for different momentum transfers q, at U=10, 4-MMHK.
Usage: python3 mmhk_plot_ed7a.py
Requires mmhk_spincorr.py, mmhk_heatcap.py, mmhk_2d_core.py in the same folder.
"""
import numpy as np
import matplotlib.pyplot as plt
from mmhk_spincorr import S_q_averaged

def main():
    beta_values = [0.5, 1, 2, 4, 6, 8, 10, 14, 18, 22]
    q_list = [((0.0, 0.0), "q=(0,0)"),
              ((0.0, np.pi), "q=(0,pi)"),
              ((np.pi, np.pi), "q=(pi,pi)")]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for q, label in q_list:
        S_vals = [S_q_averaged(10.0, q, beta, n_k=6) for beta in beta_values]
        ax.plot(beta_values, S_vals, marker='o', label=label)
        print(f"{label}:", np.round(S_vals, 3))

    ax.set_xlabel(r"$\beta t$")
    ax.set_ylabel(r"$S(q)$")
    ax.set_title("Equal-time spin correlation, 4-MMHK, U=10")
    ax.legend()
    fig.tight_layout()
    fig.savefig("ed7a_reproduction.png", dpi=150)
    print("Saved ed7a_reproduction.png")

if __name__ == "__main__":
    main()
