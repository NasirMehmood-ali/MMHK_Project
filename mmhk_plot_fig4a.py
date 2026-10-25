"""
Plots D4 vs U (reproducing part of Fig 4a from arXiv:2401.08746).
Usage: python3 mmhk_plot_fig4a.py
Requires mmhk_2d_core.py in the same folder.
"""
import matplotlib.pyplot as plt
from mmhk_2d_core import D4_average

def main():
    U_values = [0, 1, 2, 3, 4, 5, 6, 8, 10, 14, 18]
    D_values = [D4_average(U, n_k=8) for U in U_values]

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(U_values, D_values, marker='o', color='purple', label="n=4")
    ax.axhline(0.25, color='gray', linestyle='--', linewidth=0.8, label="non-int. limit (1/4)")
    ax.set_xlabel("U/t")
    ax.set_ylabel(r"$D_n$")
    ax.set_title("Double occupancy, half-filling 2D_4-MMHK ")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig4a_reproduction.png", dpi=150)
    print("Saved fig4a_reproduction.png")

if __name__ == "__main__":
    main()
