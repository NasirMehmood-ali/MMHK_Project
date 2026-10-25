"""
Plots heat capacity vs T for several U (reproducing Fig 6a).
Usage: python3 mmhk_plot_fig6.py
Requires mmhk_heatcap.py and mmhk_2d_core.py in the same folder.
"""
import numpy as np
import matplotlib.pyplot as plt
from mmhk_heatcap import Cv_vs_T_averaged

def main():
    T_values = np.linspace(0.0005, 0.01, 25)
    U_values = [0, 2, 4, 6, 8, 10]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for U in U_values:
        Cv = Cv_vs_T_averaged(U, T_values, n_k=6)
        ax.plot(T_values, Cv, marker='.', label=f"U={U}")
        print(f"U={U} done")

    ax.set_xlabel("T/t")
    ax.set_ylabel(r"$C_v$")
    ax.set_title("Heat capacity, half-filled 4-MMHK")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig6b_reproduction.png", dpi=150)
    print("Saved fig6b_reproduction.png")

if __name__ == "__main__":
    main()
