"""
Reproduces Extended Data Fig 6a: charge gap Delta vs U for t'=0 and t'=-0.25.
Uses the grand-canonical (water-filling) method, correct for both t' values.
Usage: python3 mmhk_plot_ed6a_v2.py
Requires mmhk_chargegap_v2.py, mmhk_spectral.py, mmhk_2d_core.py in the same folder.
"""
import numpy as np
import matplotlib.pyplot as plt
from mmhk_chargegap_v2 import charge_gap_grand_canonical

def main():
    U_values = [0, 1, 2, 3, 4, 5, 6, 7, 8]
    fig, ax = plt.subplots(figsize=(5, 4))
    for tprime, label in [(0.0, "t'=0"), (-0.25, "t'=-0.25")]:
        gaps = [charge_gap_grand_canonical(float(U), tprime=tprime, n_k=14) for U in U_values]
        ax.plot(U_values, gaps, marker='o', label=label)
        print(f"{label}:", np.round(gaps, 3))

    ax.set_xlabel("U")
    ax.set_ylabel(r"$\Delta$")
    ax.set_title("Charge gap, 4-MMHK")
    ax.legend()
    fig.tight_layout()
    fig.savefig("ed6a_reproduction.png", dpi=150)
    print("Saved ed6a_reproduction.png")

if __name__ == "__main__":
    main()
