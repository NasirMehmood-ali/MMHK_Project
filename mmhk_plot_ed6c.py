"""
Reproduces Extended Data Fig 6c: critical interaction Uc vs |t'|.

Usage: python3 mmhk_plot_ed6c.py
Requires mmhk_ed6c_core.py, mmhk_chargegap_v2.py, mmhk_spectral.py,
mmhk_2d_core.py all in the same folder.

Note on runtime: n_k=16 with ~22 U-grid points per t' value takes a
few minutes per t'. Ten t' values means the full sweep can take
10+ minutes total. If you want to check progress as it runs, this
script prints each t' value's Uc as soon as it's found.
"""
import numpy as np
import matplotlib.pyplot as plt
from mmhk_ed6c_core import find_Uc

def main():
    tprime_values = [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
    U_grid = np.arange(0.0, 4.4, 0.2)

    Uc_values = []
    for tp in tprime_values:
        Uc, gaps = find_Uc(tp, U_grid, n_k=16)
        Uc_values.append(Uc)
        print(f"t'={tp}: Uc={Uc}")

    fig, ax = plt.subplots(figsize=(5, 4))
    ax.plot(tprime_values, Uc_values, marker='o', color='green')
    ax.set_xlabel(r"$|t'|$")
    ax.set_ylabel(r"$U_c$")
    ax.set_title("Critical U vs |t'|")
    fig.tight_layout()
    fig.savefig("ed6c_reproduction.png", dpi=150)
    print("Saved ed6c_reproduction.png")

if __name__ == "__main__":
    main()
