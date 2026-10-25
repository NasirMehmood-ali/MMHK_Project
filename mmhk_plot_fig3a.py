"""
Reproduces Fig 3a: DOS of the half-filled 4-MMHK model, t'=0.
Usage: python3 mmhk_plot_fig3a.py
Requires mmhk_dos.py, mmhk_spectral.py, mmhk_2d_core.py in the same folder.
"""
import numpy as np
import matplotlib.pyplot as plt
from mmhk_dos import dos_4mmhk

def main():
    omega_grid = np.linspace(-10, 10, 300)
    U_values = [0, 2, 4, 6, 8, 10]

    fig, ax = plt.subplots(figsize=(6, 4.5))
    for U in U_values:
        rho = dos_4mmhk(float(U), omega_grid, eta=0.1, n_k=25)
        ax.plot(omega_grid, rho, label=f"U={U}")
        print(f"U={U} done")

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho(\omega)$")
    ax.set_title("4-MMHK model DOS (half-filled, t'=0)")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig3a_reproduction.png", dpi=150)
    print("Saved fig3a_reproduction.png")

if __name__ == "__main__":
    main()
