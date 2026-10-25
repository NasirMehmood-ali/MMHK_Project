"""
Reproduces Fig 5b with smooth curves: DOS vs omega, t'=0, U/t=10.
Usage: python3 mmhk_plot_fig5b_smooth.py
Requires mmhk_doped_smooth.py, mmhk_doped_v2.py, mmhk_spectral.py, mmhk_2d_core.py.
"""
import numpy as np
import matplotlib.pyplot as plt
from mmhk_doped_smooth import doped_dos_smooth

def main():
    omega_grid = np.linspace(-2.2, 2.2, 120)
    x_values = [0.01, 0.03, 0.05, 0.07, 0.1]

    fig, ax = plt.subplots(figsize=(5.5, 4.5))
    for x in x_values:
        rho, mu = doped_dos_smooth(10.0, x=x, omega_grid=omega_grid, eta=0.13, n_k=30, tprime=0.0)
        ax.plot(omega_grid, rho, label=f"x={x}")
        print(f"x={x}: mu(x)={mu:.3f}, max rho={rho.max():.3f}")

    ax.set_xlabel(r"$\omega$")
    ax.set_ylabel(r"$\rho(\omega)$")
    ax.set_title("DOS, t'=0, U/t=10")
    ax.legend()
    fig.tight_layout()
    fig.savefig("fig5b_smooth.png", dpi=150)
    print("Saved fig5b_smooth.png")

if __name__ == "__main__":
    main()
