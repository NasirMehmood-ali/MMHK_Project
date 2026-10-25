"""2D DOS for 4-MMHK, averaged over the reduced BZ (reproducing Fig 3a / Fig 1a style)."""
import numpy as np
from mmhk_spectral_New import spectral_function

def dos_4mmhk(U, omega_grid, eta=0.2, n_k=6, t=1.0, tprime=0.0):
    kx_vals = np.linspace(0, np.pi, n_k, endpoint=False) + np.pi/(2*n_k)
    ky_vals = np.linspace(0, np.pi, n_k, endpoint=False) + np.pi/(2*n_k)
    rho = np.zeros_like(omega_grid)
    count = 0
    for kx in kx_vals:
        for ky in ky_vals:
            rho += spectral_function(kx, ky, U, omega_grid, eta=eta, t=t, tprime=tprime)
            count += 1
    return rho / count / 8  # /8 = n(4 orbitals) x 2 spins -- standard normalization, integral(DOS)=1
