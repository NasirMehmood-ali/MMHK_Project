"""Doped 4-MMHK DOS using Lorentzian broadening, with the corrected doping-
dependent mu(x) (fixes the smoothness vs. correctness tradeoff we hit with
histogram binning)."""
import numpy as np
from mmhk_doped_v2 import assign_fillings_and_mu
from mmhk_spectral_New import spectral_weights

def doped_dos_smooth(U, x, omega_grid, eta=0.1, n_k=25, t=1.0, tprime=0.0, n=4):
    fillings, mu = assign_fillings_and_mu(U, x, t=t, tprime=tprime, n_k=n_k, n=n)
    rho = np.zeros_like(omega_grid)
    for (kx, ky), (nu, nd) in fillings.items():
        for alpha in range(n):
            for spin in ('up', 'dn'):
                om_add, w_add, om_rem, w_rem = spectral_weights(
                    kx, ky, U, alpha, spin, n=n, t=t, tprime=tprime, n_up=nu, n_dn=nd)
                for om, w in zip(om_add - mu, w_add):
                    rho += w * (eta/np.pi) / ((omega_grid - om)**2 + eta**2)
                for om, w in zip(om_rem - mu, w_rem):
                    rho += w * (eta/np.pi) / ((omega_grid - om)**2 + eta**2)
    return rho / len(fillings) / 2, mu

if __name__ == "__main__":
    import time
    omega_grid = np.linspace(-2.2, 2.2, 90)
    t0 = time.time()
    rho, mu = doped_dos_smooth(10.0, x=0.01, omega_grid=omega_grid, n_k=15, tprime=-0.25)
    print(f"mu={mu:.3f}, max rho={rho.max():.3f}  ({time.time()-t0:.1f}s)")
