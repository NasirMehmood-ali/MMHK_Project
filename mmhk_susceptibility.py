"""AF spin susceptibility S(q=(pi,pi), omega=0) for 4-MMHK (reproducing Fig 3b)."""
import numpy as np
from mmhk_heatcap import build_H_full_fock
from mmhk_2d_core import make_2x2_g
from scipy.linalg import eigh

def staggered_Sz_diag(n=4):
    """Diagonal values of S_stag for every Fock state in the full 2^(2n) space."""
    # site positions matching make_2x2_g's convention: 0=(0,0) 1=(1,0) 2=(0,1) 3=(1,1)
    signs = np.array([+1, -1, -1, +1])  # (-1)^(rx+ry) for Q=(pi,pi)
    n_modes = 2 * n
    dim = 2 ** n_modes
    D = np.zeros(dim)
    for state in range(dim):
        s = 0.0
        for a in range(n):
            occ_up = (state >> a) & 1
            occ_dn = (state >> (a + n)) & 1
            s += signs[a] * 0.5 * (occ_up - occ_dn)
        D[state] = s
    return D

def chi_static(kx, ky, U, beta, t=1.0, tprime=0.0, n=4):
    mu = U / 2.0
    g = make_2x2_g(kx, ky, t=t, tprime=tprime)
    H = build_H_full_fock(g, U, mu, n)
    evals, evecs = eigh(H)
    evals = evals - evals.min()  # numerical stability
    D = staggered_Sz_diag(n)
    # matrix elements <m|D|n> in the eigenbasis: (evecs^dagger) @ diag(D) @ evecs
    Dmat = evecs.conj().T @ (D[:, None] * evecs)
    weights = np.exp(-beta * evals)
    Z = np.sum(weights)
    Em = evals[:, None]
    En = evals[None, :]
    dE = Em - En
    with np.errstate(divide='ignore', invalid='ignore'):
        kernel = np.where(np.abs(dE) > 1e-10,
                           (np.exp(-beta*En) - np.exp(-beta*Em)) / np.where(dE == 0, 1, dE),
                           beta * np.exp(-beta*En))
    chi = np.sum(np.abs(Dmat)**2 * kernel) / Z
    return chi.real

if __name__ == "__main__":
    print("--- sanity checks: chi should be real, positive, finite ---")
    for beta in [1, 5, 10, 20]:
        chi = chi_static(0.3, 0.5, U=4.0, beta=beta)
        print(f"beta={beta:5.1f}  chi = {chi:.5f}")

def chi_averaged(U, beta, n_k=6, t=1.0, tprime=0.0, n=4):
    kx_vals = np.linspace(0, np.pi, n_k, endpoint=False) + np.pi/(2*n_k)
    ky_vals = np.linspace(0, np.pi, n_k, endpoint=False) + np.pi/(2*n_k)
    chis = [chi_static(kx, ky, U, beta, t=t, tprime=tprime)
            for kx in kx_vals for ky in ky_vals]
    return np.mean(chis) / n   # per-site normalization
