"""Equal-time spin correlation S(q) vs beta for 4-MMHK (reproducing Ext. Data Fig 7a)."""
import numpy as np
from mmhk_heatcap import build_H_full_fock
from mmhk_2d_core import make_2x2_g
from scipy.linalg import eigh

# site positions matching make_2x2_g: 0=(0,0) 1=(1,0) 2=(0,1) 3=(1,1)
SITE_POS = [(0,0), (1,0), (0,1), (1,1)]

def Sz_q_diag(q, n=4):
    """Diagonal values of the q-resolved Sz operator, for every Fock state."""
    qx, qy = q
    signs = np.array([np.cos(qx*rx + qy*ry) for (rx, ry) in SITE_POS])  # +-1 for q on this lattice
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

def S_q_at_k(kx, ky, U, q, beta, t=1.0, tprime=0.0, n=4):
    mu = U / 2.0
    g = make_2x2_g(kx, ky, t=t, tprime=tprime)
    H = build_H_full_fock(g, U, mu, n)
    evals, evecs = eigh(H)
    evals = evals - evals.min()
    D = Sz_q_diag(q, n)
    D2 = D**2
    weights = np.exp(-beta * evals)
    Z = np.sum(weights)
    # <n|D^2|n> = sum_i |evecs[i,n]|^2 * D2[i]  (D is diagonal in Fock basis)
    probs = np.abs(evecs)**2          # shape (dim, n_eigenstates)
    exp_D2_per_state = D2 @ probs      # shape (n_eigenstates,)
    S = np.sum(weights * exp_D2_per_state) / Z
    return S

def S_q_averaged(U, q, beta, n_k=6, t=1.0, tprime=0.0, n=4):
    kx_vals = np.linspace(0, np.pi, n_k, endpoint=False) + np.pi/(2*n_k)
    ky_vals = np.linspace(0, np.pi, n_k, endpoint=False) + np.pi/(2*n_k)
    vals = [S_q_at_k(kx, ky, U, q, beta, t=t, tprime=tprime, n=n)
            for kx in kx_vals for ky in ky_vals]
    return np.mean(vals)

if __name__ == "__main__":
    print("--- sanity checks ---")
    for q, label in [((0,0), "q=(0,0)"), ((0,np.pi), "q=(0,pi)"), ((np.pi,np.pi), "q=(pi,pi)")]:
        S = S_q_averaged(4.0, q, beta=1.0, n_k=4)
        print(f"{label}: S = {S:.4f}")
    print()
    print("--- high-T limit check: all q should give the same S (no correlations) ---")
    for q, label in [((0,0), "q=(0,0)"), ((0,np.pi), "q=(0,pi)"), ((np.pi,np.pi), "q=(pi,pi)")]:
        S = S_q_averaged(4.0, q, beta=0.001, n_k=4)
        print(f"{label}: S(beta->0) = {S:.4f}")
