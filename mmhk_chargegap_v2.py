"""
Grand-canonical charge gap for 4-MMHK, correct for t' != 0 (broken particle-hole symmetry).

Method: at each k, compute the ground energy E_k(N) for every possible total
particle number N=0..8 (minimizing over all (n_up,n_dn) splits with n_up+n_dn=N).
The many-body ground state of the WHOLE lattice at fixed total Ntot is obtained
by a "water-filling" argument: pool the marginal costs {E_k(N+1)-E_k(N)} from
every k together, sort them, and fill the cheapest Ntot of them. The charge gap
at half filling is the gap between the Nhalf-th and (Nhalf+1)-th cheapest
marginal costs in that pooled, sorted list.
"""
import numpy as np
from mmhk_spectral import diagonalize_sector
from mmhk_2d_core import make_2x2_g

def E_of_N_at_k(kx, ky, U, t=1.0, tprime=0.0, n=4):
    """Returns array E[N] for N=0..2n, ground energy at fixed total particle number N,
    minimized over all (n_up,n_dn) splits."""
    g = make_2x2_g(kx, ky, t=t, tprime=tprime)
    best = {}
    for nu in range(n + 1):
        for nd in range(n + 1):
            _, _, evals, _ = diagonalize_sector(g, U, n, nu, nd)
            N = nu + nd
            e0 = evals[0]
            if N not in best or e0 < best[N]:
                best[N] = e0
    return np.array([best[N] for N in range(2 * n + 1)])

def marginal_costs_at_k(kx, ky, U, t=1.0, tprime=0.0, n=4):
    E = E_of_N_at_k(kx, ky, U, t=t, tprime=tprime, n=n)
    return np.diff(E)  # length 2n, cost of the N-th electron for N=1..2n

def charge_gap_grand_canonical(U, t=1.0, tprime=0.0, n_k=6, n=4):
    kx_vals = np.linspace(0, np.pi, n_k, endpoint=False) + np.pi/(2*n_k)
    ky_vals = np.linspace(0, np.pi, n_k, endpoint=False) + np.pi/(2*n_k)
    all_costs = []
    for kx in kx_vals:
        for ky in ky_vals:
            all_costs.extend(marginal_costs_at_k(kx, ky, U, t=t, tprime=tprime, n=n))
    all_costs = np.sort(np.array(all_costs))
    n_k_count = n_k * n_k
    N_half = (n // 2) * 2 * n_k_count  # half filling: n electrons per cluster, summed over k
    # N_half-th cheapest cost (1-indexed) = index N_half-1; next one = index N_half
    mu_minus = all_costs[N_half - 1]
    mu_plus = all_costs[N_half]
    return mu_plus - mu_minus

if __name__ == "__main__":
    print("--- validation: compare to earlier (corrected) fixed-filling result at t'=0 ---")
    for U in [0.0, 1.0, 2.0, 4.0]:
        D = charge_gap_grand_canonical(U, tprime=0.0, n_k=4)
        print(f"U={U}: Delta(t'=0, grand-canonical) = {D:.4f}")
