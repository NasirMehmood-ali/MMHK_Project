"""
Doped 4-MMHK DOS, correctly referenced to the doping-dependent chemical
potential mu(x) (not the fixed half-filling U/2), using histogram binning.
"""
import numpy as np
from mmhk_spectral import diagonalize_sector, spectral_weights
from mmhk_2d_core import make_2x2_g

def best_split_per_N(kx, ky, U, t=1.0, tprime=0.0, n=4):
    g = make_2x2_g(kx, ky, t=t, tprime=tprime)
    best_E, best_split = {}, {}
    for nu in range(n + 1):
        for nd in range(n + 1):
            _, _, evals, _ = diagonalize_sector(g, U, n, nu, nd)
            N = nu + nd
            if N not in best_E or evals[0] < best_E[N]:
                best_E[N] = evals[0]
                best_split[N] = (nu, nd)
    E = np.array([best_E[N] for N in range(2 * n + 1)])
    split = [best_split[N] for N in range(2 * n + 1)]
    return E, split

def assign_fillings_and_mu(U, x, t=1.0, tprime=0.0, n_k=8, n=4):
    """Water-filling: returns (fillings dict, mu(x)) -- mu is the marginal-cost
    threshold separating filled from unfilled levels, i.e. the true doping-
    dependent chemical potential."""
    kx_vals = np.linspace(0, np.pi, n_k, endpoint=False) + np.pi/(2*n_k)
    ky_vals = np.linspace(0, np.pi, n_k, endpoint=False) + np.pi/(2*n_k)

    per_k_data = {}
    all_costs = []
    for kx in kx_vals:
        for ky in ky_vals:
            E, split = best_split_per_N(kx, ky, U, t=t, tprime=tprime, n=n)
            per_k_data[(kx, ky)] = (E, split)
            for N_from, c in enumerate(np.diff(E)):
                all_costs.append((c, kx, ky, N_from))

    n_k_count = n_k * n_k
    N_target_total = round((1 - x) * n * n_k_count)
    all_costs.sort(key=lambda tup: tup[0])
    selected = all_costs[:N_target_total]

    fill_count = {key: 0 for key in per_k_data}
    for (c, kx, ky, N_from) in selected:
        fill_count[(kx, ky)] += 1

    fillings = {key: per_k_data[key][1][N] for key, N in fill_count.items()}

    # mu(x) = midpoint between the last selected cost and first unselected cost
    if N_target_total < len(all_costs):
        mu = (all_costs[N_target_total - 1][0] + all_costs[N_target_total][0]) / 2
    else:
        mu = all_costs[-1][0]
    return fillings, mu

def doped_dos_histogram(U, x, omega_edges, n_k=8, t=1.0, tprime=0.0, n=4):
    fillings, mu = assign_fillings_and_mu(U, x, t=t, tprime=tprime, n_k=n_k, n=n)
    all_omegas, all_weights = [], []
    for (kx, ky), (nu, nd) in fillings.items():
        for alpha in range(n):
            for spin in ('up', 'dn'):
                om_add, w_add, om_rem, w_rem = spectral_weights(
                    kx, ky, U, alpha, spin, n=n, t=t, tprime=tprime, n_up=nu, n_dn=nd)
                all_omegas.extend(om_add - mu)
                all_weights.extend(w_add)
                all_omegas.extend(om_rem - mu)
                all_weights.extend(w_rem)
    all_omegas = np.array(all_omegas)
    all_weights = np.array(all_weights)
    hist, _ = np.histogram(all_omegas, bins=omega_edges, weights=all_weights)
    bin_width = omega_edges[1] - omega_edges[0]
    rho = hist / bin_width / len(fillings) / 2
    return rho, mu

if __name__ == "__main__":
    omega_edges = np.linspace(-2.2, 2.2, 90)
    rho, mu = doped_dos_histogram(10.0, x=0.01, omega_edges=omega_edges, n_k=10, tprime=-0.25)
    print(f"mu(x=0.01) = {mu:.3f}  (compare to fixed U/2 = 5.0)")
    print(f"max rho within [-2.2,2.2] = {rho.max():.4f}")
