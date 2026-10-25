"""Heat capacity for 2D 4-MMHK at half-filling (reproducing Fig. 6)."""
import numpy as np
from scipy.linalg import eigh
from mmhk_2d_core import make_2x2_g, apply_cdag_c

def build_H_full_fock(g, U, mu, n):
    """Full Fock space (all 2^(2n) states, no fixed particle-number restriction)."""
    n_modes = 2 * n
    dim = 2 ** n_modes
    basis = list(range(dim))  # every possible bit pattern is a valid state
    H = np.zeros((dim, dim), dtype=complex)
    nz_hop = [(a, b, g[a, b]) for a in range(n) for b in range(n) if g[a, b] != 0]
    for state in basis:
        for spin_offset in (0, n):
            for a, b, amp in nz_hop:
                p, q = a + spin_offset, b + spin_offset
                new_state, sign = apply_cdag_c(state, p, q)
                if new_state is None:
                    continue
                H[new_state, state] += amp * sign
        # interaction + chemical potential (diagonal)
        diag = 0.0
        n_total = bin(state).count('1')
        for a in range(n):
            occ_up = (state >> a) & 1
            occ_dn = (state >> (a + n)) & 1
            diag += U * occ_up * occ_dn
        diag -= mu * n_total
        H[state, state] += diag
    return H

def thermo_at_k(kx, ky, U, T_values, t=1.0, tprime=0.0, n=4):
    mu = U / 2.0  # half-filling, particle-hole symmetric point
    g = make_2x2_g(kx, ky, t=t, tprime=tprime)
    H = build_H_full_fock(g, U, mu, n)
    assert np.allclose(H, H.conj().T, atol=1e-8), "H not Hermitian!"
    evals = eigh(H, eigvals_only=True).real
    results = []
    for T in T_values:
        # subtract min energy for numerical stability (doesn't change physics)
        e = evals - evals.min()
        w = np.exp(-e / T)
        Z = np.sum(w)
        E_avg = np.sum(evals * w) / Z
        E2_avg = np.sum(evals**2 * w) / Z
        Cv = (E2_avg - E_avg**2) / T**2 / n   # per site -- Cv is extensive, divide by cluster size
        results.append((Z, E_avg, Cv))
    return evals, results

# --- quick single-k test with sanity checks ---
T_values = np.array([0.01, 0.5, 1.0, 2.0, 100.0])
evals, results = thermo_at_k(0.3, 0.5, U=4.0, T_values=T_values)

print(f"Number of eigenvalues: {len(evals)}  (expect 256)")
print(f"Ground state energy: {evals.min():.5f}")
print()
print(f"{'T':>8} {'Z':>12} {'<E>':>10} {'Cv':>10}")
for T, (Z, E, Cv) in zip(T_values, results):
    print(f"{T:>8} {Z:>12.4f} {E:>10.4f} {Cv:>10.5f}")
print()
print("Check: Z at T=100 should be close to 256 (all states ~equally likely)")
print("Check: Cv at T=0.01 should be close to 0 (only ground state contributes)")

def Cv_vs_T_averaged(U, T_values, n_k=4, t=1.0, tprime=0.0):
    """Average Cv(T) over a coarse (kx,ky) grid -- full Fock space diag is expensive
    (256x256 per k-point), so keep n_k small for speed."""
    kx_vals = np.linspace(0, np.pi, n_k, endpoint=False) + np.pi/(2*n_k)
    ky_vals = np.linspace(0, np.pi, n_k, endpoint=False) + np.pi/(2*n_k)
    Cv_sum = np.zeros(len(T_values))
    count = 0
    for kx in kx_vals:
        for ky in ky_vals:
            _, results = thermo_at_k(kx, ky, U, T_values, t=t, tprime=tprime)
            Cv_sum += np.array([r[2] for r in results])
            count += 1
    return Cv_sum / count
