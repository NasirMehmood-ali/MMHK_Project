import numpy as np
from itertools import combinations
from scipy.linalg import eigh

def build_basis(n_modes, n_particles):
    """All bitstrings (as ints) with n_particles bits set, out of n_modes bits."""
    basis = []
    for occ in combinations(range(n_modes), n_particles):
        state = 0
        for o in occ:
            state |= (1 << o)
        basis.append(state)
    basis.sort()
    index = {s: i for i, s in enumerate(basis)}
    return basis, index

def apply_cdag_c(state, p, q):
    """Apply c_p^dagger c_q to |state>. Returns (new_state, sign) or (None,0) if annihilated."""
    if not (state >> q) & 1:
        return None, 0
    # sign from removing at q: count occupied modes below q
    sign = (-1) ** bin(state & ((1 << q) - 1)).count('1')
    state2 = state & ~(1 << q)
    if p != q:
        if (state2 >> p) & 1:
            return None, 0
        sign *= (-1) ** bin(state2 & ((1 << p) - 1)).count('1')
        state2 |= (1 << p)
    return state2, sign

def build_Hk(g, U, n, basis, index):
    """
    g: n x n complex hopping matrix (single k).
    U: interaction strength.
    n: number of orbitals (sites) in the cluster.
    Mode convention: modes 0..n-1 = spin-up orbitals 1..n, modes n..2n-1 = spin-down orbitals 1..n.
    """
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=complex)
    for i, state in enumerate(basis):
        # hopping, spin up (modes 0..n-1) and spin down (modes n..2n-1)
        for spin_offset in (0, n):
            for a in range(n):
                for b in range(n):
                    amp = g[a, b]
                    if amp == 0:
                        continue
                    p, q = a + spin_offset, b + spin_offset
                    new_state, sign = apply_cdag_c(state, p, q)
                    if new_state is None:
                        continue
                    j = index[new_state]
                    H[j, i] += amp * sign
        # interaction U * n_{alpha,up} n_{alpha,down}
        diag = 0.0
        for a in range(n):
            occ_up = (state >> a) & 1
            occ_dn = (state >> (a + n)) & 1
            diag += U * occ_up * occ_dn
        H[i, i] += diag
    return H

def ground_state_energy(g, U, n, n_up, n_dn):
    n_modes = 2 * n
    # build basis for fixed (n_up, n_dn) sector by combining up-basis and down-basis
    basis_up, _ = build_basis(n, n_up)
    basis_dn, _ = build_basis(n, n_dn)
    basis = []
    for su in basis_up:
        for sd in basis_dn:
            basis.append(su | (sd << n))
    basis.sort()
    index = {s: i for i, s in enumerate(basis)}
    H = build_Hk(g, U, n, basis, index)
    assert np.allclose(H, H.conj().T, atol=1e-10), "H not Hermitian!"
    evals = eigh(H, eigvals_only=True)
    return evals[0].real

# ---- sanity test: n=4 1D MMHK ring, raw g(k), U=0, k=pi/8 ----
t = 1.0
k = np.pi/8
n = 4
g = np.zeros((n, n), dtype=complex)
g[0,1] = -t*np.exp(-1j*k); g[1,0] = np.conj(g[0,1])
g[1,2] = -t*np.exp(-1j*k); g[2,1] = np.conj(g[1,2])
g[2,3] = -t*np.exp(-1j*k); g[3,2] = np.conj(g[2,3])
g[3,0] = -t*np.exp(-1j*k); g[0,3] = np.conj(g[3,0])

# single-particle check first
sp_evals = np.linalg.eigvalsh(g)
analytic = np.array([-2*t*np.cos(k), -2*t*np.sin(k), 2*t*np.cos(k), 2*t*np.sin(k)])
print("single-particle eigenvalues (numeric):", np.sort(sp_evals))
print("single-particle eigenvalues (analytic):", np.sort(analytic))

# U=0 many-body ground state (2 up, 2 down) should equal 2*(sum of two lowest sp evals)
E0_U0 = ground_state_energy(g, 0.0, n, 2, 2)
expected_U0 = 2 * np.sum(np.sort(sp_evals)[:2])
print(f"\nED ground energy (U=0): {E0_U0:.6f}")
print(f"Expected (fill lowest 2 orbitals x2 spins): {expected_U0:.6f}")

# ---- Lieb-Wu Bethe ansatz ground state energy per site, half-filling, 1D Hubbard ----
from scipy.integrate import quad
from scipy.special import j0, j1

def bethe_energy_per_site(U, t=1.0):
    # E/N = -4t * integral_0^inf  J0(w)J1(w) / [w (1+exp(w U/(2t)))] dw   (standard Lieb-Wu form)
    def integrand(w):
        return j0(w) * j1(w) / (w * (1 + np.exp(w * U / (2*t))))
    val, _ = quad(integrand, 0, np.inf, limit=200)
    return -4 * t * val

# ---- n-MMHK ground energy per site, averaged over reduced BZ (twist angle grid) ----
def mmhk_energy_per_site(n, U, t=1.0, n_k=60):
    ks = np.linspace(0, 2*np.pi/n, n_k, endpoint=False) + (np.pi/n)/n_k  # avoid exact k=0 edge issues
    n_up = n // 2
    n_dn = n - n_up
    energies = []
    for k in ks:
        g = np.zeros((n, n), dtype=complex)
        for a in range(n-1):
            g[a, a+1] = -t*np.exp(-1j*k); g[a+1, a] = np.conj(g[a, a+1])
        g[n-1, 0] = -t*np.exp(1j*k); g[0, n-1] = np.conj(g[n-1, 0])
        E = ground_state_energy(g, U, n, n_up, n_dn)
        energies.append(E)
    # E per cluster averaged over k, then per site = /n
    return np.mean(energies) / n

print("\n--- Fig 2a reproduction: |E_MMHK - E_Bethe| / |E_Bethe| [%] ---")
print(f"{'U':>4} {'n=4':>10} {'n=6':>10}")
for U in [1, 2, 3, 4, 5, 6, 7, 8]:
    Ebethe = bethe_energy_per_site(U)
    row = [f"{U:>4}"]
    for n in (4, 6):
        Emmhk = mmhk_energy_per_site(n, U, n_k=40)
        dev = abs(Emmhk - Ebethe) / abs(Ebethe) * 100
        row.append(f"{dev:>10.3f}")
    print(" ".join(row))
