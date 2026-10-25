"""Core MMHK exact-diagonalization engine (1D case). No module-level execution."""
import numpy as np
from itertools import combinations
from scipy.sparse import lil_matrix, csr_matrix
from scipy.sparse.linalg import eigsh
from scipy.linalg import eigh
from scipy.integrate import quad
from scipy.special import j0, j1

def build_basis(n_modes, n_particles):
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
    if not (state >> q) & 1:
        return None, 0
    sign = (-1) ** bin(state & ((1 << q) - 1)).count('1')
    state2 = state & ~(1 << q)
    if p != q:
        if (state2 >> p) & 1:
            return None, 0
        sign *= (-1) ** bin(state2 & ((1 << p) - 1)).count('1')
        state2 |= (1 << p)
    return state2, sign

def build_Hk_sparse(g, U, n, basis, index):
    dim = len(basis)
    H = lil_matrix((dim, dim), dtype=complex)
    nz_hop = [(a, b, g[a, b]) for a in range(n) for b in range(n) if g[a, b] != 0]
    for i, state in enumerate(basis):
        for spin_offset in (0, n):
            for a, b, amp in nz_hop:
                p, q = a + spin_offset, b + spin_offset
                new_state, sign = apply_cdag_c(state, p, q)
                if new_state is None:
                    continue
                j = index[new_state]
                H[j, i] += amp * sign
        diag = 0.0
        for a in range(n):
            occ_up = (state >> a) & 1
            occ_dn = (state >> (a + n)) & 1
            diag += U * occ_up * occ_dn
        H[i, i] += diag
    return csr_matrix(H)

def make_ring_g(n, k, t=1.0):
    """g(k) for the 1D n-MMHK ring (raw form, built directly from the definition)."""
    g = np.zeros((n, n), dtype=complex)
    for a in range(n - 1):
        g[a, a+1] = -t * np.exp(-1j*k)
        g[a+1, a] = np.conj(g[a, a+1])
    g[n-1, 0] = -t * np.exp(1j*k)
    g[0, n-1] = np.conj(g[n-1, 0])
    return g

def bethe_energy_per_site(U, t=1.0):
    def integrand(w):
        return j0(w) * j1(w) / (w * (1 + np.exp(min(w * U / (2*t), 700))))
    val, _ = quad(integrand, 0, np.inf, limit=200)
    return -4 * t * val

class MMHKSolver:
    """Caches the Fock basis per (n, n_up, n_dn) so repeated U/k sweeps are fast."""
    def __init__(self):
        self._cache = {}

    def _get_basis(self, n, n_up, n_dn):
        key = (n, n_up, n_dn)
        if key not in self._cache:
            basis_up, _ = build_basis(n, n_up)
            basis_dn, _ = build_basis(n, n_dn)
            basis = sorted(su | (sd << n) for su in basis_up for sd in basis_dn)
            index = {s: i for i, s in enumerate(basis)}
            self._cache[key] = (basis, index)
        return self._cache[key]

    def ground_state_energy(self, g, U, n, n_up, n_dn):
        basis, index = self._get_basis(n, n_up, n_dn)
        H = build_Hk_sparse(g, U, n, basis, index)
        if H.shape[0] < 50:
            return eigh(H.toarray(), eigvals_only=True)[0].real
        return eigsh(H, k=1, which='SA', return_eigenvectors=False, maxiter=3000)[0].real

    def mmhk_energy_per_site(self, n, U, t=1.0, n_k=10):
        ks = np.linspace(0, 2*np.pi/n, n_k, endpoint=False) + (np.pi/n)/n_k
        n_up, n_dn = n // 2, n - n // 2
        energies = [self.ground_state_energy(make_ring_g(n, k, t), U, n, n_up, n_dn) for k in ks]
        return np.mean(energies) / n
