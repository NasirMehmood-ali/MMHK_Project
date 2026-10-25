"""2D 4-MMHK exact diagonalization: g(k) construction, ground state, double occupancy."""
import numpy as np
from itertools import combinations
from scipy.sparse import lil_matrix, csr_matrix
from scipy.linalg import eigh

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

def build_H_dense(g, U, n, basis, index):
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=complex)
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
    return H

def make_2x2_g(kx, ky, t=1.0, tprime=0.0):
    """g(k) for the 2D 4-MMHK model, Eq. (7)."""
    eps_tx = -2*t*np.cos(kx)
    eps_ty = -2*t*np.cos(ky)
    eps_tp = -4*tprime*np.cos(kx)*np.cos(ky)
    g = np.array([
        [0,      eps_tx, eps_ty, eps_tp],
        [eps_tx, 0,      eps_tp, eps_ty],
        [eps_ty, eps_tp, 0,      eps_tx],
        [eps_tp, eps_ty, eps_tx, 0     ]
    ], dtype=complex)
    return g

def count_double_occ(state, n):
    count = 0
    for a in range(n):
        occ_up = (state >> a) & 1
        occ_dn = (state >> (a + n)) & 1
        count += occ_up * occ_dn
    return count

def double_occupancy_from_state(psi, basis, n):
    D = 0.0
    for i, state in enumerate(basis):
        weight = abs(psi[i])**2
        d_i = count_double_occ(state, n)
        D += weight * d_i
    return D / n


def ground_state_D4(kx, ky, U, t=1.0, tprime=0.0, n=4, n_up=2, n_dn=2):
    """Returns (E0, D_per_site) for the n=4 cluster at given (kx,ky)."""
    basis_up, _ = build_basis(n, n_up)
    basis_dn, _ = build_basis(n, n_dn)
    basis = sorted(su | (sd << n) for su in basis_up for sd in basis_dn)
    index = {s: i for i, s in enumerate(basis)}
    g = make_2x2_g(kx, ky, t=t, tprime=tprime)
    H = build_H_dense(g, U, n, basis, index)
    assert np.allclose(H, H.conj().T, atol=1e-10), "H not Hermitian!"
    evals, evecs = eigh(H)
    E0 = evals[0]
    psi0 = evecs[:, 0]
    D = double_occupancy_from_state(psi0, basis, n)
    return E0, D

def D4_average(U, t=1.0, tprime=0.0, n_k=8):
    kx_vals = np.linspace(0, np.pi, n_k, endpoint=False) + np.pi/(2*n_k)
    ky_vals = np.linspace(0, np.pi, n_k, endpoint=False) + np.pi/(2*n_k)
    D_list = []
    for kx in kx_vals:
        for ky in ky_vals:
            _, D = ground_state_D4(kx, ky, U, t=t, tprime=tprime)
            D_list.append(D)
    return np.mean(D_list)

