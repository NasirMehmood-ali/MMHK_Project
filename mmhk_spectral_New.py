"""Single-particle spectral function via Lehmann representation, for n-MMHK clusters."""
import numpy as np
from itertools import combinations
from scipy.linalg import eigh
from mmhk_2d_core import make_2x2_g

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

def apply_single(state, mode, dagger):
    """Apply c_mode (dagger=False) or c^dagger_mode (dagger=True) to |state>."""
    occ = (state >> mode) & 1
    if dagger and occ:
        return None, 0          # can't create where already occupied
    if (not dagger) and (not occ):
        return None, 0          # can't destroy where empty
    sign = (-1) ** bin(state & ((1 << mode) - 1)).count('1')
    new_state = state ^ (1 << mode)   # flip that bit
    return new_state, sign

def build_H_dense(g, U, n, basis, index):
    dim = len(basis)
    H = np.zeros((dim, dim), dtype=complex)
    nz_hop = [(a, b, g[a, b]) for a in range(n) for b in range(n) if g[a, b] != 0]
    for i, state in enumerate(basis):
        for spin_offset in (0, n):
            for a, b, amp in nz_hop:
                p, q = a + spin_offset, b + spin_offset
                # c^dagger_p c_q as two single-operator applications
                s1, sign1 = apply_single(state, q, dagger=False)
                if s1 is None:
                    continue
                s2, sign2 = apply_single(s1, p, dagger=True)
                if s2 is None:
                    continue
                j = index[s2]
                H[j, i] += amp * sign1 * sign2
        diag = 0.0
        for a in range(n):
            occ_up = (state >> a) & 1
            occ_dn = (state >> (a + n)) & 1
            diag += U * occ_up * occ_dn
        H[i, i] += diag
    return H

def diagonalize_sector(g, U, n, n_up, n_dn):
    basis_up, _ = build_basis(n, n_up)
    basis_dn, _ = build_basis(n, n_dn)
    basis = sorted(su | (sd << n) for su in basis_up for sd in basis_dn)
    index = {s: i for i, s in enumerate(basis)}
    H = build_H_dense(g, U, n, basis, index)
    assert np.allclose(H, H.conj().T, atol=1e-10), "H not Hermitian!"
    evals, evecs = eigh(H)
    return basis, index, evals, evecs

def matrix_elements(mode, dagger, basis0, psi0, basis1, index1, evecs1):
    """<m| c^(dagger)_mode |0> for every eigenstate m of sector 1."""
    vec = np.zeros(len(basis1), dtype=complex)
    for i, state in enumerate(basis0):
        new_state, sign = apply_single(state, mode, dagger)
        if new_state is None:
            continue
        j = index1[new_state]
        vec[j] += sign * psi0[i]
    overlaps = evecs1.conj().T @ vec   # <m|vec> for every m
    return overlaps

def spectral_weights(kx, ky, U, alpha, spin, n=4, t=1.0, tprime=0.0, n_up=2, n_dn=2):
    """
    Returns (omega_add, weight_add, omega_remove, weight_remove):
    electron addition and removal energies/weights for orbital alpha, given spin ('up' or 'dn').
    """
    g = make_2x2_g(kx, ky, t=t, tprime=tprime)
    mode = alpha if spin == 'up' else alpha + n

    basis0, index0, evals0, evecs0 = diagonalize_sector(g, U, n, n_up, n_dn)
    E0 = evals0[0]
    psi0 = evecs0[:, 0]

        # addition: go to sector with one more electron in this spin channel
    if spin == 'up':
        n_up1, n_dn1 = n_up + 1, n_dn
    else:
        n_up1, n_dn1 = n_up, n_dn + 1
    if n_up1 > n or n_dn1 > n:
        omega_add, weight_add = np.array([]), np.array([])
    else:
        basis1, index1, evals1, evecs1 = diagonalize_sector(g, U, n, n_up1, n_dn1)
        overlaps_add = matrix_elements(mode, True, basis0, psi0, basis1, index1, evecs1)
        omega_add = evals1 - E0
        weight_add = np.abs(overlaps_add) ** 2

    # removal: go to sector with one fewer electron in this spin channel
    if spin == 'up':
        n_up2, n_dn2 = n_up - 1, n_dn
    else:
        n_up2, n_dn2 = n_up, n_dn - 1
    if n_up2 < 0 or n_dn2 < 0:
        omega_rem, weight_rem = np.array([]), np.array([])
    else:
        basis2, index2, evals2, evecs2 = diagonalize_sector(g, U, n, n_up2, n_dn2)
        overlaps_rem = matrix_elements(mode, False, basis0, psi0, basis2, index2, evecs2)
        omega_rem = E0 - evals2
        weight_rem = np.abs(overlaps_rem) ** 2

    return omega_add, weight_add, omega_rem, weight_rem

    # addition: go to sector with one more electron in this spin channel
    #if spin == 'up':
      #  n_up1, n_dn1 = n_up + 1, n_dn
    #else:
     #   n_up1, n_dn1 = n_up, n_dn + 1
    #basis1, index1, evals1, evecs1 = diagonalize_sector(g, U, n, n_up1, n_dn1)
    #overlaps_add = matrix_elements(mode, True, basis0, psi0, basis1, index1, evecs1)
   # omega_add = evals1 - E0
    #weight_add = np.abs(overlaps_add) ** 2

    # removal: go to sector with one fewer electron in this spin channel
    #if spin == 'up':
   #     n_up2, n_dn2 = n_up - 1, n_dn
    #else:
     #   n_up2, n_dn2 = n_up, n_dn - 1
   # basis2, index2, evals2, evecs2 = diagonalize_sector(g, U, n, n_up2, n_dn2)
    #overlaps_rem = matrix_elements(mode, False, basis0, psi0, basis2, index2, evecs2)
    #omega_rem = E0 - evals2
    #weight_rem = np.abs(overlaps_rem) ** 2

   # return omega_add, weight_add, omega_rem, weight_rem

# ---- test: sum rule check ----
if __name__ == "__main__":
    kx, ky = 0.3, 0.5
    U = 4.0
    om_add, w_add, om_rem, w_rem = spectral_weights(kx, ky, U, alpha=0, spin='up')
    total = np.sum(w_add) + np.sum(w_rem)
    print(f"Sum of addition weights: {np.sum(w_add):.6f}")
    print(f"Sum of removal weights:  {np.sum(w_rem):.6f}")
    print(f"Total (should be 1.0):   {total:.6f}")

def spectral_function(kx, ky, U, omega_grid, eta=0.2, mu=None, n=4, t=1.0, tprime=0.0, n_up=2, n_dn=2):
    """
    A(k,omega) on a given omega grid, summed over orbitals alpha=0..n-1 and both spins.
    mu: chemical potential shift (defaults to U/2, the half-filling particle-hole
        symmetric point) so the DOS is centered at omega=0 like the paper's plots.
    """
    if mu is None:
        mu = U / 2.0
    A = np.zeros_like(omega_grid)
    for alpha in range(n):
        for spin in ('up', 'dn'):
            om_add, w_add, om_rem, w_rem = spectral_weights(
                kx, ky, U, alpha, spin, n=n, t=t, tprime=tprime, n_up=n_up, n_dn=n_dn)
            om_add_shifted = om_add - mu
            om_rem_shifted = om_rem - mu
            for om, w in zip(om_add_shifted, w_add):
                A += w * (eta / np.pi) / ((omega_grid - om) ** 2 + eta ** 2)
            for om, w in zip(om_rem_shifted, w_rem):
                A += w * (eta / np.pi) / ((omega_grid - om) ** 2 + eta ** 2)
    return A

if __name__ == "__main__":
    print("\n--- spectral function symmetry check at half filling ---")
    omega_grid = np.linspace(-6, 6, 61)
    A = spectral_function(0.3, 0.5, U=4.0, omega_grid=omega_grid)
    A_flipped = A[::-1]
    print("max |A(omega) - A(-omega)| (should be small if particle-hole symmetric):",
          np.max(np.abs(A - A_flipped)))
    print("integral of A (rough check, should relate to total spectral weight):",
          np.trapezoid(A, omega_grid))
