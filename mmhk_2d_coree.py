"""
mmhk_2d_core.py

Builds the single-particle dispersion/hybridization matrix g(k) for the
n=4 MMHK model (a 2x2 real-space cluster per grouped reduced-BZ momentum),
following Eq. (7) of Mai et al., "Twisting the Hubbard model into the
Momentum-Mixing Hatsugai-Kohmoto Model" (arXiv:2401.08746).

Physical picture (from our derivation-by-hand of the n=2 case, generalized):
  - We group n=4 original momenta into one "bundle" per reduced-BZ point (kx, ky).
  - Locally Fourier-transforming within that bundle turns the 4 independent
    band energies into a 4x4 MATRIX (off-diagonal terms = new hopping/hybridization).
  - This matrix is identical in form to a 2x2 real-space Hubbard cluster with
    "twisted" boundary conditions set by (kx, ky) -- kx, ky here play the role
    of the twist angles theta_x, theta_y.

Eq. 7:
    g(k) =
      [ 0          eps_tx      eps_ty      eps_t'  ]
      [ eps_tx     0           eps_t'      eps_ty  ]
      [ eps_ty     eps_t'      0           eps_tx  ]
      [ eps_t'     eps_ty      eps_tx      0       ]

    eps_tx  = -2 t  cos(kx)
    eps_ty  = -2 t  cos(ky)
    eps_t'  = -4 t' cos(kx) cos(ky)

The reduced Brillouin zone (rBZ4) for this n=4 grouping has HALF the linear
extent of the original BZ in each direction, i.e. kx, ky in [-pi/2, pi/2).
(4 reduced-BZ points x 4 eigenvalue branches = 4 x N_reduced = N_original,
consistent with n=4 grouping four original momenta into each bundle.)
"""
import numpy as np


def make_2x2_g(kx, ky, t=1.0, tprime=0.0):
    """
    Build the 4x4 real, symmetric hybridization matrix g(k) for the n=4
    (2x2 cluster) MMHK model, Eq. 7.

    Parameters
    ----------
    kx, ky : float
        Reduced-BZ momentum components. Physically meaningful range is
        [-pi/2, pi/2) for each (see reduced_bz_grid below).
    t : float
        Nearest-neighbor hopping.
    tprime : float
        Next-nearest-neighbor hopping.

    Returns
    -------
    g : (4,4) ndarray, real symmetric
    """
    eps_tx = -2.0 * t * np.cos(kx)
    eps_ty = -2.0 * t * np.cos(ky)
    eps_tp = -4.0 * tprime * np.cos(kx) * np.cos(ky)

    g = np.array([
        [0.0,     eps_tx,  eps_ty,  eps_tp],
        [eps_tx,  0.0,     eps_tp,  eps_ty],
        [eps_ty,  eps_tp,  0.0,     eps_tx],
        [eps_tp,  eps_ty,  eps_tx,  0.0   ],
    ])
    return g


def reduced_bz_grid(n_k):
    """
    Return an (n_k x n_k) grid of (kx, ky) points tiling the reduced BZ
    for n=4 grouping: kx, ky each range over [-pi/2, pi/2).
    """
    ks = np.linspace(-np.pi/2, np.pi/2, n_k, endpoint=False)
    return ks


if __name__ == "__main__":
    # --- Sanity check #1: g(k) is real & symmetric (a valid single-particle Hamiltonian) ---
    g = make_2x2_g(0.3, 0.7, t=1.0, tprime=0.2)
    assert np.allclose(g, g.T), "g(k) should be symmetric!"
    print("g(kx=0.3, ky=0.7, t=1, t'=0.2):")
    print(np.round(g, 4))

    # --- Sanity check #2: at t'=0, unfolding the 4 eigenvalue branches over the
    #     reduced BZ [-pi/2,pi/2)^2 should EXACTLY reproduce the ordinary 2D
    #     tight-binding band xi(Kx,Ky) = -2t(cos Kx + cos Ky) over the FULL BZ
    #     [-pi,pi)^2. This is the free-electron (U=0) cross-check.
    n_k = 40
    ks = reduced_bz_grid(n_k)
    eigs_from_cluster = []
    for kx in ks:
        for ky in ks:
            g = make_2x2_g(kx, ky, t=1.0, tprime=0.0)
            eigs_from_cluster.extend(np.linalg.eigvalsh(g))
    eigs_from_cluster = np.sort(np.array(eigs_from_cluster))

    # direct calculation over the FULL BZ with a matching number of points
    Ks = np.linspace(-np.pi, np.pi, 2*n_k, endpoint=False)
    Kx, Ky = np.meshgrid(Ks, Ks)
    eigs_direct = np.sort((-2.0 * (np.cos(Kx) + np.cos(Ky))).ravel())

    print(f"\nNumber of eigenvalues from 4-site cluster over reduced BZ: {len(eigs_from_cluster)}")
    print(f"Number of eigenvalues from direct band over full BZ:       {len(eigs_direct)}")
    print(f"Max abs difference between sorted eigenvalue lists: "
          f"{np.max(np.abs(eigs_from_cluster - eigs_direct)):.2e}")
    assert np.allclose(eigs_from_cluster, eigs_direct, atol=1e-8), \
        "Mismatch! The cluster construction does not reproduce the free-electron band."
    print("MATCH -- the 4x4 cluster construction over the reduced BZ exactly")
    print("reproduces the ordinary tight-binding band over the full BZ, as required at U=0.")
