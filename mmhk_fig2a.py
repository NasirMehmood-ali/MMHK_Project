"""
Reproduces Fig 2a of Mai et al., arXiv:2401.08746 (1D n-MMHK vs Bethe ansatz).

Usage:
    python3 mmhk_fig2a.py

Requires: mmhk_core.py in the same folder, plus numpy/scipy.
Runtime: n=4,6,8 finish in seconds each. n=10 takes ~30-60s per U value
(reduce n_k below if you want it faster, at the cost of a coarser
twist-angle average). n=12 is NOT attempted here by default -- it costs
roughly 60-80s per single k-point, so a full U sweep takes hours. See
the commented-out block at the bottom if you want to run it anyway
(e.g. overnight).
"""
import sys
import time
from mmhk_core import MMHKSolver, bethe_energy_per_site

def main():
    solver = MMHKSolver()
    n_values = [4, 6, 8, 10]     # add 12 here if you have time to spare
    n_k_map = {4: 20, 6: 16, 8: 12, 10: 8, 12: 6}   # coarser grid for larger n
    U_values = [1, 2, 3, 4, 5, 6, 7, 8]

    print(f"{'U':>4}" + "".join(f"{'n='+str(n):>10}" for n in n_values))
    for U in U_values:
        t0 = time.time()
        Ebethe = bethe_energy_per_site(U)
        row = [f"{U:>4}"]
        for n in n_values:
            Emmhk = solver.mmhk_energy_per_site(n, U, n_k=n_k_map[n])
            dev = abs(Emmhk - Ebethe) / abs(Ebethe) * 100
            row.append(f"{dev:>10.3f}")
        print(" ".join(row) + f"   ({time.time()-t0:.1f}s)")
        sys.stdout.flush()

if __name__ == "__main__":
    main()

# --- optional: n=12, run overnight if you want the full paper range ---
# solver = MMHKSolver()
# for U in [1,2,3,4,5,6,7,8]:
#     Emmhk = solver.mmhk_energy_per_site(12, U, n_k=6)
#     Ebethe = bethe_energy_per_site(U)
#     print(U, abs(Emmhk-Ebethe)/abs(Ebethe)*100)
