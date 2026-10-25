"""Core Uc-extraction logic for Extended Data Fig 6c, shared across batches."""
import numpy as np
from mmhk_chargegap_v2 import charge_gap_grand_canonical

GAP_THRESHOLD = 0.2  # raised above the ~0.05-0.1 resolution-floor noise we found

def find_Uc(tprime, U_grid, n_k=16):
    gaps = np.array([charge_gap_grand_canonical(float(U), tprime=tprime, n_k=n_k)
                      for U in U_grid])
    above = gaps > GAP_THRESHOLD
    # require 2 consecutive points above threshold to reject single-point noise
    i = None
    for j in range(len(above) - 1):
        if above[j] and above[j+1]:
            i = j
            break
    if i is None or i == 0:
        return None, gaps
    U0, U1 = U_grid[i-1], U_grid[i]
    g0, g1 = gaps[i-1], gaps[i]
    Uc = U0 + (GAP_THRESHOLD - g0) * (U1 - U0) / (g1 - g0)
    return Uc, gaps
