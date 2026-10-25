"""
Computes the Fig 2a data and writes it to a CSV file for plotting elsewhere
(Mathematica, Origin, Excel, etc.) instead of using matplotlib.

Usage:
    python3 mmhk_export_csv.py

Output: fig2a_data.csv, with columns: U, n4, n6, n8, n10, ...
(one column per n value in n_values below)
"""
import csv
from mmhk_core import MMHKSolver, bethe_energy_per_site

def main():
    solver = MMHKSolver()
    n_values = [4, 6, 8, 10]     # add/remove n as needed
    n_k_map = {4: 20, 6: 16, 8: 12, 10: 8, 12: 6}
    U_values = [0.5, 1, 2, 3, 4, 5, 6, 7, 8]

    rows = []
    for U in U_values:
        Ebethe = bethe_energy_per_site(U)
        row = {"U": U}
        for n in n_values:
            Emmhk = solver.mmhk_energy_per_site(n, U, n_k=n_k_map[n])
            dev = abs(Emmhk - Ebethe) / abs(Ebethe) * 100
            row[f"n{n}"] = dev
        rows.append(row)
        print(f"U={U} done")

    fieldnames = ["U"] + [f"n{n}" for n in n_values]
    with open("fig2a_data.csv", "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print("Saved fig2a_data.csv")

if __name__ == "__main__":
    main()
