"""
Side-by-side comparison of the two M2 runs (spec: "G9 corrected after the first test run").

    python scripts/compare_m2_tiebreak.py

v1 = results/tables/m2_base_lp_v1_cluster_first_tiebreak/summary.csv (cluster-first tie-break)
v2 = results/tables/m2_base_lp/summary.csv                          (earliest-start tie-break)
Writes results/tables/m2_base_lp/v1_vs_v2_tiebreak.csv.
"""
from pathlib import Path

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
V1 = PROJECT_ROOT / "results" / "tables" / "m2_base_lp_v1_cluster_first_tiebreak" / "summary.csv"
V2 = PROJECT_ROOT / "results" / "tables" / "m2_base_lp" / "summary.csv"
OUT = PROJECT_ROOT / "results" / "tables" / "m2_base_lp" / "v1_vs_v2_tiebreak.csv"
METRICS = {"pct_reduction_vs_random_fifo": "cut_vs_fifo_pct",
           "mean_start_delay_h": "mean_delay_h",
           "local_job_share": "local_share"}


def compare_versions(v1: pd.DataFrame, v2: pd.DataFrame) -> pd.DataFrame:
    """One row per policy: v1, v2 and v2 - v1 for each metric. Policies must match."""
    if list(v1["policy"]) != list(v2["policy"]):
        raise ValueError("v1 and v2 do not contain the same policies in the same order")
    out = v1[["policy", "alpha"]].copy()
    for col, name in METRICS.items():
        out[f"{name}_v1"] = v1[col].to_numpy()
        out[f"{name}_v2"] = v2[col].to_numpy()
        out[f"{name}_change"] = out[f"{name}_v2"] - out[f"{name}_v1"]
    return out


if __name__ == "__main__":
    table = compare_versions(pd.read_csv(V1), pd.read_csv(V2))
    table.to_csv(OUT, index=False)
    print(table.round(3).to_string(index=False))
    print(f"\nSaved {OUT}")
