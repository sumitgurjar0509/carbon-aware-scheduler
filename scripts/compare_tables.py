"""
Compare regenerated result tables with the committed ones.

    python scripts/compare_tables.py
    python scripts/compare_tables.py --committed DIR --new DIR --rtol 1e-6 --atol 1e-8

For every CSV: same columns, same number of rows, text identical, and
every number equal within |new - committed| <= atol + rtol * |committed|.
Rows are compared in order (the pipeline is deterministic, so a changed
order is itself a difference).

Status per file:
    MATCH        equal within tolerance
    DIFFERENT    something differs (details are printed)
    MISSING      committed table that was not regenerated
    NO_BASELINE  new table with nothing committed to compare against

Exit code: 0 = every committed table matched, 1 = DIFFERENT or MISSING,
2 = nothing to compare (so an empty folder can never "pass").
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DEFAULT_COMMITTED = PROJECT_ROOT / "results" / "tables_committed"
DEFAULT_NEW = PROJECT_ROOT / "results" / "tables"
RTOL = 1e-6
ATOL = 1e-8


def _is_number_column(series: pd.Series) -> bool:
    # True/False columns count as text: they must match exactly.
    return pd.api.types.is_numeric_dtype(series) and not pd.api.types.is_bool_dtype(series)


def compare_frames(committed: pd.DataFrame, new: pd.DataFrame,
                   rtol: float = RTOL, atol: float = ATOL) -> dict:
    """
    Returns {"problems": [text, ...], "max_abs_diff": float}.
    An empty problem list means the two tables match.
    """
    problems = []
    max_abs_diff = 0.0

    if list(committed.columns) != list(new.columns):
        problems.append(f"columns differ: committed={list(committed.columns)} "
                        f"new={list(new.columns)}")
        return {"problems": problems, "max_abs_diff": max_abs_diff}
    if len(committed) != len(new):
        problems.append(f"row count differs: committed={len(committed)} new={len(new)}")
        return {"problems": problems, "max_abs_diff": max_abs_diff}

    for col in committed.columns:
        a, b = committed[col], new[col]
        if _is_number_column(a) and _is_number_column(b):
            x = a.to_numpy(dtype=float)
            y = b.to_numpy(dtype=float)
            # equal_nan: an empty cell in both tables counts as equal.
            bad = ~np.isclose(y, x, rtol=rtol, atol=atol, equal_nan=True)
            both_finite = np.isfinite(x) & np.isfinite(y)
            if both_finite.any():
                max_abs_diff = max(max_abs_diff,
                                   float(np.abs(y[both_finite] - x[both_finite]).max()))
        else:
            x = a.astype(object).where(a.notna(), "<empty>").astype(str).to_numpy()
            y = b.astype(object).where(b.notna(), "<empty>").astype(str).to_numpy()
            bad = x != y

        if bad.any():
            i = int(np.flatnonzero(bad)[0])
            problems.append(f"column '{col}': {int(bad.sum())} of {len(a)} rows differ "
                            f"(first at row {i}: committed={a.iloc[i]!r}, new={b.iloc[i]!r})")

    return {"problems": problems, "max_abs_diff": max_abs_diff}


def compare_dirs(committed_dir: Path, new_dir: Path,
                 rtol: float = RTOL, atol: float = ATOL) -> list[dict]:
    """One result dict per CSV name found in either folder."""
    committed = {p.name: p for p in committed_dir.glob("*.csv")}
    new = {p.name: p for p in new_dir.glob("*.csv")}

    results = []
    for name in sorted(set(committed) | set(new)):
        result = {"file": name, "problems": [], "max_abs_diff": None}
        if name not in new:
            result["status"] = "MISSING"
        elif name not in committed:
            result["status"] = "NO_BASELINE"
        else:
            result.update(compare_frames(pd.read_csv(committed[name]),
                                         pd.read_csv(new[name]), rtol, atol))
            result["status"] = "DIFFERENT" if result["problems"] else "MATCH"
        results.append(result)
    return results


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--committed", type=Path, default=DEFAULT_COMMITTED)
    parser.add_argument("--new", type=Path, default=DEFAULT_NEW)
    parser.add_argument("--rtol", type=float, default=RTOL)
    parser.add_argument("--atol", type=float, default=ATOL)
    args = parser.parse_args(argv)

    if not any(args.committed.glob("*.csv")):
        print(f"No CSV files in {args.committed} -- nothing to compare against.")
        return 2

    results = compare_dirs(args.committed, args.new, args.rtol, args.atol)
    print(f"committed: {args.committed}\nnew:       {args.new}\n"
          f"tolerance: rtol={args.rtol:g}, atol={args.atol:g}\n")
    for r in results:
        diff = "" if r["max_abs_diff"] is None else f"  max |diff| = {r['max_abs_diff']:.3g}"
        print(f"{r['status']:<12} {r['file']}{diff}")
        for problem in r["problems"]:
            print(f"             - {problem}")

    statuses = [r["status"] for r in results]
    print("\n" + ", ".join(f"{s}: {statuses.count(s)}"
                           for s in ("MATCH", "DIFFERENT", "MISSING", "NO_BASELINE")))
    return 1 if {"DIFFERENT", "MISSING"} & set(statuses) else 0


if __name__ == "__main__":
    sys.exit(main())
