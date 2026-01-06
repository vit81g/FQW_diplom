#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explain_anomalies.py

"Explainability" layer for SOC analysts:
- For each anomalous entity (user/host) on a target day, compute which 3–5 features deviated the most
  from that entity's historical baseline (median/MAD robust z-score).
- Add severity label (critical/high/medium/low) based on rank percentiles inside the day.

Inputs (in --work dir):
  - features_users_clean.csv
  - features_hosts_clean.csv
  - anomalies_users_YYYY-MM-DD.csv
  - anomalies_hosts_YYYY-MM-DD.csv

Outputs (in --work dir):
  - anomalies_users_YYYY-MM-DD_explain.csv
  - anomalies_hosts_YYYY-MM-DD_explain.csv
  - anomalies_all_YYYY-MM-DD_explain.csv  (combined users+hosts)

Usage:
  python explain_anomalies.py --work .\work --date 2025-12-31
  python explain_anomalies.py --work .\work                (date defaults to latest anomalies_* available)
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ID_COLS = ["entity", "date"]
DATE_RE = re.compile(r"anomalies_(users|hosts)_(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE)


def _read_csv(path: Path, dtype=str) -> pd.DataFrame:
    return pd.read_csv(path, dtype=dtype, keep_default_na=True)


def _pick_date_from_workdir(work_dir: Path) -> str:
    # pick the latest date among anomalies_users_*.csv and anomalies_hosts_*.csv
    dates: List[str] = []
    for p in work_dir.glob("anomalies_*_*.csv"):
        m = DATE_RE.match(p.name)
        if m:
            dates.append(m.group(2))
    if not dates:
        raise FileNotFoundError("No anomalies_*.csv found in work dir. Run train_anomaly_models.py first.")
    # sort by actual date
    dt = pd.to_datetime(pd.Series(dates), errors="coerce")
    return str(dt.max().date())


def _robust_z(x: float, med: float, mad: float, fallback_std: float) -> float:
    # MAD -> robust scale; 1.4826 makes MAD comparable to std for normal distribution
    if mad and mad > 0:
        return float((x - med) / (1.4826 * mad))
    if fallback_std and fallback_std > 0:
        return float((x - med) / fallback_std)
    # if no dispersion, only sign of deviation matters (or 0)
    if x == med:
        return 0.0
    return float(np.sign(x - med) * 999.0)


def _baseline_stats(train_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Returns (median_df, mad_df, std_df) indexed by entity.
    """
    g = train_df.groupby("entity", dropna=False)

    med = g[feature_cols].median(numeric_only=True)
    # MAD: median(|x - median|)
    mad = g[feature_cols].apply(lambda x: (x - x.median(numeric_only=True)).abs().median(numeric_only=True))
    std = g[feature_cols].std(numeric_only=True).fillna(0)

    return med, mad, std


def _severity_by_rank(df_day: pd.DataFrame) -> pd.Series:
    """
    Severity based on rank percentiles within the day:
      - critical: top 5%
      - high:     next 15% (<=20%)
      - medium:   next 30% (<=50%)
      - low:      rest
    Uses rank_combined where 1 is most anomalous.
    """
    if "rank_combined" not in df_day.columns:
        # fallback: compute rank from score_combined_norm if present
        if "score_combined_norm" in df_day.columns:
            df_day = df_day.copy()
            df_day["rank_combined"] = df_day["score_combined_norm"].rank(ascending=False, method="average")
        else:
            return pd.Series(["low"] * len(df_day), index=df_day.index)

    n = len(df_day)
    if n == 0:
        return pd.Series([], dtype=str)

    thr_crit = max(1, math.ceil(0.05 * n))
    thr_high = max(thr_crit, math.ceil(0.20 * n))
    thr_med = max(thr_high, math.ceil(0.50 * n))

    r = pd.to_numeric(df_day["rank_combined"], errors="coerce").fillna(n + 1)

    out = pd.Series(["low"] * n, index=df_day.index)
    out[r <= thr_med] = "medium"
    out[r <= thr_high] = "high"
    out[r <= thr_crit] = "critical"
    return out


def _coerce_features(df: pd.DataFrame, feature_cols: List[str]) -> pd.DataFrame:
    df = df.copy()
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0)
    return df


def _explain_one_kind(work_dir: Path,
                      kind: str,
                      target_date: str,
                      top_k: int) -> pd.DataFrame:
    kind = kind.lower()
    if kind not in {"users", "hosts"}:
        raise ValueError("kind must be 'users' or 'hosts'")

    feats_file = work_dir / f"features_{kind}_clean.csv"
    anom_file = work_dir / f"anomalies_{kind}_{target_date}.csv"

    if not feats_file.exists():
        raise FileNotFoundError(f"Missing features file: {feats_file}")
    if not anom_file.exists():
        raise FileNotFoundError(f"Missing anomalies file: {anom_file}")

    feats = _read_csv(feats_file, dtype=str)
    anom = _read_csv(anom_file, dtype=str)

    # normalize date in both
    feats["date"] = pd.to_datetime(feats["date"], errors="coerce").dt.date.astype("string")
    anom["date"] = pd.to_datetime(anom["date"], errors="coerce").dt.date.astype("string")

    # choose feature columns from features table
    feature_cols = [c for c in feats.columns if c not in ID_COLS]
    if not feature_cols:
        raise ValueError(f"No feature columns detected in {feats_file.name}")

    feats = _coerce_features(feats, feature_cols)

    # build baseline from all days except target day
    train = feats[feats["date"] != target_date].copy()
    if train.empty:
        train = feats.copy()

    med_df, mad_df, std_df = _baseline_stats(train, feature_cols)

    # select day slice and merge with anomaly list
    day = feats[feats["date"] == target_date].copy()
    if day.empty:
        raise ValueError(f"{kind}: no feature rows for date {target_date} in {feats_file.name}")

    merged = day.merge(anom, on=["entity", "date"], how="inner", suffixes=("", "_anom"))
    if merged.empty:
        # If anomalies file contains fewer entities than day slice, that's fine;
        # if join is empty, filenames/date likely mismatch.
        raise ValueError(f"{kind}: join between features and anomalies is empty for date {target_date}. "
                         f"Check that anomalies_{kind}_{target_date}.csv exists and matches 'entity' names.")

    # severity by rank within anomalies list
    sev = _severity_by_rank(merged)
    merged["severity"] = sev

    # compute top contributors per row
    contrib_rows: List[Dict[str, object]] = []

    for _, row in merged.iterrows():
        ent = row["entity"]
        # baseline vectors; if entity never seen in train, fall back to global baseline
        if ent in med_df.index:
            med = med_df.loc[ent]
            mad = mad_df.loc[ent]
            std = std_df.loc[ent]
        else:
            med = train[feature_cols].median(numeric_only=True)
            mad = (train[feature_cols] - med).abs().median(numeric_only=True)
            std = train[feature_cols].std(numeric_only=True).fillna(0)

        z_list = []
        for c in feature_cols:
            x = float(row[c]) if pd.notna(row[c]) else 0.0
            z = _robust_z(x, float(med.get(c, 0.0)), float(mad.get(c, 0.0)), float(std.get(c, 0.0)))
            z_list.append((c, z, x, float(med.get(c, 0.0))))

        # sort by absolute deviation
        z_list.sort(key=lambda t: abs(t[1]), reverse=True)
        top = z_list[:top_k]

        # compact string and structured columns
        out = {
            "entity_type": "user" if kind == "users" else "host",
            "entity": ent,
            "date": row["date"],
            "severity": row["severity"],
        }

        # propagate scores/ranks if present
        for col in ["score_isolation_forest", "score_lof", "score_combined_norm",
                    "rank_isolation_forest", "rank_lof", "rank_combined"]:
            if col in merged.columns:
                out[col] = row.get(col)

        # compact representation: feature:+z (value vs baseline)
        parts = []
        for c, z, x, b in top:
            sign = "+" if z >= 0 else ""
            parts.append(f"{c}:{sign}{z:.2f} (v={x:.0f}, base={b:.0f})")
        out["top_contributors"] = "; ".join(parts)

        # structured columns contrib1..contribK
        for i, (c, z, x, b) in enumerate(top, start=1):
            out[f"contrib{i}_feature"] = c
            out[f"contrib{i}_z"] = float(z)
            out[f"contrib{i}_value"] = float(x)
            out[f"contrib{i}_baseline_median"] = float(b)

        contrib_rows.append(out)

    report = pd.DataFrame(contrib_rows)

    # ensure stable column order
    preferred = [
        "entity_type", "entity", "date", "severity",
        "score_combined_norm", "rank_combined",
        "score_isolation_forest", "score_lof",
        "rank_isolation_forest", "rank_lof",
        "top_contributors",
    ]
    cols = []
    for c in preferred:
        if c in report.columns:
            cols.append(c)
    # then remaining columns
    for c in report.columns:
        if c not in cols:
            cols.append(c)
    report = report[cols].sort_values(["severity", "rank_combined"], ascending=[True, True], kind="mergesort")

    out_file = work_dir / f"anomalies_{kind}_{target_date}_explain.csv"
    report.to_csv(out_file, index=False)
    print(f"[+] Wrote: {out_file} (rows={len(report)})")
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, help="Work directory")
    p.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: latest anomalies_* date)")
    p.add_argument("--top-features", type=int, default=5, help="How many top deviating features to include (3-5 typical)")
    args = p.parse_args()

    work_dir = Path(args.work)
    if not work_dir.exists():
        raise FileNotFoundError(f"Work dir not found: {work_dir}")

    target_date = args.date or _pick_date_from_workdir(work_dir)
    target_date = str(pd.to_datetime(target_date, errors="raise").date())

    k = int(args.top_features)
    if k < 1 or k > 10:
        raise ValueError("--top-features must be between 1 and 10")

    users = _explain_one_kind(work_dir, "users", target_date, k)
    hosts = _explain_one_kind(work_dir, "hosts", target_date, k)

    all_df = pd.concat([users, hosts], ignore_index=True)
    out_all = work_dir / f"anomalies_all_{target_date}_explain.csv"
    all_df.to_csv(out_all, index=False)
    print(f"[+] Wrote: {out_all} (rows={len(all_df)})")
    print("[*] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
