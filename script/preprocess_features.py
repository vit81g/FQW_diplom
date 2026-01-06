#!/usr/bin/env python3
"""
preprocess_features.py

Purpose:
  Prepare feature tables for anomaly detection (Isolation Forest / LOF):
  - Reads features_users.csv and features_hosts.csv
  - Converts 'date' to ISO date (YYYY-MM-DD) where possible
  - Replaces NaN in feature columns with 0
  - Writes *_clean.csv outputs

Usage:
  python preprocess_features.py --work ./work
  python preprocess_features.py --users ./work/features_users.csv --hosts ./work/features_hosts.csv --out-dir ./work
"""

from __future__ import annotations

import argparse
from pathlib import Path
import pandas as pd


ID_COLS = ["entity", "date"]


def _load_csv(path: Path) -> pd.DataFrame:
    # dtype=str keeps identifiers stable; we'll convert feature columns to numeric where possible
    return pd.read_csv(path, dtype=str, keep_default_na=True)


def _coerce_date(df: pd.DataFrame) -> pd.DataFrame:
    if "date" in df.columns:
        # Parse robustly; keep only date part
        dt = pd.to_datetime(df["date"], errors="coerce", utc=False)
        # If already YYYY-MM-DD, fine; else normalize
        df["date"] = dt.dt.date.astype("string")
    return df


def _coerce_features_numeric(df: pd.DataFrame) -> pd.DataFrame:
    # Convert all non-ID columns to numeric (float), errors become NaN (then we'll fill with 0)
    for col in df.columns:
        if col in ID_COLS:
            continue
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fillna_zero(df: pd.DataFrame) -> pd.DataFrame:
    # Fill NaN in feature columns with 0; keep ID columns as-is
    feature_cols = [c for c in df.columns if c not in ID_COLS]
    df[feature_cols] = df[feature_cols].fillna(0)
    return df


def _validate(df: pd.DataFrame, name: str) -> None:
    missing = [c for c in ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")
    # entity/date should be non-null for most rows; warn if many missing
    null_entity = df["entity"].isna().sum()
    null_date = df["date"].isna().sum()
    if null_entity or null_date:
        print(f"[!] Warning: {name} has nulls: entity={null_entity}, date={null_date}")


def preprocess(in_path: Path, out_path: Path, name: str) -> None:
    df = _load_csv(in_path)
    _validate(df, name)
    df = _coerce_date(df)
    df = _coerce_features_numeric(df)
    df = _fillna_zero(df)

    # Optional: sort for determinism
    df = df.sort_values(["entity", "date"], kind="mergesort").reset_index(drop=True)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    print(f"[+] Wrote {name}: {out_path} (rows={len(df)}, cols={len(df.columns)})")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=str, default=None, help="Work directory containing features_*.csv")
    p.add_argument("--users", type=str, default=None, help="Path to features_users.csv")
    p.add_argument("--hosts", type=str, default=None, help="Path to features_hosts.csv")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default: work dir)")
    p.add_argument("--out-users", type=str, default="features_users_clean.csv", help="Output filename for users")
    p.add_argument("--out-hosts", type=str, default="features_hosts_clean.csv", help="Output filename for hosts")
    args = p.parse_args()

    if args.work:
        work = Path(args.work)
        users_in = Path(args.users) if args.users else work / "features_users.csv"
        hosts_in = Path(args.hosts) if args.hosts else work / "features_hosts.csv"
        out_dir = Path(args.out_dir) if args.out_dir else work
    else:
        if not args.users or not args.hosts:
            p.error("Either provide --work or both --users and --hosts")
        users_in = Path(args.users)
        hosts_in = Path(args.hosts)
        out_dir = Path(args.out_dir) if args.out_dir else users_in.parent

    users_out = out_dir / args.out_users
    hosts_out = out_dir / args.out_hosts

    if not users_in.exists():
        raise FileNotFoundError(f"Users features file not found: {users_in}")
    if not hosts_in.exists():
        raise FileNotFoundError(f"Hosts features file not found: {hosts_in}")

    preprocess(users_in, users_out, "users")
    preprocess(hosts_in, hosts_out, "hosts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
