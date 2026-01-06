#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_features.py

Reads daily normalized CSV files from a work/ directory (produced by the normalization script)
and builds two feature tables:
  - features_users.csv  (entity=userXXX per day)
  - features_hosts.csv  (entity=hostname per day)

Designed as an offline (e.g., daily) preprocessing stage to feed anomaly detection models
(Isolation Forest / LOF / One-Class SVM).

Input filename conventions supported (case-insensitive):
  user files:
    user###_SIEM_YYYY-MM-DD.csv
    user###_PAN_YYYY-MM-DD.csv
  host files:
    <hostname>_SIEM_YYYY-MM-DD.csv
    <hostname>_PAN_YYYY-MM-DD.csv

If your filenames differ slightly, the script will still try to parse:
  - source = SIEM or PAN (must be present in name)
  - date   = last occurrence of YYYY-MM-DD in filename
  - entity = everything before _SIEM_ or _PAN_

Usage:
  python build_features.py --work ./work
Outputs:
  ./work/features_users.csv
  ./work/features_hosts.csv
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


INVALID_USER_TOKENS = {
    "", "-", "—", "_", "null", "none", "nan", "n/a", "na", "unknown",
    "local service", "network service", "система", "system",
}


DATE_RE = re.compile(r"(\\d{4}-\\d{2}-\\d{2})")
SOURCE_RE = re.compile(r"_(SIEM|PAN)_", re.IGNORECASE)
USER_ENTITY_RE = re.compile(r"^user\\d{3}$", re.IGNORECASE)


@dataclass(frozen=True)
class ParsedName:
    entity: str
    source: str   # "SIEM" or "PAN"
    date: str     # "YYYY-MM-DD"
    entity_type: str  # "user" or "host"


def parse_work_filename(path: Path) -> Optional[ParsedName]:
    name = path.name
    if not name.lower().endswith(".csv"):
        return None
    if name.lower() in {"user_mapping.csv", "features_users.csv", "features_hosts.csv"}:
        return None

    src_m = SOURCE_RE.search(name)
    if not src_m:
        return None
    source = src_m.group(1).upper()

    dates = DATE_RE.findall(name)
    if not dates:
        return None
    date = dates[-1]  # last date in name

    # entity is everything before _SIEM_ / _PAN_
    parts = re.split(SOURCE_RE, name, maxsplit=1, flags=re.IGNORECASE)
    # re.split returns: [prefix, group1, suffix]
    if not parts or len(parts) < 3:
        return None
    entity_raw = parts[0].rstrip("_").strip()
    if not entity_raw:
        return None

    entity = entity_raw
    entity_type = "user" if USER_ENTITY_RE.match(entity.lower()) else "host"
    return ParsedName(entity=entity, source=source, date=date, entity_type=entity_type)


def _safe_nunique(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    s = df[col].astype(str).str.strip()
    s = s[~s.str.lower().isin(INVALID_USER_TOKENS)]
    s = s[s != ""]
    return int(s.nunique(dropna=True))


def _safe_count(df: pd.DataFrame, col: str) -> int:
    if col not in df.columns:
        return 0
    s = df[col].astype(str).str.strip()
    s = s[~s.str.lower().isin(INVALID_USER_TOKENS)]
    s = s[s != ""]
    return int(len(s))


def _time_features(df: pd.DataFrame, prefix: str) -> Dict[str, float]:
    """
    Derives simple time-of-day statistics from Timestamp column if present.
    Assumes Timestamp already in ISO-like / human-readable format.
    """
    out: Dict[str, float] = {
        f"{prefix}_unique_hours": 0.0,
        f"{prefix}_night_share": 0.0,     # 00:00-05:59
        f"{prefix}_business_share": 0.0,  # 09:00-17:59
    }
    if "Timestamp" not in df.columns:
        return out

    ts = pd.to_datetime(df["Timestamp"], errors="coerce", utc=False)
    ts = ts.dropna()
    if ts.empty:
        return out

    hours = ts.dt.hour
    total = float(len(hours))
    out[f"{prefix}_unique_hours"] = float(hours.nunique())
    out[f"{prefix}_night_share"] = float(((hours >= 0) & (hours < 6)).sum()) / total
    out[f"{prefix}_business_share"] = float(((hours >= 9) & (hours < 18)).sum()) / total
    return out


def _collect_user_features(df: pd.DataFrame, source: str) -> Dict[str, float]:
    """
    Features for user-day.
    """
    p = source.lower()
    feats: Dict[str, float] = {}
    feats[f"{p}_events_total"] = float(len(df))
    feats[f"{p}_unique_destination_addr"] = float(_safe_nunique(df, "DestinationAddress"))
    feats[f"{p}_unique_destination_user"] = float(_safe_nunique(df, "DestinationUserName"))
    feats[f"{p}_unique_source_process"] = float(_safe_nunique(df, "SourceProcessName"))
    feats[f"{p}_unique_event_class"] = float(_safe_nunique(df, "DeviceEventClassID"))
    feats[f"{p}_unique_category"] = float(_safe_nunique(df, "DeviceEventCategory"))
    feats[f"{p}_unique_name"] = float(_safe_nunique(df, "Name"))
    feats.update(_time_features(df, p))
    return feats


def _collect_host_features(df: pd.DataFrame, source: str) -> Dict[str, float]:
    """
    Features for host-day.
    Hostnames are not anonymized; user fields are assumed anonymized already (userXXX),
    but we still compute counts over them.
    """
    p = source.lower()
    feats: Dict[str, float] = {}
    feats[f"{p}_events_total"] = float(len(df))
    feats[f"{p}_unique_destination_addr"] = float(_safe_nunique(df, "DestinationAddress"))
    feats[f"{p}_unique_source_process"] = float(_safe_nunique(df, "SourceProcessName"))
    feats[f"{p}_unique_event_class"] = float(_safe_nunique(df, "DeviceEventClassID"))
    feats[f"{p}_unique_category"] = float(_safe_nunique(df, "DeviceEventCategory"))
    feats[f"{p}_unique_name"] = float(_safe_nunique(df, "Name"))

    # unique users touching the host
    # Combine SourceUserName and DestinationUserName, ignoring machine accounts like HOSTNAME$
    users = []
    for col in ("SourceUserName", "DestinationUserName"):
        if col in df.columns:
            s = df[col].astype(str).str.strip()
            s = s[~s.str.lower().isin(INVALID_USER_TOKENS)]
            s = s[s != ""]
            s = s[~s.str.endswith("$")]  # machine accounts
            users.append(s)
    if users:
        all_users = pd.concat(users, ignore_index=True)
        feats[f"{p}_unique_users"] = float(all_users.nunique(dropna=True))
    else:
        feats[f"{p}_unique_users"] = 0.0

    feats.update(_time_features(df, p))
    return feats


def _read_csv_safely(path: Path) -> pd.DataFrame:
    """
    Reads CSV produced by normalization step. Uses robust settings for quotes/newlines.
    """
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8",
        encoding_errors="replace",
        engine="python",
        quoting=csv.QUOTE_MINIMAL,  # output CSV should already be well-formed
    )


def build_features(work_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Returns (features_users_df, features_hosts_df)
    """
    records_users: List[Dict[str, float]] = []
    records_hosts: List[Dict[str, float]] = []

    # Accumulate by (entity, date)
    acc: Dict[Tuple[str, str, str], Dict[str, float]] = {}  # (entity_type, entity, date) -> feats

    for path in sorted(work_dir.glob("*.csv")):
        parsed = parse_work_filename(path)
        if not parsed:
            continue

        df = _read_csv_safely(path)

        key = (parsed.entity_type, parsed.entity, parsed.date)
        if key not in acc:
            acc[key] = {
                "entity": parsed.entity,
                "date": parsed.date,
            }

        if parsed.entity_type == "user":
            feats = _collect_user_features(df, parsed.source)
        else:
            feats = _collect_host_features(df, parsed.source)

        # merge (later files overwrite same keys; should not happen for same entity/date/source)
        acc[key].update(feats)

    # Post-process and split to users/hosts dataframes
    for (entity_type, entity, date), row in acc.items():
        # derived date fields
        dt = pd.to_datetime(date, errors="coerce")
        if pd.isna(dt):
            row["day_of_week"] = 0.0
            row["is_weekend"] = 0.0
        else:
            dow = int(dt.dayofweek)  # Mon=0
            row["day_of_week"] = float(dow)
            row["is_weekend"] = float(1 if dow >= 5 else 0)

        # derived ratios
        siem_total = float(row.get("siem_events_total", 0.0))
        pan_total = float(row.get("pan_events_total", 0.0))
        total = siem_total + pan_total
        row["pan_share_of_all_events"] = float(pan_total / total) if total > 0 else 0.0

        if entity_type == "user":
            records_users.append(row)
        else:
            records_hosts.append(row)

    users_df = pd.DataFrame(records_users).fillna(0.0)
    hosts_df = pd.DataFrame(records_hosts).fillna(0.0)

    # Stable column order: entity, date first
    def order_cols(df: pd.DataFrame) -> pd.DataFrame:
        if df.empty:
            return df
        cols = list(df.columns)
        fixed = ["entity", "date"]
        tail = [c for c in cols if c not in fixed]
        return df[fixed + sorted(tail)]

    users_df = order_cols(users_df).sort_values(["entity", "date"]).reset_index(drop=True)
    hosts_df = order_cols(hosts_df).sort_values(["entity", "date"]).reset_index(drop=True)

    return users_df, hosts_df


def main() -> int:
    ap = argparse.ArgumentParser(description="Build feature tables from normalized daily CSV files in work/ folder.")
    ap.add_argument("--work", default="./work", help="Path to work/ folder with daily CSV files.")
    ap.add_argument("--out-users", default="features_users.csv", help="Output filename for user features (written inside work/ by default).")
    ap.add_argument("--out-hosts", default="features_hosts.csv", help="Output filename for host features (written inside work/ by default).")
    args = ap.parse_args()

    work_dir = Path(args.work).resolve()
    if not work_dir.exists() or not work_dir.is_dir():
        print(f"[!] work directory not found: {work_dir}")
        return 2

    users_df, hosts_df = build_features(work_dir)

    out_users = (work_dir / args.out_users).resolve()
    out_hosts = (work_dir / args.out_hosts).resolve()

    users_df.to_csv(out_users, index=False, encoding="utf-8")
    hosts_df.to_csv(out_hosts, index=False, encoding="utf-8")

    print(f"[*] users features: {out_users}  rows={len(users_df)} cols={len(users_df.columns) if not users_df.empty else 0}")
    print(f"[*] hosts features: {out_hosts}  rows={len(hosts_df)} cols={len(hosts_df.columns) if not hosts_df.empty else 0}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
