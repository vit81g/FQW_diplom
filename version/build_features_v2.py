#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
build_features.py (v2)
----------------------
Reads per-day CSV files from a work directory and produces:
- features_users.csv  (entity=userXXX, per day)
- features_hosts.csv  (entity=hostname, per day)

Key improvements vs v1:
- Robust filename parsing (supports *_SIEM_YYYY-MM-DD.csv, *_SIEM_data.csv, *_SIEM_date.csv, etc.)
- If date not in filename, infer date(s) from data (Date column or Timestamp)
- Handles empty/partial inputs gracefully (no KeyError)
- Emits diagnostics: how many files found/parsed/skipped and why
"""

from __future__ import annotations

import argparse
import re
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import pandas as pd


# Columns expected in normalized daily files (may be subset)
COL_TIMESTAMP = "Timestamp"
COL_DATE = "Date"
COL_NAME = "Name"
COL_DEVICE_PRODUCT = "DeviceProduct"
COL_DEVICE_VENDOR = "DeviceVendor"
COL_DEST_ADDR = "DestinationAddress"
COL_SRC_PROC = "SourceProcessName"
COL_EVENT_CLASS = "DeviceEventClassID"
COL_EVENT_CAT = "DeviceEventCategory"
COL_SRC_USER = "SourceUserName"
COL_DST_USER = "DestinationUserName"

EXCLUDE_FILES = {
    "user_mapping.csv",
    "features_users.csv",
    "features_hosts.csv",
}

DATE_RE = re.compile(r"\b(\d{4}-\d{2}-\d{2})\b")
# entity_source_anything.csv, where source is SIEM or PAN (case-insensitive)
SOURCE_RE = re.compile(r"_(SIEM|PAN)_", re.IGNORECASE)

# Tokens to ignore when counting users (host-level features)
INVALID_USER_TOKENS = {
    "", "-", "—", "–",
    "null", "none", "nan", "n/a", "na", "unknown", "undefined",
}
# Service accounts that often appear and are not real end-users (optional)
SERVICE_USERS = {
    "local service", "network service", "system", "система",
}


@dataclass
class FileMeta:
    path: Path
    entity: str
    source: str  # "SIEM" or "PAN"
    date_from_name: Optional[str]  # YYYY-MM-DD if present, else None


def _is_machine_account(u: str) -> bool:
    return u.endswith("$")


def _norm_user_token(u: str) -> Optional[str]:
    if u is None:
        return None
    s = str(u).strip()
    if not s:
        return None
    low = s.lower()
    if low in INVALID_USER_TOKENS:
        return None
    if low in SERVICE_USERS:
        return None
    if _is_machine_account(s):
        return None
    return s


def _parse_meta(p: Path) -> Optional[FileMeta]:
    name = p.name
    if name in EXCLUDE_FILES:
        return None
    if p.suffix.lower() != ".csv":
        return None

    m = SOURCE_RE.search(name)
    if not m:
        return None
    source = m.group(1).upper()

    # entity is everything before _SIEM_ / _PAN_
    entity = name[:m.start()].rstrip("_").strip()
    if not entity:
        return None

    date_m = DATE_RE.search(name)
    date_from_name = date_m.group(1) if date_m else None

    return FileMeta(path=p, entity=entity, source=source, date_from_name=date_from_name)


def _read_csv_safely(path: Path) -> pd.DataFrame:
    # Files were produced by our normalizer (CSV), but keep robust options.
    return pd.read_csv(
        path,
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8",
        encoding_errors="replace",
    )


def _ensure_datetime_cols(df: pd.DataFrame) -> pd.DataFrame:
    if COL_TIMESTAMP in df.columns:
        ts = pd.to_datetime(df[COL_TIMESTAMP], errors="coerce", utc=False)
        df["_ts"] = ts
        # If Date column absent, compute it
        if COL_DATE not in df.columns:
            df[COL_DATE] = ts.dt.strftime("%Y-%m-%d").fillna("")
    else:
        df["_ts"] = pd.NaT
        if COL_DATE not in df.columns:
            df[COL_DATE] = ""
    return df


def _split_by_date(df: pd.DataFrame, fallback_date: Optional[str]) -> Dict[str, pd.DataFrame]:
    """
    Returns dict date->df. If date values are empty and fallback_date provided, uses it.
    If still empty, returns single bucket "unknown".
    """
    df = _ensure_datetime_cols(df)

    # Normalize Date values
    df[COL_DATE] = df[COL_DATE].astype(str).str.strip()
    # If all empty and fallback_date exists
    if (df[COL_DATE] == "").all() and fallback_date:
        df[COL_DATE] = fallback_date

    # Still empty?
    if (df[COL_DATE] == "").all():
        return {"unknown": df}

    buckets: Dict[str, pd.DataFrame] = {}
    for d, part in df.groupby(COL_DATE, dropna=False):
        dd = str(d).strip() if d is not None else ""
        dd = dd or "unknown"
        buckets[dd] = part.copy()
    return buckets


def _safe_nunique(series: pd.Series) -> int:
    if series is None:
        return 0
    s = series.astype(str)
    s = s[s.str.strip() != ""]
    return int(s.nunique(dropna=True))


def _hour_shares(df: pd.DataFrame) -> Tuple[int, float, float]:
    """
    Returns: unique_hours, night_share (00-05), business_share (09-17)
    """
    if "_ts" not in df.columns or df["_ts"].isna().all():
        return 0, 0.0, 0.0
    hours = df["_ts"].dt.hour
    unique_hours = int(hours.nunique(dropna=True))
    total = len(df)
    if total <= 0:
        return unique_hours, 0.0, 0.0
    night = ((hours >= 0) & (hours <= 5)).sum()
    business = ((hours >= 9) & (hours <= 17)).sum()
    return unique_hours, float(night) / total, float(business) / total


def _extract_user_set(df: pd.DataFrame) -> int:
    users: List[str] = []
    for col in (COL_SRC_USER, COL_DST_USER):
        if col in df.columns:
            users.extend([u for u in df[col].tolist()])
    normed = []
    for u in users:
        nu = _norm_user_token(u)
        if nu:
            normed.append(nu.lower())
    return len(set(normed))


def _build_row(entity: str, date: str, source: str, df: pd.DataFrame, include_unique_users: bool) -> Dict[str, object]:
    df = _ensure_datetime_cols(df)

    # Common nunique
    unique_dest = _safe_nunique(df[COL_DEST_ADDR]) if COL_DEST_ADDR in df.columns else 0
    unique_proc = _safe_nunique(df[COL_SRC_PROC]) if COL_SRC_PROC in df.columns else 0
    unique_class = _safe_nunique(df[COL_EVENT_CLASS]) if COL_EVENT_CLASS in df.columns else 0
    unique_cat = _safe_nunique(df[COL_EVENT_CAT]) if COL_EVENT_CAT in df.columns else 0
    unique_name = _safe_nunique(df[COL_NAME]) if COL_NAME in df.columns else 0

    unique_hours, night_share, business_share = _hour_shares(df)

    row = {
        "entity": entity,
        "date": date,
        f"{source.lower()}_events_total": int(len(df)),
        f"{source.lower()}_unique_destination_addr": unique_dest,
        f"{source.lower()}_unique_source_process": unique_proc if source == "SIEM" else 0,
        f"{source.lower()}_unique_event_class": unique_class,
        f"{source.lower()}_unique_category": unique_cat,
        f"{source.lower()}_unique_name": unique_name,
        f"{source.lower()}_unique_hours": unique_hours,
        f"{source.lower()}_night_share": night_share,
        f"{source.lower()}_business_share": business_share,
    }
    if include_unique_users:
        row[f"{source.lower()}_unique_users"] = _extract_user_set(df)
    return row


def _merge_rows(base: Dict[str, object], other: Dict[str, object]) -> Dict[str, object]:
    # base has entity/date; merge feature columns, keeping entity/date
    out = dict(base)
    for k, v in other.items():
        if k in ("entity", "date"):
            continue
        out[k] = v
    return out


def _finalize(df: pd.DataFrame, kind: str) -> pd.DataFrame:
    """
    Ensure consistent column set and derived features.
    """
    # Build minimal schema
    base_cols = ["entity", "date"]
    if df.empty:
        # Return empty with schema
        if kind == "users":
            cols = base_cols + [
                "siem_events_total", "pan_events_total",
                "siem_unique_destination_addr", "pan_unique_destination_addr",
                "siem_unique_source_process",
                "siem_unique_event_class", "pan_unique_event_class",
                "siem_unique_category", "pan_unique_category",
                "siem_unique_name", "pan_unique_name",
                "siem_unique_hours", "pan_unique_hours",
                "siem_night_share", "pan_night_share",
                "siem_business_share", "pan_business_share",
                "day_of_week", "is_weekend", "pan_share_of_all_events",
            ]
        else:
            cols = base_cols + [
                "siem_events_total", "pan_events_total",
                "siem_unique_destination_addr", "pan_unique_destination_addr",
                "siem_unique_source_process",
                "siem_unique_event_class", "pan_unique_event_class",
                "siem_unique_category", "pan_unique_category",
                "siem_unique_name", "pan_unique_name",
                "siem_unique_hours", "pan_unique_hours",
                "siem_night_share", "pan_night_share",
                "siem_business_share", "pan_business_share",
                "siem_unique_users", "pan_unique_users",
                "day_of_week", "is_weekend", "pan_share_of_all_events",
            ]
        return pd.DataFrame(columns=cols)

    # Derived time features
    dts = pd.to_datetime(df["date"], errors="coerce")
    df["day_of_week"] = dts.dt.dayofweek.fillna(-1).astype(int)
    df["is_weekend"] = df["day_of_week"].isin([5, 6]).astype(int)

    # Ensure totals exist
    for col in ("siem_events_total", "pan_events_total"):
        if col not in df.columns:
            df[col] = 0

    total = df["siem_events_total"].astype(float) + df["pan_events_total"].astype(float)
    df["pan_share_of_all_events"] = (df["pan_events_total"].astype(float) / total).where(total > 0, 0.0)

    # Fill missing feature columns with zeros
    def _ensure(cols: Iterable[str], default=0):
        for c in cols:
            if c not in df.columns:
                df[c] = default

    common = [
        "siem_unique_destination_addr", "pan_unique_destination_addr",
        "siem_unique_source_process",
        "siem_unique_event_class", "pan_unique_event_class",
        "siem_unique_category", "pan_unique_category",
        "siem_unique_name", "pan_unique_name",
        "siem_unique_hours", "pan_unique_hours",
        "siem_night_share", "pan_night_share",
        "siem_business_share", "pan_business_share",
    ]
    _ensure(common, 0)
    if kind == "hosts":
        _ensure(["siem_unique_users", "pan_unique_users"], 0)

    # Order columns
    ordered = ["entity", "date",
               "siem_events_total", "pan_events_total",
               "siem_unique_destination_addr", "pan_unique_destination_addr",
               "siem_unique_source_process",
               "siem_unique_event_class", "pan_unique_event_class",
               "siem_unique_category", "pan_unique_category",
               "siem_unique_name", "pan_unique_name",
               "siem_unique_hours", "pan_unique_hours",
               "siem_night_share", "pan_night_share",
               "siem_business_share", "pan_business_share"]
    if kind == "hosts":
        ordered += ["siem_unique_users", "pan_unique_users"]
    ordered += ["day_of_week", "is_weekend", "pan_share_of_all_events"]

    df = df[ordered].copy()
    # Sort if possible
    if "entity" in df.columns and "date" in df.columns:
        df = df.sort_values(["entity", "date"]).reset_index(drop=True)
    return df


def build_features(work_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    metas: List[FileMeta] = []
    skipped: List[Tuple[str, str]] = []

    for p in sorted(work_dir.glob("*.csv")):
        meta = _parse_meta(p)
        if meta is None:
            skipped.append((p.name, "filename_not_recognized_or_excluded"))
            continue
        metas.append(meta)

    if not metas:
        print("[!] No input CSV files recognized in work directory.", file=sys.stderr)
        print("    Expected patterns like: user001_SIEM_YYYY-MM-DD.csv or host_PAN_YYYY-MM-DD.csv", file=sys.stderr)

    # Store aggregated rows keyed by (entity,date)
    user_rows: Dict[Tuple[str, str], Dict[str, object]] = {}
    host_rows: Dict[Tuple[str, str], Dict[str, object]] = {}

    parsed_files = 0
    produced_buckets = 0

    for meta in metas:
        try:
            df = _read_csv_safely(meta.path)
        except Exception as e:
            skipped.append((meta.path.name, f"read_error:{type(e).__name__}"))
            continue

        buckets = _split_by_date(df, meta.date_from_name)
        parsed_files += 1
        produced_buckets += len(buckets)

        for date, part in buckets.items():
            key = (meta.entity, date)

            is_user_entity = meta.entity.lower().startswith("user") and meta.entity[4:].isdigit()
            include_unique_users = not is_user_entity  # only for hosts

            row = _build_row(meta.entity, date, meta.source, part, include_unique_users)

            if is_user_entity:
                base = user_rows.get(key, {"entity": meta.entity, "date": date})
                user_rows[key] = _merge_rows(base, row)
            else:
                base = host_rows.get(key, {"entity": meta.entity, "date": date})
                host_rows[key] = _merge_rows(base, row)

    users_df = pd.DataFrame(list(user_rows.values()))
    hosts_df = pd.DataFrame(list(host_rows.values()))

    users_df = _finalize(users_df, kind="users")
    hosts_df = _finalize(hosts_df, kind="hosts")

    # Diagnostics
    print(f"[*] Work dir: {work_dir}")
    print(f"[*] Found CSV files: {len(list(work_dir.glob('*.csv')))}")
    print(f"[*] Recognized inputs: {len(metas)} | Parsed: {parsed_files} | Produced day-buckets: {produced_buckets}")
    print(f"[*] Output rows: users={len(users_df)} hosts={len(hosts_df)}")
    if skipped:
        # Show first 15 skips
        print("[!] Skipped/ignored files (first 15):")
        for fn, reason in skipped[:15]:
            print(f"    - {fn}: {reason}")
        if len(skipped) > 15:
            print(f"    ... and {len(skipped)-15} more")

    return users_df, hosts_df


def main() -> int:
    ap = argparse.ArgumentParser(description="Build per-day feature tables for anomaly detection (IsolationForest/LOF).")
    ap.add_argument("--work", required=True, help="Path to work directory with per-day CSV files.")
    ap.add_argument("--out-users", default="features_users.csv", help="Output filename for users feature table.")
    ap.add_argument("--out-hosts", default="features_hosts.csv", help="Output filename for hosts feature table.")
    args = ap.parse_args()

    work_dir = Path(args.work).resolve()
    if not work_dir.exists() or not work_dir.is_dir():
        print(f"[!] Work directory not found or not a directory: {work_dir}", file=sys.stderr)
        return 2

    users_df, hosts_df = build_features(work_dir)

    users_out = work_dir / args.out_users
    hosts_out = work_dir / args.out_hosts
    users_df.to_csv(users_out, index=False)
    hosts_df.to_csv(hosts_out, index=False)
    print(f"[+] Wrote: {users_out}")
    print(f"[+] Wrote: {hosts_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
