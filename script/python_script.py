#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Normalize SIEM exports (TSV) for ML analysis:
- reads all *.tsv / *.txt from ./data (relative to this script)
- drops columns: ClusterID, ClusterName, TenantID, TenantName (if present)
- anonymizes users to userXXX (user001, user002, ...)
- splits events into daily CSV files
- exports PAN-OS TRAFFIC events (Palo Alto Networks) into separate daily CSV files
- exports user mapping table (real_user -> userXXX) as CSV

Usage:
  python python_script.py
  python python_script.py --data ./data --work ./work --chunksize 200000
"""

from __future__ import annotations

import argparse
import csv
import re
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple, List

import pandas as pd


DROP_COLS = {"ClusterID", "ClusterName", "TenantID", "TenantName"}

# Most common time column names in SIEM exports (you can extend this list safely)
TIME_COL_CANDIDATES = [
    "Timestamp", "EventTime", "event_time", "time", "TimeGenerated", "StartTime", "EndTime"
]

# Columns where user identifiers are most likely to appear
USER_COL_HINTS = ("user", "account", "login", "principal", "subject", "sourceusername", "destinationusername")

PAN_TRAFFIC_SIGNATURE = {
    "Name": "TRAFFIC",
    "DeviceProduct": "PAN-OS",
    "DeviceVendor": "Palo Alto Networks",
}


@dataclass(frozen=True)
class FileRole:
    """Heuristic role of the input file: user-focused, host-focused or unknown."""
    kind: str  # "user" | "host" | "unknown"


def detect_file_role(stem: str) -> FileRole:
    up = stem.upper()
    if up.startswith(("U.D.", "UD_", "U_", "USER_", "USER.")):
        return FileRole("user")
    if up.startswith(("C.D.", "CD_", "C_", "HOST_", "PC_", "COMPUTER_", "HOST.", "TMTP-", "SRV-", "WS-", "WKST-")) or "TMTP" in up:
        return FileRole("host")
    return FileRole("unknown")


def extract_real_user_from_filename(stem: str) -> str:
    """
    Best-effort extraction of the real user name from file name.
    Examples:
      U.D.a.avakian -> a.avakian
      a.avakian     -> a.avakian
      user_name     -> user_name
    """
    parts = stem.split(".")
    if len(parts) >= 3 and parts[0].upper() in {"U", "C"} and parts[1].upper() == "D":
        # U.D.<user>
        return ".".join(parts[2:])
    return stem


def find_time_col(columns: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    for c in TIME_COL_CANDIDATES:
        if c in cols:
            return c
    # fallback: anything that contains 'time' or 'date' and looks plausible
    for c in cols:
        lc = c.lower()
        if "time" in lc or "timestamp" in lc or "date" in lc:
            return c
    return None


def is_pan_traffic(df: pd.DataFrame) -> pd.Series:
    required = list(PAN_TRAFFIC_SIGNATURE.keys())
    if not all(col in df.columns for col in required):
        return pd.Series([False] * len(df), index=df.index)
    mask = pd.Series([True] * len(df), index=df.index)
    for col, val in PAN_TRAFFIC_SIGNATURE.items():
        mask &= (df[col].astype(str) == val)
    return mask


def maybe_extract_hostname(df: pd.DataFrame) -> Optional[str]:
    """
    Tries to derive a stable hostname/computer account from data.
    Preference: a column with computer account ending with '$' (e.g. TMTP-003672$).
    """
    candidates: List[str] = []
    for col in df.columns:
        if col.lower() in {"sourceusername", "destinationusername", "computer", "hostname", "devicehostname", "host"}:
            s = df[col].astype(str)
            candidates.extend([v for v in s.tolist() if isinstance(v, str) and v.endswith("$") and len(v) > 1])
    if not candidates:
        # broad scan (cheaper on small chunks, still acceptable on first chunk)
        for col in df.columns:
            s = df[col].astype(str)
            sample = s.head(5000).tolist()
            candidates.extend([v for v in sample if isinstance(v, str) and v.endswith("$") and len(v) > 1])
    if not candidates:
        return None
    most_common, _ = Counter(candidates).most_common(1)[0]
    return most_common.rstrip("$")


def normalize_timestamp(df: pd.DataFrame, time_col: str) -> Tuple[pd.DataFrame, str]:
    """
    Converts timestamp column to a human-readable format and adds a 'Date' column (YYYY-MM-DD).
    The original time column is overwritten as string: 'YYYY-MM-DD HH:MM:SS' (local).
    """
    ts = pd.to_datetime(df[time_col], errors="coerce", utc=False)
    # If timezone info exists, pandas keeps tz-aware timestamps; we only need stable "local" view
    df[time_col] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["Date"] = ts.dt.strftime("%Y-%m-%d")
    return df, time_col


def anonymize_user_values(df: pd.DataFrame, real_user: str, alias: str) -> pd.DataFrame:
    """
    Replaces occurrences of real_user with alias (case-insensitive) in likely user/account columns.
    Also replaces DOMAIN\\user or user@domain patterns by alias (full value replaced with alias).
    """
    real_user = (real_user or "").strip()
    if not real_user:
        return df

    real_low = real_user.lower()

    # Build a conservative set of columns to touch
    cols_to_touch = []
    for c in df.columns:
        lc = c.lower()
        if any(h in lc for h in USER_COL_HINTS):
            cols_to_touch.append(c)

    # If no hint-based columns exist, fall back to touching DestinationUserName/SourceUserName if present
    for c in ("DestinationUserName", "SourceUserName", "UserName", "AccountName"):
        if c in df.columns and c not in cols_to_touch:
            cols_to_touch.append(c)

    if not cols_to_touch:
        return df

    domain_suffix_re = re.compile(r"(?i)(^.*\\)?%s$" % re.escape(real_user))
    upn_re = re.compile(r"(?i)^%s@.*$" % re.escape(real_user))

    for c in cols_to_touch:
        s = df[c].astype(str)

        # exact match (case-insensitive)
        mask_exact = s.str.lower() == real_low

        # DOMAIN\user  -> alias
        mask_domain = s.apply(lambda v: bool(domain_suffix_re.match(v)) if isinstance(v, str) else False)

        # user@domain -> alias
        mask_upn = s.apply(lambda v: bool(upn_re.match(v)) if isinstance(v, str) else False)

        mask = mask_exact | mask_domain | mask_upn
        if mask.any():
            df.loc[mask, c] = alias

    return df


def safe_append_csv(df: pd.DataFrame, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    write_header = not out_path.exists()
    df.to_csv(out_path, index=False, mode="a", header=write_header, encoding="utf-8")


def normalize_file(
    path: Path,
    data_role: FileRole,
    user_map: Dict[str, str],
    next_user_index: int,
    work_dir: Path,
    chunksize: int = 200_000,
) -> int:
    """
    Processes one TSV file. Returns updated next_user_index.
    """
    stem = path.stem
    real_user = extract_real_user_from_filename(stem)

    # user alias assignment
    if real_user not in user_map and real_user.strip():
        user_map[real_user] = f"user{next_user_index:03d}"
        next_user_index += 1
    user_alias = user_map.get(real_user, "user000")

    # read in chunks for scalability
    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        encoding="utf-8",
        engine="python",
        chunksize=chunksize,
    )

    hostname: Optional[str] = None

    for i, chunk in enumerate(reader):
        # drop noisy columns (if present)
        chunk = chunk.drop(columns=[c for c in DROP_COLS if c in chunk.columns], errors="ignore")

        time_col = find_time_col(chunk.columns)
        if not time_col:
            # If there is no time column, we cannot split by day; dump as-is
            out = work_dir / f"{user_alias}_SIEM_no_time.csv"
            safe_append_csv(chunk, out)
            continue

        chunk, time_col = normalize_timestamp(chunk, time_col)
        chunk = anonymize_user_values(chunk, real_user, user_alias)

        # host name is derived once (first chunk) for host-focused files
        if hostname is None and data_role.kind == "host":
            hostname = maybe_extract_hostname(chunk) or stem

        # split PAN traffic vs other SIEM events
        pan_mask = is_pan_traffic(chunk)
        non_pan = chunk.loc[~pan_mask].copy()
        pan = chunk.loc[pan_mask].copy()

        # write per-day
        for date, df_day in non_pan.groupby("Date", dropna=False):
            df_day = df_day.sort_values(by=[time_col], kind="mergesort")
            if data_role.kind == "host":
                prefix = (hostname or stem)
            else:
                prefix = user_alias
            out = work_dir / f"{prefix}_SIEM_{date}.csv"
            safe_append_csv(df_day, out)

        for date, df_day in pan.groupby("Date", dropna=False):
            df_day = df_day.sort_values(by=[time_col], kind="mergesort")
            out = work_dir / f"{user_alias}_PAN_{date}.csv"
            safe_append_csv(df_day, out)

    return next_user_index


def write_user_mapping(work_dir: Path, user_map: Dict[str, str]) -> Path:
    out = work_dir / "user_mapping.csv"
    rows = sorted(user_map.items(), key=lambda x: x[1])
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_name", "userXXX"])
        for real, alias in rows:
            w.writerow([real, alias])
    return out


def iter_input_files(data_dir: Path) -> List[Path]:
    files: List[Path] = []
    for ext in ("*.tsv", "*.txt"):
        files.extend(sorted(data_dir.glob(ext)))
    return files


def main() -> int:
    p = argparse.ArgumentParser(description="Normalize SIEM TSV exports for ML analysis.")
    p.add_argument("--data", type=Path, default=None, help="Input folder with TSV files (default: ./data)")
    p.add_argument("--work", type=Path, default=None, help="Output folder (default: ./work)")
    p.add_argument("--chunksize", type=int, default=200_000, help="Read chunksize for large TSV files.")
    args = p.parse_args()

    script_dir = Path(__file__).resolve().parent
    data_dir = args.data or (script_dir / "data")
    work_dir = args.work or (script_dir / "work")
    work_dir.mkdir(parents=True, exist_ok=True)

    files = iter_input_files(data_dir)
    if not files:
        print(f"[!] No input files found in: {data_dir}")
        return 2

    user_map: Dict[str, str] = {}
    next_user_index = 1

    for path in files:
        role = detect_file_role(path.stem)
        print(f"[*] Processing: {path.name} (role={role.kind})")
        next_user_index = normalize_file(
            path=path,
            data_role=role,
            user_map=user_map,
            next_user_index=next_user_index,
            work_dir=work_dir,
            chunksize=args.chunksize,
        )

    mapping_path = write_user_mapping(work_dir, user_map)
    print(f"[+] Done. Output folder: {work_dir}")
    print(f"[+] User mapping: {mapping_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
