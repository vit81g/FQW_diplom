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
from urllib.parse import unquote


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

# Tokens that do not represent a real user identity in exports and must NOT be mapped to userXXX
INVALID_USER_TOKENS = {
    "", "-", "—", "–",
    "null", "(null)", "none", "nan",
    "n/a", "na", "undefined", "unknown",
}


@dataclass(frozen=True)
class FileRole:
    """Heuristic role of the input file: user-focused, host-focused or unknown."""
    kind: str  # "user" | "host" | "unknown"


def detect_file_role(stem: str) -> FileRole:
    """Heuristically detect whether file is user-focused or host-focused."""
    up = stem.upper()

    # Explicit prefixes
    if up.startswith(("U.D.", "UD_", "U_", "USER_", "USER.")):
        return FileRole("user")
    if up.startswith(("C.D.", "CD_", "C_", "HOST_", "PC_", "COMPUTER_", "HOST.", "TMTP-", "SRV-", "WS-", "WKST-")) or "TMTP" in up:
        return FileRole("host")

    # Pattern: <prefix>_<id>
    if "_" in stem:
        tail = stem.split("_")[-1]
        if "." in tail:
            return FileRole("user")
        if re.search(r"\d", tail) or "-" in tail:
            return FileRole("host")

    return FileRole("unknown")

def extract_owner_from_filename(stem: str) -> str:
    """Extract owner identifier from file stem.

    Examples:
      U.D.a.avakian  -> a.avakian
      12_A.Avakian   -> A.Avakian
      12_TMTP-003672 -> TMTP-003672
    """
    s = (stem or "").strip()
    if not s:
        return s

    up = s.upper()
    if up.startswith(("U.D.", "C.D.")):
        parts = s.split(".")
        if len(parts) >= 3:
            return ".".join(parts[2:])
        return s

    if "_" in s:
        return s.split("_")[-1]

    return s

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
    """Detect PAN-OS TRAFFIC events (Palo Alto Networks) in the export."""
    required = {"Name", "DeviceProduct", "DeviceVendor"}
    if not required.issubset(df.columns):
        return pd.Series([False] * len(df), index=df.index)

    name = df["Name"].astype(str).str.strip().str.upper()
    prod = df["DeviceProduct"].astype(str).str.strip().str.upper()
    vend = df["DeviceVendor"].astype(str)

    return name.eq("TRAFFIC") & prod.eq("PAN-OS") & vend.str.contains("Palo Alto Networks", case=False, na=False)

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

def _base_user_token(v: str) -> Optional[str]:
    """Return canonical username key used for anonymization.

    Normalizes different representations of the same user:
    - URL-encoded values (e.g. %5c, %40)
    - DOMAIN\\user  -> user
    - user@domain     -> user

    Excludes:
    - empty / placeholder tokens like '-' or 'null'
    - machine accounts ending with '$'
    """
    if v is None:
        return None
    if not isinstance(v, str):
        v = str(v)

    s = v.strip()
    if not s:
        return None

    # Treat URL-encoded values as equivalents (e.g. 'uclh%5cuser%40uclh.ru')
    try:
        s = unquote(s)
    except Exception:
        pass

    s = s.strip()
    if not s:
        return None

    low = s.lower()
    if low in INVALID_USER_TOKENS:
        return None

    # Drop machine accounts
    if s.endswith("$"):
        return None

    # DOMAIN\user
    if "\\" in s:
        s = s.split("\\")[-1]

    # user@domain
    if "@" in s:
        s = s.split("@")[0]

    s = s.strip()
    if not s:
        return None

    low = s.lower()
    if low in INVALID_USER_TOKENS:
        return None

    if s.endswith("$"):
        return None

    return low


def ensure_user_map_for_df(df: pd.DataFrame, user_map: Dict[str, str], next_user_index: int) -> int:
    """Add mappings for any user tokens seen in key columns of df."""
    cols = [c for c in ("SourceUserName", "DestinationUserName") if c in df.columns]
    if not cols:
        return next_user_index

    tokens = set()
    for c in cols:
        for v in df[c].head(5000).tolist():
            t = _base_user_token(v)
            if t:
                tokens.add(t)

    for t in sorted(tokens, key=lambda x: x.lower()):
        if t not in user_map:
            user_map[t] = f"user{next_user_index:03d}"
            next_user_index += 1

    return next_user_index


def apply_user_map(df: pd.DataFrame, user_map: Dict[str, str]) -> pd.DataFrame:
    """Replace usernames in SourceUserName/DestinationUserName using user_map (keys are lowercased)."""
    if df.empty or not user_map:
        return df

    cols = [c for c in ("SourceUserName", "DestinationUserName") if c in df.columns]
    if not cols:
        return df

    def _map(v: str) -> str:
        t = _base_user_token(v)
        if not t:
            return v
        alias = user_map.get(t)
        return alias if alias else v

    for c in cols:
        df[c] = df[c].map(_map)

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
    owner = extract_owner_from_filename(stem)

    # If role is still unknown, infer from owner token (dot -> user, otherwise host)
    role_kind = data_role.kind
    if role_kind == "unknown":
        role_kind = "user" if "." in owner else "host"

    # Ensure owner has a stable alias for user-focused files
    user_alias: Optional[str] = None
    if role_kind == "user":
        owner_key = owner.strip().lower()
        if owner_key and owner_key not in user_map:
            user_map[owner_key] = f"user{next_user_index:03d}"
            next_user_index += 1
        user_alias = user_map.get(owner_key, "user000")


    # read in chunks for scalability
    # NOTE: SIEM TSV payloads may contain unescaped quotes inside fields (e.g., Windows event text).
    # Using QUOTE_NONE makes the parser treat quotes as ordinary characters and prevents ParserError.
    reader = pd.read_csv(
        path,
        sep="\t",
        dtype=str,
        keep_default_na=False,
        na_filter=False,
        encoding="utf-8",
        encoding_errors="replace",
        engine="python",
        quoting=csv.QUOTE_NONE,
        chunksize=chunksize,
    )

    hostname: Optional[str] = None

    for i, chunk in enumerate(reader):
        # drop noisy columns (if present)
        chunk = chunk.drop(columns=[c for c in DROP_COLS if c in chunk.columns], errors="ignore")

        time_col = find_time_col(chunk.columns)
        if not time_col:
            # If there is no time column, we cannot split by day; dump as-is
            prefix = (hostname or owner or stem) if role_kind == "host" else (user_alias or "user000")
            out = work_dir / f"{prefix}_SIEM_no_time.csv"
            safe_append_csv(chunk, out)
            continue

        chunk, time_col = normalize_timestamp(chunk, time_col)

        # Build/extend user mapping based on actual usernames in data and apply it
        next_user_index = ensure_user_map_for_df(chunk, user_map, next_user_index)
        chunk = apply_user_map(chunk, user_map)


        # host name is derived once (first chunk) for host-focused files
        if hostname is None and role_kind == "host":
            hostname = maybe_extract_hostname(chunk) or owner or stem

        # split PAN traffic vs other SIEM events
        pan_mask = is_pan_traffic(chunk)
        non_pan = chunk.loc[~pan_mask].copy()
        pan = chunk.loc[pan_mask].copy()

        # write per-day
        for date, df_day in non_pan.groupby("Date", dropna=False):
            df_day = df_day.sort_values(by=[time_col], kind="mergesort")
            if role_kind == "host":
                prefix = (hostname or owner or stem)
            else:
                prefix = (user_alias or "user000")
            out = work_dir / f"{prefix}_SIEM_{date}.csv"
            safe_append_csv(df_day, out)

        for date, df_day in pan.groupby("Date", dropna=False):
            df_day = df_day.sort_values(by=[time_col], kind="mergesort")
            prefix_pan = (hostname or owner or stem) if role_kind == "host" else (user_alias or "user000")
            out = work_dir / f"{prefix_pan}_PAN_{date}.csv"
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