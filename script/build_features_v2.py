#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
build_features_v2.py

Сбор ML-признаков (user/day и host/day) из суточных CSV в ./work.

Ожидаемые входы в work/:
- userXXX_SIEM_YYYY-MM-DD.csv
- userXXX_PAN_YYYY-MM-DD.csv
- hostname_SIEM_YYYY-MM-DD.csv
- hostname_PAN_YYYY-MM-DD.csv

Если PAN-события попали в SIEM-файлы (бывает) — скрипт дополнительно отделит их
по сигнатуре: Name=TRAFFIC, DeviceProduct=PAN-OS, DeviceVendor содержит "Palo Alto Networks".

Выход (features/):
- features_users.csv
- features_hosts.csv

Запуск:
python build_features_v2.py --work .\\work --features-dir .\\features
"""

from __future__ import annotations

import argparse
import re
from pathlib import Path
from typing import Dict, Tuple, Optional, List

import pandas as pd


EXCLUDE_FILES: set[str] = {
    "user_mapping.csv",
    "features_users.csv",
    "features_hosts.csv",
    "features_users_clean.csv",
    "features_hosts_clean.csv",
}

# Колонки вашей выгрузки (скрипт терпим к отсутствующим)
COL_TIMESTAMP: str = "Timestamp"
COL_DATE: str = "Date"
COL_NAME: str = "Name"
COL_DEVICE_PRODUCT: str = "DeviceProduct"
COL_DEVICE_VENDOR: str = "DeviceVendor"
COL_DST_ADDR: str = "DestinationAddress"
COL_SRC_PROC: str = "SourceProcessName"
COL_EVENT_CLASS: str = "DeviceEventClassID"
COL_CATEGORY: str = "DeviceEventCategory"
COL_DST_USER: str = "DestinationUserName"
COL_SRC_USER: str = "SourceUserName"

INVALID_USER_TOKENS: set[str] = {
    "", "-", "—", "–", "null", "none", "nan", "n/a", "na", "unknown", "undef", "undefined"
}

DATE_RE = re.compile(r"(\d{4}-\d{2}-\d{2})", re.IGNORECASE)


def _is_pan_traffic(df: pd.DataFrame) -> pd.Series:
    """Проверяет PAN-traffic: Name=TRAFFIC, DeviceProduct=PAN-OS, DeviceVendor содержит 'Palo Alto Networks'."""
    if not {COL_NAME, COL_DEVICE_PRODUCT, COL_DEVICE_VENDOR}.issubset(df.columns):
        return pd.Series([False] * len(df), index=df.index)
    name = df[COL_NAME].astype(str).str.strip().str.upper()
    prod = df[COL_DEVICE_PRODUCT].astype(str).str.strip().str.upper()
    vend = df[COL_DEVICE_VENDOR].astype(str).str.strip()
    return (
        name.eq("TRAFFIC")
        & prod.eq("PAN-OS")
        & vend.str.contains("Palo Alto Networks", case=False, na=False)
    )


def _extract_date_from_filename(name: str) -> Optional[str]:
    """Извлекает дату (YYYY-MM-DD) из имени файла."""
    m = DATE_RE.findall(name)
    return m[-1] if m else None


def _classify_kind_from_filename(name: str) -> Optional[str]:
    """Определяет тип файла по имени: SIEM или PAN."""
    up = name.upper()
    if "_SIEM_" in up or up.endswith("_SIEM.CSV") or "SIEM" in up:
        return "SIEM"
    if "_PAN_" in up or up.endswith("_PAN.CSV") or "PAN" in up:
        return "PAN"
    return None


def _extract_entity_from_filename(name: str) -> str:
    """Извлекает сущность (userXXX или hostname) из имени файла."""
    stem = Path(name).stem
    up = stem.upper()
    for token in ("_SIEM_", "_PAN_"):
        idx = up.find(token)
        if idx > 0:
            return stem[:idx]
    for token in ("_SIEM", "_PAN"):
        idx = up.find(token)
        if idx > 0:
            return stem[:idx]
    return stem


def _role_from_entity(entity: str) -> str:
    """Определяет роль сущности: user или host."""
    if re.fullmatch(r"user\d{3}", entity, flags=re.IGNORECASE):
        return "user"
    return "host"


def _safe_read_csv(path: Path) -> pd.DataFrame:
    """Читает суточные CSV как строки (данные уже нормализованы)."""
    return pd.read_csv(path, dtype=str, keep_default_na=True)


def _ensure_date_column(df: pd.DataFrame, fallback_date: Optional[str]) -> pd.DataFrame:
    """Обеспечивает наличие колонки Date, при необходимости — по Timestamp или имени файла."""
    if COL_DATE in df.columns:
        df[COL_DATE] = df[COL_DATE].astype(str)
        return df

    # Иначе пробуем Timestamp.
    if COL_TIMESTAMP in df.columns:
        dt = pd.to_datetime(df[COL_TIMESTAMP], errors="coerce")
        df[COL_DATE] = dt.dt.date.astype("string")
        return df

    # Иначе берём дату из имени файла.
    if fallback_date:
        df[COL_DATE] = fallback_date
        return df

    df[COL_DATE] = pd.NA
    return df


def _hour_series(df: pd.DataFrame) -> pd.Series:
    """Возвращает часы события (0-23) из Timestamp."""
    if COL_TIMESTAMP not in df.columns:
        return pd.Series([pd.NA] * len(df), index=df.index)
    dt = pd.to_datetime(df[COL_TIMESTAMP], errors="coerce")
    return dt.dt.hour


def _share_in_hours(hours: pd.Series, start: int, end: int) -> float:
    """Доля событий в диапазоне часов (включительно)."""
    if hours is None or hours.isna().all():
        return 0.0
    valid = hours.dropna().astype(int)
    if len(valid) == 0:
        return 0.0
    return float(((valid >= start) & (valid <= end)).mean())


def _nunique(df: pd.DataFrame, col: str) -> int:
    """Количество уникальных непустых значений в колонке."""
    if col not in df.columns:
        return 0
    s = df[col].dropna().astype(str).str.strip()
    s = s[s != ""]
    return int(s.nunique())


def _unique_users(df: pd.DataFrame) -> int:
    """Для host/day: сколько уникальных пользователей взаимодействовало с хостом."""
    cols = [c for c in (COL_SRC_USER, COL_DST_USER) if c in df.columns]
    if not cols:
        return 0
    s = pd.concat([df[c].astype(str) for c in cols], ignore_index=True)
    s = s.str.strip()
    s = s[~s.str.lower().isin(INVALID_USER_TOKENS)]
    s = s[~s.str.endswith("$", na=False)]
    s = s[s != ""]
    return int(s.nunique())


def _aggregate_one(entity: str, date: str, df_siem: pd.DataFrame, df_pan: pd.DataFrame, role: str) -> Dict[str, object]:
    """Собирает агрегированные признаки по одной сущности за день."""
    hours_siem = _hour_series(df_siem)
    hours_pan = _hour_series(df_pan)

    row: Dict[str, object] = {
        "entity": entity,
        "date": date,
        "day_of_week": int(pd.to_datetime(date).dayofweek) if pd.notna(date) else 0,
        "is_weekend": int(pd.to_datetime(date).dayofweek >= 5) if pd.notna(date) else 0,
    }

    # SIEM признаки.
    row.update({
        "siem_events_total": int(len(df_siem)),
        "siem_unique_destination_addr": _nunique(df_siem, COL_DST_ADDR),
        "siem_unique_source_process": _nunique(df_siem, COL_SRC_PROC),
        "siem_unique_event_class": _nunique(df_siem, COL_EVENT_CLASS),
        "siem_unique_category": _nunique(df_siem, COL_CATEGORY),
        "siem_unique_name": _nunique(df_siem, COL_NAME),
        "siem_unique_hours": int(hours_siem.dropna().nunique()) if not hours_siem.isna().all() else 0,
        "siem_night_share": _share_in_hours(hours_siem, 0, 5),
        "siem_business_share": _share_in_hours(hours_siem, 9, 17),
    })

    # PAN признаки.
    row.update({
        "pan_events_total": int(len(df_pan)),
        "pan_unique_destination_addr": _nunique(df_pan, COL_DST_ADDR),
        "pan_unique_event_class": _nunique(df_pan, COL_EVENT_CLASS),
        "pan_unique_category": _nunique(df_pan, COL_CATEGORY),
        "pan_unique_name": _nunique(df_pan, COL_NAME),
        "pan_unique_hours": int(hours_pan.dropna().nunique()) if not hours_pan.isna().all() else 0,
        "pan_night_share": _share_in_hours(hours_pan, 0, 5),
        "pan_business_share": _share_in_hours(hours_pan, 9, 17),
    })

    total = row["siem_events_total"] + row["pan_events_total"]
    row["pan_share_of_all_events"] = float(row["pan_events_total"] / total) if total > 0 else 0.0

    if role == "host":
        row["siem_unique_users"] = _unique_users(df_siem)
        row["pan_unique_users"] = _unique_users(df_pan)

    return row


def build_features(work_dir: Path) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Главная функция построения признаков для users и hosts."""
    all_csv: list[Path] = list(work_dir.glob("*.csv"))
    csv_files: list[Path] = sorted([p for p in all_csv if p.name not in EXCLUDE_FILES])

    print(f"[*] Work dir: {work_dir}")
    print(f"[*] Found CSV files: {len(all_csv)}")

    recognized: int = 0
    parsed: int = 0
    skipped: List[Tuple[str, str]] = []

    # (role, entity, day) -> {"SIEM": df, "PAN": df}
    buckets: Dict[Tuple[str, str, str], Dict[str, pd.DataFrame]] = {}

    for path in csv_files:
        fname = path.name
        entity = _extract_entity_from_filename(fname)
        role = _role_from_entity(entity)
        kind = _classify_kind_from_filename(fname) or "SIEM"
        fdate = _extract_date_from_filename(fname)

        recognized += 1
        try:
            df = _safe_read_csv(path)
            parsed += 1
        except Exception as e:
            skipped.append((fname, f"read_error: {e}"))
            continue

        df = _ensure_date_column(df, fdate)
        if df[COL_DATE].isna().all():
            skipped.append((fname, "no_date_in_file_or_columns"))
            continue

        # Если kind=SIEM и внутри есть PAN-traffic — отделяем.
        if kind == "SIEM":
            pan_mask = _is_pan_traffic(df)
            df_pan = df[pan_mask].copy()
            df_siem = df[~pan_mask].copy()
        else:
            df_pan = df.copy()
            df_siem = df.iloc[0:0].copy()

        for day, part in df_siem.groupby(COL_DATE, dropna=True):
            key = (role, entity, str(day))
            cur = buckets.setdefault(key, {}).get("SIEM", pd.DataFrame())
            buckets[key]["SIEM"] = pd.concat([cur, part], ignore_index=True)

        for day, part in df_pan.groupby(COL_DATE, dropna=True):
            key = (role, entity, str(day))
            cur = buckets.setdefault(key, {}).get("PAN", pd.DataFrame())
            buckets[key]["PAN"] = pd.concat([cur, part], ignore_index=True)

    user_rows: List[Dict[str, object]] = []
    host_rows: List[Dict[str, object]] = []

    for (role, entity, day), d in buckets.items():
        df_siem = d.get("SIEM", pd.DataFrame())
        df_pan = d.get("PAN", pd.DataFrame())
        row = _aggregate_one(entity, day, df_siem, df_pan, role)
        (user_rows if role == "user" else host_rows).append(row)

    users_df = pd.DataFrame(user_rows)
    hosts_df = pd.DataFrame(host_rows)

    print(f"[*] Recognized inputs: {recognized} | Parsed: {parsed} | Produced day-buckets: {len(buckets)}")
    print(f"[*] Output rows: users={len(users_df)} hosts={len(hosts_df)}")

    if skipped:
        print("[!] Skipped/ignored files (first 15):")
        for fn, why in skipped[:15]:
            print(f"    - {fn}: {why}")

    if not users_df.empty:
        users_df = users_df.sort_values(["entity", "date"]).reset_index(drop=True)
    if not hosts_df.empty:
        hosts_df = hosts_df.sort_values(["entity", "date"]).reset_index(drop=True)

    return users_df, hosts_df


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, help="Путь к папке work с суточными CSV")
    p.add_argument("--features-dir", default="features", help="Папка для вывода features_*.csv (default: features)")
    args = p.parse_args()

    work_dir = Path(args.work)
    if not work_dir.exists():
        raise FileNotFoundError(f"Work directory not found: {work_dir}")

    features_dir = Path(args.features_dir)
    if not features_dir.is_absolute():
        features_dir = work_dir.parent / features_dir

    users_df, hosts_df = build_features(work_dir)

    features_dir.mkdir(parents=True, exist_ok=True)
    out_users = features_dir / "features_users.csv"
    out_hosts = features_dir / "features_hosts.csv"
    users_df.to_csv(out_users, index=False)
    hosts_df.to_csv(out_hosts, index=False)

    print(f"[+] Wrote: {out_users}")
    print(f"[+] Wrote: {out_hosts}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
