#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
python_script.py

Нормализация выгрузок SIEM (TSV) для последующего ML-анализа.

Функции скрипта:
- читает все *.tsv / *.txt из ./data (относительно каталога скрипта);
- удаляет служебные колонки (ClusterID, ClusterName, TenantID, TenantName), если они есть;
- анонимизирует пользователей в формат userXXX (user001, user002, ...);
- разбивает события по дням и сохраняет в CSV;
- отдельно выгружает события PAN-OS TRAFFIC (Palo Alto Networks) в отдельные CSV;
- сохраняет таблицу соответствия real_user -> userXXX.

Запуск:
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


DROP_COLS: set[str] = {"ClusterID", "ClusterName", "TenantID", "TenantName"}

# Наиболее распространённые варианты названия времени в выгрузках SIEM.
TIME_COL_CANDIDATES: list[str] = [
    "Timestamp", "EventTime", "event_time", "time", "TimeGenerated", "StartTime", "EndTime"
]

# Подсказки для колонок, где чаще всего встречаются имена пользователей.
USER_COL_HINTS: tuple[str, ...] = (
    "user",
    "account",
    "login",
    "principal",
    "subject",
    "sourceusername",
    "destinationusername",
)

PAN_TRAFFIC_SIGNATURE: dict[str, str] = {
    "Name": "TRAFFIC",
    "DeviceProduct": "PAN-OS",
    "DeviceVendor": "Palo Alto Networks",
}

# Токены, которые не являются реальными пользователями (их нельзя мапить на userXXX).
INVALID_USER_TOKENS: set[str] = {
    "", "-", "—", "–",
    "null", "(null)", "none", "nan",
    "n/a", "na", "undefined", "unknown",
}


@dataclass(frozen=True)
class FileRole:
    """Роль входного файла по эвристике: user/host/unknown."""
    kind: str  # "user" | "host" | "unknown"


def detect_file_role(stem: str) -> FileRole:
    """Эвристически определяет роль файла: пользовательский или хостовый."""
    up = stem.upper()

    # Явные префиксы.
    if up.startswith(("U.D.", "UD_", "U_", "USER_", "USER.")):
        return FileRole("user")
    if up.startswith(("C.D.", "CD_", "C_", "HOST_", "PC_", "COMPUTER_", "HOST.", "TMTP-", "SRV-", "WS-", "WKST-")) or "TMTP" in up:
        return FileRole("host")

    # Шаблон: <prefix>_<id>
    if "_" in stem:
        tail = stem.split("_")[-1]
        if "." in tail:
            return FileRole("user")
        if re.search(r"\d", tail) or "-" in tail:
            return FileRole("host")

    return FileRole("unknown")

def extract_owner_from_filename(stem: str) -> str:
    """Выделяет идентификатор владельца из имени файла.

    Примеры:
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
    """Возвращает название колонки времени, если она найдена."""
    cols: list[str] = list(columns)
    for c in TIME_COL_CANDIDATES:
        if c in cols:
            return c
    # Запасной вариант: любое поле, где есть 'time' или 'date'.
    for c in cols:
        lc = c.lower()
        if "time" in lc or "timestamp" in lc or "date" in lc:
            return c
    return None


def is_pan_traffic(df: pd.DataFrame) -> pd.Series:
    """Определяет события PAN-OS TRAFFIC (Palo Alto Networks) в выгрузке."""
    required = {"Name", "DeviceProduct", "DeviceVendor"}
    if not required.issubset(df.columns):
        return pd.Series([False] * len(df), index=df.index)

    name = df["Name"].astype(str).str.strip().str.upper()
    prod = df["DeviceProduct"].astype(str).str.strip().str.upper()
    vend = df["DeviceVendor"].astype(str)

    return name.eq("TRAFFIC") & prod.eq("PAN-OS") & vend.str.contains("Palo Alto Networks", case=False, na=False)

def maybe_extract_hostname(df: pd.DataFrame) -> Optional[str]:
    """Пытается извлечь устойчивое имя хоста/компьютера из данных."""
    candidates: List[str] = []
    for col in df.columns:
        if col.lower() in {"sourceusername", "destinationusername", "computer", "hostname", "devicehostname", "host"}:
            s = df[col].astype(str)
            candidates.extend([v for v in s.tolist() if isinstance(v, str) and v.endswith("$") and len(v) > 1])
    if not candidates:
        # Обход всех колонок (актуально для первого чанка).
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
    Приводит столбец времени к читабельному виду и добавляет колонку Date (YYYY-MM-DD).
    Оригинальный столбец времени перезаписывается строкой: 'YYYY-MM-DD HH:MM:SS'.
    """
    ts = pd.to_datetime(df[time_col], errors="coerce", utc=False)
    # Если есть tz-информация, pandas хранит tz-aware; нам нужен локальный "срез".
    df[time_col] = ts.dt.strftime("%Y-%m-%d %H:%M:%S")
    df["Date"] = ts.dt.strftime("%Y-%m-%d")
    return df, time_col


def anonymize_user_values(df: pd.DataFrame, real_user: str, alias: str) -> pd.DataFrame:
    """
    Заменяет значения real_user на alias (без учёта регистра) в пользовательских колонках.
    Также заменяет значения вида DOMAIN\\user или user@domain на alias.
    """
    real_user = (real_user or "").strip()
    if not real_user:
        return df

    real_low: str = real_user.lower()

    # Строим список колонок, где вероятнее всего содержится пользователь.
    cols_to_touch: list[str] = []
    for c in df.columns:
        lc = c.lower()
        if any(h in lc for h in USER_COL_HINTS):
            cols_to_touch.append(c)

    # Если явных "user" колонок нет, пробуем стандартные поля.
    for c in ("DestinationUserName", "SourceUserName", "UserName", "AccountName"):
        if c in df.columns and c not in cols_to_touch:
            cols_to_touch.append(c)

    if not cols_to_touch:
        return df

    def _normalize_user(value: str) -> str:
        if value is None:
            return value
        s = str(value)
        s_decoded = unquote(s)
        s_low = s_decoded.lower()
        if s_low == real_low:
            return alias
        if "\\" in s_decoded and s_decoded.split("\\")[-1].lower() == real_low:
            return alias
        if "@" in s_decoded and s_decoded.split("@")[0].lower() == real_low:
            return alias
        return s

    for col in cols_to_touch:
        df[col] = df[col].map(_normalize_user)
    return df

def _base_user_token(v: str) -> Optional[str]:
    """Возвращает канонический ключ пользователя для анонимизации.

    Нормализует разные представления одного пользователя:
    - URL-encoded значения (например, %5c, %40);
    - DOMAIN\\user  -> user;
    - user@domain     -> user.

    Исключает:
    - пустые/служебные токены вроде '-' или 'null';
    - машинные аккаунты, оканчивающиеся на '$'.
    """
    if v is None:
        return None
    if not isinstance(v, str):
        v = str(v)

    s = v.strip()
    if not s:
        return None

    # URL-encoded значения приводим к нормальной форме.
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

    # Исключаем машинные аккаунты.
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
    """Добавляет в user_map все новые пользователи, найденные в df."""
    cols = [c for c in ("SourceUserName", "DestinationUserName") if c in df.columns]
    if not cols:
        return next_user_index

    tokens: set[str] = set()
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
    """Заменяет пользователей в SourceUserName/DestinationUserName на alias из user_map."""
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
    """Безопасная запись CSV с дозаписью и заголовком только при первом создании файла."""
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
    Обрабатывает один TSV-файл. Возвращает обновлённый next_user_index.
    """
    stem = path.stem
    owner = extract_owner_from_filename(stem)

    # Если роль не определена, берём эвристику по owner (точка -> user, иначе host).
    role_kind = data_role.kind
    if role_kind == "unknown":
        role_kind = "user" if "." in owner else "host"

    # Для user-файлов обеспечиваем стабильный alias владельца.
    user_alias: Optional[str] = None
    if role_kind == "user":
        owner_key = owner.strip().lower()
        if owner_key and owner_key not in user_map:
            user_map[owner_key] = f"user{next_user_index:03d}"
            next_user_index += 1
        user_alias = user_map.get(owner_key, "user000")

    # Читаем по чанкам для больших файлов.
    # Примечание: в TSV могут быть неэкранированные кавычки. QUOTE_NONE помогает избежать ParserError.
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

    for chunk in reader:
        # Удаляем служебные колонки (если есть).
        chunk = chunk.drop(columns=[c for c in DROP_COLS if c in chunk.columns], errors="ignore")

        time_col = find_time_col(chunk.columns)
        if not time_col:
            # Если нет времени, нельзя делить на дни — пишем как есть.
            prefix = (hostname or owner or stem) if role_kind == "host" else (user_alias or "user000")
            out = work_dir / f"{prefix}_SIEM_no_time.csv"
            safe_append_csv(chunk, out)
            continue

        chunk, time_col = normalize_timestamp(chunk, time_col)

        # Обновляем user_map по данным и применяем.
        next_user_index = ensure_user_map_for_df(chunk, user_map, next_user_index)
        chunk = apply_user_map(chunk, user_map)


        # Для host-файлов определяем имя хоста по первому чанку.
        if hostname is None and role_kind == "host":
            hostname = maybe_extract_hostname(chunk) or owner or stem

        # Отделяем PAN-traffic от остальных событий.
        pan_mask = is_pan_traffic(chunk)
        non_pan = chunk.loc[~pan_mask].copy()
        pan = chunk.loc[pan_mask].copy()

        # Запись по дням.
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
    """Сохраняет таблицу соответствия реальный пользователь -> alias."""
    # Формируем путь к итоговому файлу.
    out = work_dir / "user_mapping.csv"
    # Сортируем пары по alias, чтобы результат был стабильным.
    rows = sorted(user_map.items(), key=lambda x: x[1])
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["user_name", "userXXX"])
        for real, alias in rows:
            w.writerow([real, alias])
    return out


def iter_input_files(data_dir: Path) -> List[Path]:
    """Возвращает список входных TSV/TXT файлов в папке data."""
    # Собираем все TSV/TXT файлы и сортируем для стабильности.
    files: List[Path] = []
    for ext in ("*.tsv", "*.txt"):
        files.extend(sorted(data_dir.glob(ext)))
    return files


def main() -> int:
    # Настраиваем аргументы CLI.
    p = argparse.ArgumentParser(description="Normalize SIEM TSV exports for ML analysis.")
    p.add_argument("--data", type=Path, default=None, help="Input folder with TSV files (default: ./data)")
    p.add_argument("--work", type=Path, default=None, help="Output folder (default: ./work)")
    p.add_argument("--chunksize", type=int, default=200_000, help="Read chunksize for large TSV files.")
    # Читаем аргументы.
    args = p.parse_args()

    # Определяем директории относительно скрипта.
    script_dir: Path = Path(__file__).resolve().parent
    data_dir: Path = args.data or (script_dir / "data")
    work_dir: Path = args.work or (script_dir / "work")
    # Гарантируем, что папка вывода существует.
    work_dir.mkdir(parents=True, exist_ok=True)

    # Ищем входные файлы.
    files = iter_input_files(data_dir)
    if not files:
        print(f"[!] No input files found in: {data_dir}")
        return 2

    # Таблица соответствия пользователь -> alias.
    user_map: Dict[str, str] = {}
    # Счётчик для генерации userXXX.
    next_user_index: int = 1

    for path in files:
        # Определяем роль файла по имени.
        role: FileRole = detect_file_role(path.stem)
        print(f"[*] Processing: {path.name} (role={role.kind})")
        # Нормализуем конкретный файл.
        next_user_index = normalize_file(
            path=path,
            data_role=role,
            user_map=user_map,
            next_user_index=next_user_index,
            work_dir=work_dir,
            chunksize=args.chunksize,
        )

    # Сохраняем таблицу соответствия userXXX.
    mapping_path = write_user_mapping(work_dir, user_map)
    print(f"[+] Done. Output folder: {work_dir}")
    print(f"[+] User mapping: {mapping_path.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
