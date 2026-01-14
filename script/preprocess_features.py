#!/usr/bin/env python3
"""
preprocess_features.py

Подготовка таблиц признаков для алгоритмов поиска аномалий (Isolation Forest / LOF).

Что делает:
  - читает features_users.csv и features_hosts.csv из папки features;
  - приводит колонку date к формату YYYY-MM-DD;
  - заполняет пропуски в числовых признаках нулями;
  - сохраняет *_clean.csv в папку features (по умолчанию).

Запуск:
  python preprocess_features.py --work ./features
  python preprocess_features.py --users ./features/features_users.csv --hosts ./features/features_hosts.csv --out-dir ./features
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


ID_COLS: list[str] = ["entity", "date"]


def _load_csv(path: Path) -> pd.DataFrame:
    """Читает CSV с типом str для стабильности идентификаторов."""
    # dtype=str сохраняет идентификаторы как строки.
    return pd.read_csv(path, dtype=str, keep_default_na=True)


def _coerce_date(df: pd.DataFrame) -> pd.DataFrame:
    """Преобразует колонку date к ISO-формату (YYYY-MM-DD)."""
    if "date" in df.columns:
        # Парсим и берём только дату.
        dt = pd.to_datetime(df["date"], errors="coerce", utc=False)
        # Если уже YYYY-MM-DD, то остается без изменений.
        df["date"] = dt.dt.date.astype("string")
    return df


def _coerce_features_numeric(df: pd.DataFrame) -> pd.DataFrame:
    """Приводит все не-ID колонки к числовому типу."""
    # Ошибки преобразования становятся NaN, затем заменяются на 0.
    for col in df.columns:
        if col in ID_COLS:
            continue
        # Приводим к числовому типу, сохраняя NaN при ошибках.
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def _fillna_zero(df: pd.DataFrame) -> pd.DataFrame:
    """Заполняет пропуски в признаках нулями."""
    # Выделяем только признакные колонки.
    feature_cols: list[str] = [c for c in df.columns if c not in ID_COLS]
    # Заполняем пропуски нулями.
    df[feature_cols] = df[feature_cols].fillna(0)
    return df


def _validate(df: pd.DataFrame, name: str) -> None:
    """Проверяет наличие ключевых колонок и выводит предупреждения."""
    # Проверяем наличие обязательных колонок.
    missing: list[str] = [c for c in ID_COLS if c not in df.columns]
    if missing:
        raise ValueError(f"{name}: missing required columns: {missing}")
    # entity/date не должны быть пустыми для большинства строк.
    null_entity = df["entity"].isna().sum()
    null_date = df["date"].isna().sum()
    if null_entity or null_date:
        print(f"[!] Warning: {name} has nulls: entity={null_entity}, date={null_date}")


def preprocess(in_path: Path, out_path: Path, name: str) -> None:
    """Основная логика подготовки данных (чтение -> нормализация -> запись)."""
    # Загружаем исходные данные.
    df = _load_csv(in_path)
    # Валидируем структуру таблицы.
    _validate(df, name)
    # Приводим даты к стандарту.
    df = _coerce_date(df)
    # Конвертируем признаки в числовые значения.
    df = _coerce_features_numeric(df)
    # Заменяем пропуски нулями.
    df = _fillna_zero(df)

    # Сортировка для детерминированного результата.
    df = df.sort_values(["entity", "date"], kind="mergesort").reset_index(drop=True)

    # Создаём выходную директорию и сохраняем файл.
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    # Печатаем сводную информацию о файле.
    print(f"[+] Wrote {name}: {out_path} (rows={len(df)}, cols={len(df.columns)})")


def main() -> int:
    # Описываем параметры командной строки.
    p = argparse.ArgumentParser()
    p.add_argument("--work", type=str, default="features", help="Features directory containing features_*.csv")
    p.add_argument("--users", type=str, default=None, help="Path to features_users.csv")
    p.add_argument("--hosts", type=str, default=None, help="Path to features_hosts.csv")
    p.add_argument("--out-dir", type=str, default=None, help="Output directory (default: features dir)")
    p.add_argument("--out-users", type=str, default="features_users_clean.csv", help="Output filename for users")
    p.add_argument("--out-hosts", type=str, default="features_hosts_clean.csv", help="Output filename for hosts")
    # Читаем параметры CLI.
    args = p.parse_args()

    if args.work:
        # Режим с рабочей директорией.
        work: Path = Path(args.work)
        users_in: Path = Path(args.users) if args.users else work / "features_users.csv"
        hosts_in: Path = Path(args.hosts) if args.hosts else work / "features_hosts.csv"
        out_dir: Path = Path(args.out_dir) if args.out_dir else work
    else:
        # Режим с явными путями к файлам.
        if not args.users or not args.hosts:
            p.error("Either provide --work or both --users and --hosts")
        users_in = Path(args.users)
        hosts_in = Path(args.hosts)
        out_dir = Path(args.out_dir) if args.out_dir else users_in.parent

    # Определяем пути для выходных файлов.
    users_out: Path = out_dir / args.out_users
    hosts_out: Path = out_dir / args.out_hosts

    # Проверяем наличие входных файлов.
    if not users_in.exists():
        raise FileNotFoundError(f"Users features file not found: {users_in}")
    if not hosts_in.exists():
        raise FileNotFoundError(f"Hosts features file not found: {hosts_in}")

    # Обрабатываем пользователей и хосты по отдельности.
    preprocess(users_in, users_out, "users")
    preprocess(hosts_in, hosts_out, "hosts")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
