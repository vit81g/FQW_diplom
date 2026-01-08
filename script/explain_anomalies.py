#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
explain_anomalies.py

Слой "объяснимости" для аналитиков SOC:
- для каждой аномальной сущности (user/host) на выбранную дату вычисляет 3–5 признаков,
  которые сильнее всего отклонились от базовой линии (median/MAD robust z-score);
- добавляет метку серьёзности (critical/high/medium/low) по процентилям ранга внутри дня.

Вход (в папке --work):
  - features_users_clean.csv
  - features_hosts_clean.csv
  - anomalies_users_YYYY-MM-DD.csv
  - anomalies_hosts_YYYY-MM-DD.csv

Выход (в папке --work):
  - anomalies_users_YYYY-MM-DD_explain.csv
  - anomalies_hosts_YYYY-MM-DD_explain.csv
  - anomalies_all_YYYY-MM-DD_explain.csv  (users+hosts вместе)

Запуск:
  python explain_anomalies.py --work .\\work --date 2025-12-31
  python explain_anomalies.py --work .\\work                (дата по умолчанию — последняя anomalies_*)
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


ID_COLS: list[str] = ["entity", "date"]
DATE_RE = re.compile(r"anomalies_(users|hosts)_(\d{4}-\d{2}-\d{2})\.csv$", re.IGNORECASE)


def _read_csv(path: Path, dtype=str) -> pd.DataFrame:
    """Универсальная обёртка над pd.read_csv."""
    return pd.read_csv(path, dtype=dtype, keep_default_na=True)


def _pick_date_from_workdir(work_dir: Path) -> str:
    """Выбирает последнюю дату по файлам anomalies_users_* и anomalies_hosts_*."""
    dates: List[str] = []
    for p in work_dir.glob("anomalies_*_*.csv"):
        m = DATE_RE.match(p.name)
        if m:
            dates.append(m.group(2))
    if not dates:
        raise FileNotFoundError("No anomalies_*.csv found in work dir. Run train_anomaly_models.py first.")
    # Сортировка по фактической дате.
    dt = pd.to_datetime(pd.Series(dates), errors="coerce")
    return str(dt.max().date())


def _robust_z(x: float, med: float, mad: float, fallback_std: float) -> float:
    """Робастный z-score: по MAD или std (fallback)."""
    # MAD -> робастный масштаб; 1.4826 приводит MAD к масштабу std для нормального распределения.
    if mad and mad > 0:
        return float((x - med) / (1.4826 * mad))
    if fallback_std and fallback_std > 0:
        return float((x - med) / fallback_std)
    # Если нет разброса, учитываем только знак отклонения (или 0).
    if x == med:
        return 0.0
    return float(np.sign(x - med) * 999.0)


def _baseline_stats(train_df: pd.DataFrame, feature_cols: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Возвращает (median_df, mad_df, std_df), индексированный по entity."""
    g = train_df.groupby("entity", dropna=False)

    med = g[feature_cols].median(numeric_only=True)
    # MAD: медиана |x - median|.
    mad = g[feature_cols].apply(lambda x: (x - x.median(numeric_only=True)).abs().median(numeric_only=True))
    std = g[feature_cols].std(numeric_only=True).fillna(0)

    return med, mad, std


def _severity_by_rank(df_day: pd.DataFrame) -> pd.Series:
    """
    Определяет уровень серьёзности по процентилям ранга внутри дня:
      - critical: top 5%
      - high:     следующие 15% (<=20%)
      - medium:   следующие 30% (<=50%)
      - low:      остальные
    """
    if "rank_combined" not in df_day.columns:
        # Запасной вариант: если есть score_combined_norm, строим rank.
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
    """Приводит признаки к числовому типу и заполняет NaN нулями."""
    df = df.copy()
    for c in feature_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feature_cols] = df[feature_cols].fillna(0)
    return df


def _explain_one_kind(
    work_dir: Path,
    kind: str,
    target_date: str,
    top_k: int,
) -> pd.DataFrame:
    """Строит объяснения для users или hosts на выбранную дату."""
    kind = kind.lower()
    if kind not in {"users", "hosts"}:
        raise ValueError("kind must be 'users' or 'hosts'")

    feats_file = work_dir / f"features_{kind}_clean.csv"
    anom_file = work_dir / f"anomalies_{kind}_{target_date}.csv"

    if not feats_file.exists():
        raise FileNotFoundError(f"Missing features file: {feats_file}")
    if not anom_file.exists():
        raise FileNotFoundError(f"Missing anomalies file: {anom_file}")

    feats: pd.DataFrame = _read_csv(feats_file, dtype=str)
    anom: pd.DataFrame = _read_csv(anom_file, dtype=str)

    # Нормализуем дату в обеих таблицах.
    feats["date"] = pd.to_datetime(feats["date"], errors="coerce").dt.date.astype("string")
    anom["date"] = pd.to_datetime(anom["date"], errors="coerce").dt.date.astype("string")

    # Выбираем признаки из таблицы features.
    feature_cols: list[str] = [c for c in feats.columns if c not in ID_COLS]
    if not feature_cols:
        raise ValueError(f"No feature columns detected in {feats_file.name}")

    feats = _coerce_features(feats, feature_cols)

    # Строим базовую линию по всем дням, кроме целевого.
    train: pd.DataFrame = feats[feats["date"] != target_date].copy()
    if train.empty:
        train = feats.copy()

    med_df, mad_df, std_df = _baseline_stats(train, feature_cols)

    # Отбираем строки за день и объединяем с таблицей аномалий.
    day: pd.DataFrame = feats[feats["date"] == target_date].copy()
    if day.empty:
        raise ValueError(f"{kind}: no feature rows for date {target_date} in {feats_file.name}")

    merged: pd.DataFrame = day.merge(anom, on=["entity", "date"], how="inner", suffixes=("", "_anom"))
    if merged.empty:
        # Если в anomalies меньше сущностей, чем в дневной выборке, это нормально.
        # Если join пустой — вероятно, ошибка в дате или именах.
        raise ValueError(f"{kind}: join between features and anomalies is empty for date {target_date}. "
                         f"Check that anomalies_{kind}_{target_date}.csv exists and matches 'entity' names.")

    # Уровень серьёзности по рангу внутри списка аномалий.
    sev: pd.Series = _severity_by_rank(merged)
    merged["severity"] = sev

    # Вычисляем топ факторов для каждой строки.
    contrib_rows: List[Dict[str, object]] = []

    for _, row in merged.iterrows():
        ent = row["entity"]
        # Если сущность не встречалась в train, используем глобальную базовую линию.
        if ent in med_df.index:
            med = med_df.loc[ent]
            mad = mad_df.loc[ent]
            std = std_df.loc[ent]
        else:
            med = train[feature_cols].median(numeric_only=True)
            mad = (train[feature_cols] - med).abs().median(numeric_only=True)
            std = train[feature_cols].std(numeric_only=True).fillna(0)

        z_list: list[tuple[str, float, float, float]] = []
        for c in feature_cols:
            x = float(row[c]) if pd.notna(row[c]) else 0.0
            z = _robust_z(x, float(med.get(c, 0.0)), float(mad.get(c, 0.0)), float(std.get(c, 0.0)))
            z_list.append((c, z, x, float(med.get(c, 0.0))))

        # Сортировка по абсолютному отклонению.
        z_list.sort(key=lambda t: abs(t[1]), reverse=True)
        top: list[tuple[str, float, float, float]] = z_list[:top_k]

        # Короткая строка и структурированные колонки.
        out = {
            "entity_type": "user" if kind == "users" else "host",
            "entity": ent,
            "date": row["date"],
            "severity": row["severity"],
        }

        # Пробрасываем оценки/ранги, если они есть.
        for col in ["score_isolation_forest", "score_lof", "score_combined_norm",
                    "rank_isolation_forest", "rank_lof", "rank_combined"]:
            if col in merged.columns:
                out[col] = row.get(col)

        # Компактная форма: feature:+z (значение vs базовая линия).
        parts = []
        for c, z, x, b in top:
            sign = "+" if z >= 0 else ""
            parts.append(f"{c}:{sign}{z:.2f} (v={x:.0f}, base={b:.0f})")
        out["top_contributors"] = "; ".join(parts)

        # Структурированные колонки contrib1..contribK.
        for i, (c, z, x, b) in enumerate(top, start=1):
            out[f"contrib{i}_feature"] = c
            out[f"contrib{i}_z"] = float(z)
            out[f"contrib{i}_value"] = float(x)
            out[f"contrib{i}_baseline_median"] = float(b)

        contrib_rows.append(out)

    report: pd.DataFrame = pd.DataFrame(contrib_rows)

    # Стабилизируем порядок колонок.
    preferred = [
        "entity_type", "entity", "date", "severity",
        "score_combined_norm", "rank_combined",
        "score_isolation_forest", "score_lof",
        "rank_isolation_forest", "rank_lof",
        "top_contributors",
    ]
    cols: list[str] = []
    for c in preferred:
        if c in report.columns:
            cols.append(c)
    # Затем добавляем остальные колонки.
    for c in report.columns:
        if c not in cols:
            cols.append(c)
    report = report[cols].sort_values(["severity", "rank_combined"], ascending=[True, True], kind="mergesort")

    out_file: Path = work_dir / f"anomalies_{kind}_{target_date}_explain.csv"
    report.to_csv(out_file, index=False)
    print(f"[+] Wrote: {out_file} (rows={len(report)})")
    return report


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, help="Work directory")
    p.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: latest anomalies_* date)")
    p.add_argument("--top-features", type=int, default=5, help="How many top deviating features to include (3-5 typical)")
    args = p.parse_args()

    work_dir: Path = Path(args.work)
    if not work_dir.exists():
        raise FileNotFoundError(f"Work dir not found: {work_dir}")

    target_date: str = args.date or _pick_date_from_workdir(work_dir)
    target_date = str(pd.to_datetime(target_date, errors="raise").date())

    k: int = int(args.top_features)
    if k < 1 or k > 10:
        raise ValueError("--top-features must be between 1 and 10")

    users: pd.DataFrame = _explain_one_kind(work_dir, "users", target_date, k)
    hosts: pd.DataFrame = _explain_one_kind(work_dir, "hosts", target_date, k)

    all_df: pd.DataFrame = pd.concat([users, hosts], ignore_index=True)
    out_all: Path = work_dir / f"anomalies_all_{target_date}_explain.csv"
    all_df.to_csv(out_all, index=False)
    print(f"[+] Wrote: {out_all} (rows={len(all_df)})")
    print("[*] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
