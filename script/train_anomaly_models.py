#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
train_anomaly_models.py

Обучение и скоринг детекторов аномалий (Isolation Forest + LOF) на дневных таблицах признаков.
Экспортирует TOP-N аномалий для выбранной даты (по умолчанию — последняя дата в датасете).

Вход (work dir):
  - features_users_clean.csv
  - features_hosts_clean.csv

Выход (work dir):
  - anomalies_users_YYYY-MM-DD.csv
  - anomalies_hosts_YYYY-MM-DD.csv
  - anomalies_users_YYYY-MM-DD_meta.json
  - anomalies_hosts_YYYY-MM-DD_meta.json

Особенности:
- Модели обучаются на всех днях, кроме target day (train = df[date != target]).
  Если есть только один день, обучение выполняется на всех строках.
- LOF используется в режиме novelty для скоринга новых наблюдений.

Запуск:
  python train_anomaly_models.py --work .\\work
  python train_anomaly_models.py --work .\\work --date 2025-12-17 --top 30
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Tuple, Optional

import numpy as np
import pandas as pd

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


ID_COLS: list[str] = ["entity", "date"]


def _load_features(path: Path, name: str) -> pd.DataFrame:
    """Загружает таблицу признаков и нормализует колонку date."""
    if not path.exists():
        raise FileNotFoundError(f"{name} features file not found: {path}")
    df = pd.read_csv(path, dtype=str, keep_default_na=True)
    for c in ID_COLS:
        if c not in df.columns:
            raise ValueError(f"{name}: required column missing: {c}")
    # Нормализуем дату.
    dt = pd.to_datetime(df["date"], errors="coerce")
    if dt.isna().all():
        raise ValueError(f"{name}: could not parse any dates from column 'date'")
    df["date"] = dt.dt.date.astype("string")
    return df


def _prepare_matrix(df: pd.DataFrame) -> Tuple[pd.DataFrame, np.ndarray, list[str]]:
    """Готовит матрицу признаков (numeric) и возвращает список колонок."""
    feature_cols: list[str] = [c for c in df.columns if c not in ID_COLS]
    if not feature_cols:
        raise ValueError("No feature columns found (expected columns besides entity/date).")
    X: pd.DataFrame = df[feature_cols].copy()

    # Приводим к числовому типу и заполняем NaN нулями.
    for c in feature_cols:
        X[c] = pd.to_numeric(X[c], errors="coerce")
    X = X.fillna(0)

    return X, X.to_numpy(dtype=float), feature_cols


def _choose_target_date(df: pd.DataFrame, target: Optional[str]) -> str:
    """Выбирает целевую дату (явно заданную или максимальную)."""
    if target:
        # Проверяем формат, парсингом даты.
        t = pd.to_datetime(target, errors="raise").date()
        return str(t)
    # По умолчанию — максимальная дата.
    return str(pd.to_datetime(df["date"]).max().date())


def _split_train_test(df: pd.DataFrame, target_date: str) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Разделяет данные на train/test по дате."""
    test_df: pd.DataFrame = df[df["date"] == target_date].copy()
    if test_df.empty:
        raise ValueError(f"No rows found for target date: {target_date}. Available dates: {sorted(df['date'].unique())[:10]} ...")
    train_df: pd.DataFrame = df[df["date"] != target_date].copy()
    if train_df.empty:
        # Если есть только один день, используем весь набор.
        train_df = df.copy()
    return train_df, test_df


def _fit_models(
    X_train: np.ndarray,
    contamination: float,
    n_estimators: int,
    max_samples: str | int,
    n_neighbors: int,
    random_state: int,
) -> Dict[str, object]:
    """Обучает IsolationForest и LOF (novelty) с робастным масштабированием."""
    scaler: RobustScaler = RobustScaler(with_centering=True, with_scaling=True, unit_variance=False)

    iso: IsolationForest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples=max_samples,
        random_state=random_state,
        n_jobs=-1,
    )

    # LOF в режиме novelty для скоринга новых наблюдений.
    lof: LocalOutlierFactor = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        metric="minkowski",
        p=2,
    )

    iso_pipe: Pipeline = Pipeline([("scaler", scaler), ("model", iso)])
    lof_pipe: Pipeline = Pipeline([("scaler", scaler), ("model", lof)])

    iso_pipe.fit(X_train)
    lof_pipe.fit(X_train)

    return {"iso": iso_pipe, "lof": lof_pipe}


def _score(models: Dict[str, object], X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """Считает аномальные скоры по двум моделям."""
    iso_pipe: Pipeline = models["iso"]
    lof_pipe: Pipeline = models["lof"]

    # decision_function: чем больше, тем "нормальнее".
    iso_inlier = iso_pipe.decision_function(X)
    lof_inlier = lof_pipe.decision_function(X)

    # Переводим в аномальный скор: чем больше, тем более аномально.
    iso_score = -iso_inlier
    lof_score = -lof_inlier

    return iso_score, lof_score


def _minmax(a: np.ndarray) -> np.ndarray:
    """Нормализация min-max."""
    amin: float = float(np.min(a))
    amax: float = float(np.max(a))
    if not np.isfinite(amin) or not np.isfinite(amax) or amax == amin:
        return np.zeros_like(a, dtype=float)
    return (a - amin) / (amax - amin)


def _build_report(
    test_df: pd.DataFrame,
    iso_score: np.ndarray,
    lof_score: np.ndarray,
) -> pd.DataFrame:
    """Формирует итоговую таблицу скоринга и рангов."""
    out: pd.DataFrame = test_df[ID_COLS].copy()

    out["score_isolation_forest"] = iso_score
    out["score_lof"] = lof_score

    out["score_isolation_forest_norm"] = _minmax(iso_score)
    out["score_lof_norm"] = _minmax(lof_score)
    out["score_combined_norm"] = (out["score_isolation_forest_norm"] + out["score_lof_norm"]) / 2.0

    # Ранг: 1 = наиболее аномальный.
    out["rank_isolation_forest"] = out["score_isolation_forest"].rank(ascending=False, method="average")
    out["rank_lof"] = out["score_lof"].rank(ascending=False, method="average")
    out["rank_combined"] = (out["rank_isolation_forest"] + out["rank_lof"]) / 2.0

    out = out.sort_values(["rank_combined", "rank_isolation_forest", "rank_lof"]).reset_index(drop=True)
    return out


def _write_outputs(
    work_dir: Path,
    prefix: str,
    target_date: str,
    report_df: pd.DataFrame,
    meta: Dict[str, object],
    top_n: int,
) -> None:
    """Сохраняет CSV с топом аномалий и JSON с метаданными."""
    out_csv: Path = work_dir / f"anomalies_{prefix}_{target_date}.csv"
    out_json: Path = work_dir / f"anomalies_{prefix}_{target_date}_meta.json"

    report_top: pd.DataFrame = report_df.head(top_n).copy()
    report_top.to_csv(out_csv, index=False)

    with out_json.open("w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    print(f"[+] Wrote: {out_csv} (top={len(report_top)})")
    print(f"[+] Wrote: {out_json}")


def run_one(
    work_dir: Path,
    name: str,
    in_file: str,
    prefix: str,
    target_date: Optional[str],
    top_n: int,
    contamination: float,
    n_estimators: int,
    max_samples: str | int,
    n_neighbors: int,
    random_state: int,
) -> None:
    """Полный цикл обучения и скоринга для одного типа сущности (users/hosts)."""
    df: pd.DataFrame = _load_features(work_dir / in_file, name)
    tdate: str = _choose_target_date(df, target_date)
    train_df, test_df = _split_train_test(df, tdate)

    _, X_train, feature_cols = _prepare_matrix(train_df)
    _, X_test, _ = _prepare_matrix(test_df)

    models = _fit_models(
        X_train=X_train,
        contamination=contamination,
        n_estimators=n_estimators,
        max_samples=max_samples,
        n_neighbors=n_neighbors,
        random_state=random_state,
    )

    iso_score, lof_score = _score(models, X_test)
    report = _build_report(test_df, iso_score, lof_score)

    meta: Dict[str, object] = {
        "entity_type": prefix,
        "target_date": tdate,
        "rows_train": int(len(train_df)),
        "rows_test": int(len(test_df)),
        "feature_columns": feature_cols,
        "parameters": {
            "contamination": contamination,
            "isolation_forest": {
                "n_estimators": n_estimators,
                "max_samples": max_samples,
                "random_state": random_state,
            },
            "lof": {
                "n_neighbors": n_neighbors,
                "novelty": True,
            },
            "scaler": "RobustScaler",
        },
    }

    _write_outputs(work_dir, prefix, tdate, report, meta, top_n)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, help="Path to work directory")
    p.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: latest in dataset)")
    p.add_argument("--top", type=int, default=30, help="TOP-N anomalies to export per entity type")
    p.add_argument("--contamination", type=float, default=0.05, help="Expected anomaly fraction (0..0.5). Used by IF/LOF.")
    p.add_argument("--n-estimators", type=int, default=300, help="Isolation Forest: number of trees")
    p.add_argument("--max-samples", default="auto", help="Isolation Forest: max_samples (auto or int)")
    p.add_argument("--n-neighbors", type=int, default=20, help="LOF: n_neighbors (typical 10-30)")
    p.add_argument("--random-state", type=int, default=42, help="Random seed")
    args = p.parse_args()

    work_dir: Path = Path(args.work)
    if not work_dir.exists():
        raise FileNotFoundError(f"Work directory not found: {work_dir}")

    # Валидация диапазона contamination.
    if not (0.0 < args.contamination <= 0.5):
        raise ValueError("--contamination must be in (0, 0.5]")

    # Разбор параметра max_samples.
    max_samples: str | int
    if str(args.max_samples).lower() == "auto":
        max_samples = "auto"
    else:
        try:
            max_samples = int(args.max_samples)
        except Exception as e:
            raise ValueError("--max-samples must be 'auto' or an integer") from e

    run_one(
        work_dir=work_dir,
        name="users",
        in_file="features_users_clean.csv",
        prefix="users",
        target_date=args.date,
        top_n=args.top,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        max_samples=max_samples,
        n_neighbors=args.n_neighbors,
        random_state=args.random_state,
    )

    run_one(
        work_dir=work_dir,
        name="hosts",
        in_file="features_hosts_clean.csv",
        prefix="hosts",
        target_date=args.date,
        top_n=args.top,
        contamination=args.contamination,
        n_estimators=args.n_estimators,
        max_samples=max_samples,
        n_neighbors=args.n_neighbors,
        random_state=args.random_state,
    )

    print("[*] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
