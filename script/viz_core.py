#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
viz_core.py

Общие функции для визуализации и отчётности по аномалиям.

Читает:
  - features_users_clean.csv
  - features_hosts_clean.csv

Формирует графики и CSV-резюме в указанной директории.

Важно (для Windows):
  Используется backend Matplotlib "Agg", чтобы избежать падений Tkinter/TkAgg
  при сохранении PNG в неинтерактивных скриптах (например, auto_generate_reports.py).
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict, List

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")  # Важно: установка backend до импорта pyplot.
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline


ID_COLS: list[str] = ["entity", "date"]

SEV_RU: dict[str, str] = {
    "critical": "Критично",
    "high": "Высокий",
    "medium": "Средний",
    "low": "Низкий",
}

SEV_COLORS: dict[str, str] = {
    "critical": "#d62728",  # red
    "high": "#ff7f0e",      # orange
    "medium": "#bcbd22",    # olive
    "low": "#1f77b4",       # blue
}


def ensure_dir(p: Path) -> Path:
    """Создаёт директорию, если её нет."""
    # Создаём папку и всех родителей.
    p.mkdir(parents=True, exist_ok=True)
    return p


def load_features(work_dir: Path, kind: str) -> pd.DataFrame:
    """Загружает подготовленные признаки users/hosts и приводит к числовым типам."""
    # Формируем путь к файлу.
    p = work_dir / f"features_{kind}_clean.csv"
    if not p.exists():
        raise FileNotFoundError(
            f"Missing {p}. Expected outputs of preprocess_features.py in features dir."
        )
    # Читаем CSV как строки.
    df: pd.DataFrame = pd.read_csv(p, dtype=str, keep_default_na=True)
    for c in ID_COLS:
        if c not in df.columns:
            raise ValueError(f"{p.name}: missing required column: {c}")

    # Приводим дату к стандартизированному виду.
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")

    # Определяем список колонок с признаками.
    feat_cols: list[str] = [c for c in df.columns if c not in ID_COLS]
    if not feat_cols:
        raise ValueError(f"{p.name}: no feature columns found")

    # Переводим признаки в числовой тип.
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    # Заполняем пропуски нулями.
    df[feat_cols] = df[feat_cols].fillna(0)
    return df


def available_dates(df_users: pd.DataFrame, df_hosts: pd.DataFrame) -> List[str]:
    """Возвращает список уникальных дат, отсортированный по возрастанию."""
    # Собираем все даты из двух таблиц.
    d: pd.Series = pd.concat([df_users["date"], df_hosts["date"]], ignore_index=True)
    # Нормализуем к дате, отбрасываем NaN и сортируем.
    d = pd.to_datetime(d, errors="coerce").dropna().dt.date.astype("string").unique().tolist()
    return sorted(d)


def pick_latest_date(df_users: pd.DataFrame, df_hosts: pd.DataFrame) -> str:
    """Выбирает последнюю доступную дату."""
    d = available_dates(df_users, df_hosts)
    if not d:
        raise ValueError("No dates found in features tables.")
    return d[-1]


def minmax(a: np.ndarray) -> np.ndarray:
    """Min-max нормализация массива."""
    # Вычисляем диапазон значений.
    amin: float = float(np.min(a))
    amax: float = float(np.max(a))
    if not np.isfinite(amin) or not np.isfinite(amax) or amax == amin:
        return np.zeros_like(a, dtype=float)
    # Приводим к интервалу [0, 1].
    return (a - amin) / (amax - amin)


def severity_by_rank(n: int, ranks: np.ndarray) -> List[str]:
    """
    Процентильная серьёзность внутри дня:
      - critical: верхние 5%
      - high:     следующие 15% (<=20%)
      - medium:   следующие 30% (<=50%)
      - low:      остальные
    Ранг 1 — наиболее аномальный.
    """
    thr_crit: int = max(1, math.ceil(0.05 * n))
    thr_high: int = max(thr_crit, math.ceil(0.20 * n))
    thr_med: int = max(thr_high, math.ceil(0.50 * n))

    # Проставляем уровни по рангу.
    out: List[str] = []
    for r in ranks:
        if r <= thr_crit:
            out.append("critical")
        elif r <= thr_high:
            out.append("high")
        elif r <= thr_med:
            out.append("medium")
        else:
            out.append("low")
    return out


def fit_models(
    X_train: np.ndarray,
    contamination: float,
    n_estimators: int,
    n_neighbors: int,
    random_state: int,
) -> Dict[str, object]:
    """Обучает Isolation Forest и LOF (novelty) для заданной выборки."""
    # Скейлер для устойчивости к выбросам.
    scaler: RobustScaler = RobustScaler(with_centering=True, with_scaling=True, unit_variance=False)

    # Конфигурация Isolation Forest.
    iso: IsolationForest = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    lof: LocalOutlierFactor = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        metric="minkowski",
        p=2,
    )

    # Создаём пайплайны и обучаем.
    iso_pipe: Pipeline = Pipeline([("scaler", scaler), ("model", iso)])
    lof_pipe: Pipeline = Pipeline([("scaler", scaler), ("model", lof)])

    iso_pipe.fit(X_train)
    lof_pipe.fit(X_train)
    return {"iso": iso_pipe, "lof": lof_pipe}


def score_day(
    df: pd.DataFrame,
    day: str,
    contamination: float,
    n_estimators: int,
    n_neighbors: int,
    random_state: int,
) -> pd.DataFrame:
    """Считает скоринг аномалий для заданного дня."""
    # Собираем список признаков.
    feat_cols: list[str] = [c for c in df.columns if c not in ID_COLS]

    # Выделяем тестовую часть по заданной дате.
    test: pd.DataFrame = df[df["date"] == day].copy()
    if test.empty:
        raise ValueError(f"No rows for date={day}")

    # Обучение на остальных днях (или на всех, если других нет).
    train: pd.DataFrame = df[df["date"] != day].copy()
    if train.empty:
        train = df.copy()

    X_train: np.ndarray = train[feat_cols].to_numpy(dtype=float)
    X_test: np.ndarray = test[feat_cols].to_numpy(dtype=float)

    models = fit_models(X_train, contamination, n_estimators, n_neighbors, random_state)

    # decision_function: чем больше, тем "нормальнее" -> инвертируем.
    iso_s = -models["iso"].decision_function(X_test)
    lof_s = -models["lof"].decision_function(X_test)

    # Формируем таблицу результатов.
    out: pd.DataFrame = test[ID_COLS].copy()
    out["score_iso"] = iso_s
    out["score_lof"] = lof_s
    out["score_iso_norm"] = minmax(iso_s)
    out["score_lof_norm"] = minmax(lof_s)
    out["score_combined_norm"] = (out["score_iso_norm"] + out["score_lof_norm"]) / 2.0
    out["rank_combined"] = out["score_combined_norm"].rank(ascending=False, method="average")
    out["severity"] = severity_by_rank(len(out), out["rank_combined"].to_numpy())
    return out


def build_trend(
    df: pd.DataFrame,
    dates: List[str],
    contamination: float,
    n_estimators: int,
    n_neighbors: int,
    random_state: int,
) -> pd.DataFrame:
    """Строит тренд по датам: распределение по severity и топ сущности."""
    # Контейнер для агрегированных строк по дням.
    rows: list[dict[str, object]] = []
    for day in dates:
        # Считаем скоринг на конкретный день.
        s: pd.DataFrame = score_day(df, day, contamination, n_estimators, n_neighbors, random_state)
        # Считаем распределение по уровням серьёзности.
        counts: dict[str, int] = s["severity"].value_counts().to_dict()
        # Ищем самую аномальную сущность.
        top1: pd.DataFrame = s.sort_values("rank_combined").head(1)
        rows.append(
            {
                "date": day,
                "total": int(len(s)),
                "critical": int(counts.get("critical", 0)),
                "high": int(counts.get("high", 0)),
                "medium": int(counts.get("medium", 0)),
                "low": int(counts.get("low", 0)),
                "top_entity": top1["entity"].iloc[0] if not top1.empty else "",
                "top_score": float(
                    pd.to_numeric(top1["score_combined_norm"], errors="coerce").fillna(0).iloc[0]
                )
                if not top1.empty
                else 0.0,
            }
        )
    # Возвращаем DataFrame со строками по дням.
    return pd.DataFrame(rows)


# ---------- Хелперы для графиков (без явных цветов) ----------

def save_top_bar(out_dir: Path, scores: pd.DataFrame, title: str, filename: str, top: int) -> None:
    """Строит столбчатый график top-N аномалий."""
    # Сортируем по рангу и берём top-N.
    df: pd.DataFrame = scores.sort_values("rank_combined").head(top).copy()
    if df.empty:
        return
    # Строим график.
    plt.figure()
    plt.bar(
        df["entity"].astype(str),
        pd.to_numeric(df["score_combined_norm"], errors="coerce").fillna(0).to_numpy(),
    )
    plt.xticks(rotation=90)
    plt.ylabel("Combined anomaly score (0..1)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=160)
    plt.close()


def save_severity_pie(out_dir: Path, scores: pd.DataFrame, title: str, filename: str) -> None:
    """Строит pie-график распределения по severity."""
    # Считаем распределение по уровням серьёзности.
    counts: pd.Series = scores["severity"].value_counts()
    if counts.empty:
        return
    # Готовим подписи и цвета.
    labels: list[str] = [SEV_RU.get(k, k) for k in counts.index.tolist()]
    colors: list[str] = [SEV_COLORS.get(k, "#cccccc") for k in counts.index.tolist()]
    # Рисуем круговую диаграмму.
    plt.figure()
    plt.pie(counts.to_numpy(), labels=labels, autopct="%1.1f%%", colors=colors)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=160)
    plt.close()


def save_trend_line(out_dir: Path, trend: pd.DataFrame, title: str, filename: str, ycol: str) -> None:
    """Линейный график тренда по выбранной метрике."""
    # Проверяем, что данные доступны.
    if trend.empty or ycol not in trend.columns:
        return
    # Подготавливаем оси.
    plt.figure()
    x: pd.Series = pd.to_datetime(trend["date"], errors="coerce")
    y: pd.Series = pd.to_numeric(trend[ycol], errors="coerce").fillna(0)
    color = SEV_COLORS.get(ycol)
    # Рисуем линию с маркерами.
    if color:
        plt.plot(x, y, marker="o", color=color)
    else:
        plt.plot(x, y, marker="o")
    plt.ylabel(ycol)
    plt.title(title)
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=160)
    plt.close()


def save_trend_stacked(out_dir: Path, trend: pd.DataFrame, title: str, filename: str) -> None:
    """Stacked bar для распределения severity по дням."""
    # Проверяем, что есть данные.
    if trend.empty:
        return
    # Сортируем по дате и готовим подписи.
    df: pd.DataFrame = trend.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    x: list[str] = df["date"].dt.strftime("%Y-%m-%d").tolist()

    # Готовим значения по уровням серьёзности.
    sev_cols: list[str] = ["critical", "high", "medium", "low"]
    vals: list[np.ndarray] = [pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy() for c in sev_cols]

    # Рисуем stacked bar.
    plt.figure()
    bottom: np.ndarray = np.zeros(len(df))
    for c, v in zip(sev_cols, vals):
        plt.bar(x, v, bottom=bottom, label=SEV_RU.get(c, c), color=SEV_COLORS.get(c, "#cccccc"))
        bottom = bottom + v

    plt.xticks(rotation=45)
    plt.ylabel("Количество аномалий")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=160)
    plt.close()
