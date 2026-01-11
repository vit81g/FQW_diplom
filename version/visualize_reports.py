#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import math
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.ensemble import IsolationForest
from sklearn.neighbors import LocalOutlierFactor
from sklearn.preprocessing import RobustScaler
from sklearn.pipeline import Pipeline

ID_COLS = ["entity", "date"]
SEV_RU = {"critical": "Критично", "high": "Высокий", "medium": "Средний", "low": "Низкий"}


def load_features(work_dir: Path, kind: str) -> pd.DataFrame:
    p = work_dir / f"features_{kind}_clean.csv"
    if not p.exists():
        raise FileNotFoundError(f"Missing {p}. Run build_features_v2.py + preprocess_features.py first.")
    df = pd.read_csv(p, dtype=str, keep_default_na=True)
    for c in ID_COLS:
        if c not in df.columns:
            raise ValueError(f"{p.name}: missing required column {c}")
    df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")

    feat_cols = [c for c in df.columns if c not in ID_COLS]
    for c in feat_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[feat_cols] = df[feat_cols].fillna(0)
    return df


def available_dates(df_users: pd.DataFrame, df_hosts: pd.DataFrame) -> List[str]:
    d = pd.concat([df_users["date"], df_hosts["date"]], ignore_index=True)
    d = pd.to_datetime(d, errors="coerce").dropna().dt.date.astype("string").unique().tolist()
    return sorted(d)


def minmax(a: np.ndarray) -> np.ndarray:
    amin = float(np.min(a))
    amax = float(np.max(a))
    if not np.isfinite(amin) or not np.isfinite(amax) or amax == amin:
        return np.zeros_like(a, dtype=float)
    return (a - amin) / (amax - amin)


def severity_by_rank(n: int, ranks: np.ndarray) -> List[str]:
    thr_crit = max(1, math.ceil(0.05 * n))
    thr_high = max(thr_crit, math.ceil(0.20 * n))
    thr_med = max(thr_high, math.ceil(0.50 * n))
    out = []
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


def fit_models(X_train: np.ndarray, contamination: float, n_estimators: int, n_neighbors: int, random_state: int) -> Dict[str, object]:
    scaler = RobustScaler(with_centering=True, with_scaling=True, unit_variance=False)

    iso = IsolationForest(
        n_estimators=n_estimators,
        contamination=contamination,
        max_samples="auto",
        random_state=random_state,
        n_jobs=-1,
    )
    lof = LocalOutlierFactor(
        n_neighbors=n_neighbors,
        contamination=contamination,
        novelty=True,
        metric="minkowski",
        p=2,
    )

    iso_pipe = Pipeline([("scaler", scaler), ("model", iso)])
    lof_pipe = Pipeline([("scaler", scaler), ("model", lof)])

    iso_pipe.fit(X_train)
    lof_pipe.fit(X_train)
    return {"iso": iso_pipe, "lof": lof_pipe}


def score_day(df: pd.DataFrame, day: str, contamination: float, n_estimators: int, n_neighbors: int, random_state: int) -> pd.DataFrame:
    feat_cols = [c for c in df.columns if c not in ID_COLS]
    test = df[df["date"] == day].copy()
    if test.empty:
        raise ValueError(f"No rows for date={day}")

    train = df[df["date"] != day].copy()
    if train.empty:
        train = df.copy()

    X_train = train[feat_cols].to_numpy(dtype=float)
    X_test = test[feat_cols].to_numpy(dtype=float)

    models = fit_models(X_train, contamination, n_estimators, n_neighbors, random_state)

    # decision_function: higher = more normal -> invert to anomaly score
    iso_s = -models["iso"].decision_function(X_test)
    lof_s = -models["lof"].decision_function(X_test)

    out = test[ID_COLS].copy()
    out["score_iso"] = iso_s
    out["score_lof"] = lof_s
    out["score_iso_norm"] = minmax(iso_s)
    out["score_lof_norm"] = minmax(lof_s)
    out["score_combined_norm"] = (out["score_iso_norm"] + out["score_lof_norm"]) / 2.0
    out["rank_combined"] = out["score_combined_norm"].rank(ascending=False, method="average")
    out["severity"] = severity_by_rank(len(out), out["rank_combined"].to_numpy())
    return out


def ensure_dir(p: Path) -> Path:
    p.mkdir(parents=True, exist_ok=True)
    return p


def save_top_bar(out_dir: Path, scores: pd.DataFrame, title: str, filename: str, top: int) -> None:
    df = scores.sort_values("rank_combined").head(top).copy()
    if df.empty:
        return
    plt.figure()
    plt.bar(df["entity"].astype(str), pd.to_numeric(df["score_combined_norm"], errors="coerce").fillna(0).to_numpy())
    plt.xticks(rotation=90)
    plt.ylabel("Combined anomaly score (0..1)")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=160)
    plt.close()


def save_severity_pie(out_dir: Path, scores: pd.DataFrame, title: str, filename: str) -> None:
    counts = scores["severity"].value_counts()
    if counts.empty:
        return
    labels = [SEV_RU.get(k, k) for k in counts.index.tolist()]
    plt.figure()
    plt.pie(counts.to_numpy(), labels=labels, autopct="%1.1f%%")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=160)
    plt.close()


def save_trend_stacked(out_dir: Path, trend: pd.DataFrame, title: str, filename: str) -> None:
    if trend.empty:
        return
    df = trend.copy()
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.sort_values("date")
    x = df["date"].dt.strftime("%Y-%m-%d").tolist()

    sev_cols = ["critical", "high", "medium", "low"]
    vals = [pd.to_numeric(df[c], errors="coerce").fillna(0).to_numpy() for c in sev_cols]

    plt.figure()
    bottom = np.zeros(len(df))
    for c, v in zip(sev_cols, vals):
        plt.bar(x, v, bottom=bottom, label=SEV_RU.get(c, c))
        bottom = bottom + v

    plt.xticks(rotation=45)
    plt.ylabel("Количество аномалий")
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_dir / filename, dpi=160)
    plt.close()


def build_trend(df: pd.DataFrame, dates: List[str], contamination: float, n_estimators: int, n_neighbors: int, random_state: int) -> pd.DataFrame:
    rows = []
    for day in dates:
        s = score_day(df, day, contamination, n_estimators, n_neighbors, random_state)
        counts = s["severity"].value_counts().to_dict()
        top1 = s.sort_values("rank_combined").head(1)
        rows.append({
            "date": day,
            "total": int(len(s)),
            "critical": int(counts.get("critical", 0)),
            "high": int(counts.get("high", 0)),
            "medium": int(counts.get("medium", 0)),
            "low": int(counts.get("low", 0)),
            "top_entity": top1["entity"].iloc[0] if not top1.empty else "",
            "top_score": float(pd.to_numeric(top1["score_combined_norm"], errors="coerce").fillna(0).iloc[0]) if not top1.empty else 0.0,
        })
    return pd.DataFrame(rows)


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True)
    p.add_argument("--scope", choices=["day", "week", "month"], required=True)
    p.add_argument("--date", default=None, help="Target/end date YYYY-MM-DD (default: latest from features)")
    p.add_argument("--top", type=int, default=20)
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--n-neighbors", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    work = Path(args.work)
    users = load_features(work, "users")
    hosts = load_features(work, "hosts")

    all_dates = available_dates(users, hosts)
    if not all_dates:
        raise ValueError("No dates found in features_*_clean.csv")

    target = args.date or all_dates[-1]
    target = str(pd.to_datetime(target, errors="raise").date())

    out_base = ensure_dir(work / "reports")

    if args.scope == "day":
        out = ensure_dir(out_base / f"day_{target}")
        su = score_day(users, target, args.contamination, args.n_estimators, args.n_neighbors, args.random_state)
        sh = score_day(hosts, target, args.contamination, args.n_estimators, args.n_neighbors, args.random_state)

        su.to_csv(out / f"day_scores_users_{target}.csv", index=False)
        sh.to_csv(out / f"day_scores_hosts_{target}.csv", index=False)

        save_top_bar(out, su, f"TOP {args.top} аномалий пользователей за {target}", f"top_users_{target}.png", args.top)
        save_top_bar(out, sh, f"TOP {args.top} аномалий хостов за {target}", f"top_hosts_{target}.png", args.top)
        save_severity_pie(out, su, f"Серьёзность (пользователи) {target}", f"severity_users_{target}.png")
        save_severity_pie(out, sh, f"Серьёзность (хосты) {target}", f"severity_hosts_{target}.png")

        print(f"[+] Saved to: {out}")
        return 0

    window = 7 if args.scope == "week" else 30
    if target not in all_dates:
        raise ValueError(f"date={target} not present in features dates")

    idx = all_dates.index(target)
    start = max(0, idx - (window - 1))
    dates = all_dates[start:idx + 1]

    out = ensure_dir(out_base / f"{args.scope}_{target}")

    tu = build_trend(users, dates, args.contamination, args.n_estimators, args.n_neighbors, args.random_state)
    th = build_trend(hosts, dates, args.contamination, args.n_estimators, args.n_neighbors, args.random_state)

    tu.to_csv(out / f"trend_users_{args.scope}_{target}.csv", index=False)
    th.to_csv(out / f"trend_hosts_{args.scope}_{target}.csv", index=False)

    save_trend_stacked(out, tu, f"Серьёзность по дням (пользователи) - {args.scope}", f"trend_users_severity_{args.scope}_{target}.png")
    save_trend_stacked(out, th, f"Серьёзность по дням (хосты) - {args.scope}", f"trend_hosts_severity_{args.scope}_{target}.png")

    print(f"[+] Saved to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
