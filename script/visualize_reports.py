#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_reports.py

CLI-инструмент для генерации графиков аномалий (day/week/month).
Графики сохраняются в: report/<scope>_<date>/ (по умолчанию рядом с папкой features).

Примеры:
  python visualize_reports.py --work .\\features --scope day --top-pct 0.05
  python visualize_reports.py --work .\\features --scope week --date 2025-12-31
  python visualize_reports.py --work .\\features --scope month
  python visualize_reports.py --work .\\features --report-dir .\\report
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Optional

import pandas as pd

import viz_core as core


def run(
    work: Path,
    report_dir: Path,
    scope: str,
    date: Optional[str],
    top_pct: float,
    contamination: float,
    n_estimators: int,
    n_neighbors: int,
    random_state: int,
) -> Path:
    """Строит отчёт заданного масштаба и сохраняет графики."""
    # Загружаем подготовленные признаки.
    users: pd.DataFrame = core.load_features(work, "users")
    hosts: pd.DataFrame = core.load_features(work, "hosts")

    # Получаем список доступных дат.
    available: list[str] = core.available_dates(users, hosts)
    if not available:
        raise ValueError("No dates found in features files.")

    # Выбираем целевую дату (по умолчанию — последняя).
    target: str = date if date else available[-1]
    target = str(pd.to_datetime(target, errors="raise").date())

    # Создаём базовую директорию отчёта.
    out_base: Path = core.ensure_dir(report_dir)

    if scope == "day":
        # Дневной отчёт: отдельная папка.
        out = core.ensure_dir(out_base / f"day_{target}")
        # Рассчитываем дневной скоринг.
        su = core.score_day(users, target, contamination, n_estimators, n_neighbors, random_state)
        sh = core.score_day(hosts, target, contamination, n_estimators, n_neighbors, random_state)

        # Сохраняем CSV со скорингом.
        su.to_csv(out / f"day_scores_users_{target}.csv", index=False)
        sh.to_csv(out / f"day_scores_hosts_{target}.csv", index=False)

        # Определяем число топовых аномалий.
        top_users = max(1, math.ceil(top_pct * len(su)))
        top_hosts = max(1, math.ceil(top_pct * len(sh)))
        pct_label = int(top_pct * 100)
        # Строим бар-чарты по топ-аномалиям.
        core.save_top_bar(
            out,
            su,
            f"TOP {pct_label}% users anomalies for {target} (n={top_users})",
            f"top_users_{target}.png",
            top_users,
        )
        core.save_top_bar(
            out,
            sh,
            f"TOP {pct_label}% hosts anomalies for {target} (n={top_hosts})",
            f"top_hosts_{target}.png",
            top_hosts,
        )
        # Строим круговые диаграммы по серьёзности.
        core.save_severity_pie(out, su, f"Users severity {target}", f"severity_users_{target}.png")
        core.save_severity_pie(out, sh, f"Hosts severity {target}", f"severity_hosts_{target}.png")
        return out

    # Для недели/месяца задаём ширину окна.
    window: int = 7 if scope == "week" else 30
    if target not in available:
        raise ValueError(f"Date {target} is not present in data. Last available: {available[-1]}")

    # Вычисляем окно дат относительно целевой даты.
    idx: int = available.index(target)
    start: int = max(0, idx - (window - 1))
    dates: list[str] = available[start:idx + 1]

    # Создаём папку под тренды.
    out = core.ensure_dir(out_base / f"{scope}_{target}")

    # Строим тренды аномалий.
    tu: pd.DataFrame = core.build_trend(users, dates, contamination, n_estimators, n_neighbors, random_state)
    th: pd.DataFrame = core.build_trend(hosts, dates, contamination, n_estimators, n_neighbors, random_state)

    # Сохраняем трендовые таблицы.
    tu.to_csv(out / f"trend_users_{scope}_{target}.csv", index=False)
    th.to_csv(out / f"trend_hosts_{scope}_{target}.csv", index=False)

    # Строим линейные графики суммарной аномальности.
    core.save_trend_line(out, tu, f"Users TOTAL trend ({scope}) to {target}", f"trend_users_total_{scope}_{target}.png", "total")
    core.save_trend_line(out, th, f"Hosts TOTAL trend ({scope}) to {target}", f"trend_hosts_total_{scope}_{target}.png", "total")

    # Строим линейные графики критичных аномалий.
    core.save_trend_line(out, tu, f"Users CRITICAL trend ({scope}) to {target}", f"trend_users_critical_{scope}_{target}.png", "critical")
    core.save_trend_line(out, th, f"Hosts CRITICAL trend ({scope}) to {target}", f"trend_hosts_critical_{scope}_{target}.png", "critical")

    # Строим диаграммы распределения серьёзности.
    core.save_trend_stacked(out, tu, f"Users severity per day ({scope})", f"trend_users_severity_{scope}_{target}.png")
    core.save_trend_stacked(out, th, f"Hosts severity per day ({scope})", f"trend_hosts_severity_{scope}_{target}.png")
    return out


def main() -> int:
    # Описываем аргументы командной строки.
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="features", help="Features directory (e.g., .\\features)")
    p.add_argument("--report-dir", default="report", help="Output folder for reports (default: report)")
    p.add_argument("--scope", required=True, choices=["day", "week", "month"])
    p.add_argument("--date", default=None, help="Target date YYYY-MM-DD (optional)")
    p.add_argument("--top-pct", type=float, default=0.05, help="Top share for users/hosts (e.g., 0.05 = 5%)")
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--n-neighbors", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)
    # Парсим аргументы.
    args = p.parse_args()

    # Готовим директории.
    work_dir = Path(args.work)
    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = work_dir.parent / report_dir

    # Запускаем генерацию отчёта.
    out: Path = run(
        work_dir,
        report_dir,
        args.scope,
        args.date,
        args.top_pct,
        args.contamination,
        args.n_estimators,
        args.n_neighbors,
        args.random_state,
    )
    # Печатаем путь к результатам.
    print(f"[+] Saved to: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
