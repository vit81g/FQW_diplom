#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_generate_reports.py

Генерация всех отчётов одним запуском:
- day
- week
- month

Результаты сохраняются в: report/<scope>_<date>/ (по умолчанию рядом с папкой features).
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import pandas as pd

import viz_core as core


def _run_one(
    work: Path,
    report_dir: Path,
    scope: str,
    target: str,
    top_pct: float,
    contamination: float,
    n_estimators: int,
    n_neighbors: int,
    random_state: int,
) -> Path:
    """Строит отчёт для указанного масштаба (day/week/month)."""
    # Загружаем подготовленные признаки пользователей и хостов.
    users: pd.DataFrame = core.load_features(work, "users")
    hosts: pd.DataFrame = core.load_features(work, "hosts")
    # Определяем доступные даты в данных.
    available: list[str] = core.available_dates(users, hosts)
    if target not in available:
        # Сообщаем об ошибке, если дата отсутствует.
        raise ValueError(f"Date {target} not present. Last available: {available[-1]}")

    # Готовим папку назначения для отчётов.
    out_base: Path = core.ensure_dir(report_dir)

    if scope == "day":
        # Для дневного отчёта создаём отдельную папку.
        out = core.ensure_dir(out_base / f"day_{target}")
        # Рассчитываем скоринг аномалий для пользователей и хостов.
        su = core.score_day(users, target, contamination, n_estimators, n_neighbors, random_state)
        sh = core.score_day(hosts, target, contamination, n_estimators, n_neighbors, random_state)

        # Сохраняем таблицы скоринга в CSV.
        su.to_csv(out / f"day_scores_users_{target}.csv", index=False)
        sh.to_csv(out / f"day_scores_hosts_{target}.csv", index=False)

        # Вычисляем количество топовых аномалий.
        top_users = max(1, math.ceil(top_pct * len(su)))
        top_hosts = max(1, math.ceil(top_pct * len(sh)))
        pct_label = int(top_pct * 100)
        # Строим бар-чарты по топовым аномалиям.
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
        # Сохраняем круговые диаграммы по уровню серьёзности.
        core.save_severity_pie(out, su, f"Users severity {target}", f"severity_users_{target}.png")
        core.save_severity_pie(out, sh, f"Hosts severity {target}", f"severity_hosts_{target}.png")
        return out

    # Для недели/месяца вычисляем окно дат относительно цели.
    window: int = 7 if scope == "week" else 30
    idx: int = available.index(target)
    start: int = max(0, idx - (window - 1))
    dates: list[str] = available[start:idx + 1]

    # Создаём папку под результат выбранного масштаба.
    out = core.ensure_dir(out_base / f"{scope}_{target}")

    # Строим тренды по выбранному диапазону.
    tu: pd.DataFrame = core.build_trend(users, dates, contamination, n_estimators, n_neighbors, random_state)
    th: pd.DataFrame = core.build_trend(hosts, dates, contamination, n_estimators, n_neighbors, random_state)

    # Сохраняем тренд-таблицы.
    tu.to_csv(out / f"trend_users_{scope}_{target}.csv", index=False)
    th.to_csv(out / f"trend_hosts_{scope}_{target}.csv", index=False)

    # Строим линейные тренды по суммарной аномальности.
    core.save_trend_line(out, tu, f"Users TOTAL trend ({scope}) to {target}", f"trend_users_total_{scope}_{target}.png", "total")
    core.save_trend_line(out, th, f"Hosts TOTAL trend ({scope}) to {target}", f"trend_hosts_total_{scope}_{target}.png", "total")

    # Строим линейные тренды по критичным аномалиям.
    core.save_trend_line(out, tu, f"Users CRITICAL trend ({scope}) to {target}", f"trend_users_critical_{scope}_{target}.png", "critical")
    core.save_trend_line(out, th, f"Hosts CRITICAL trend ({scope}) to {target}", f"trend_hosts_critical_{scope}_{target}.png", "critical")

    # Строим стековые диаграммы по серьёзности.
    core.save_trend_stacked(out, tu, f"Users severity per day ({scope})", f"trend_users_severity_{scope}_{target}.png")
    core.save_trend_stacked(out, th, f"Hosts severity per day ({scope})", f"trend_hosts_severity_{scope}_{target}.png")
    return out


def main() -> int:
    # Настраиваем аргументы командной строки.
    p = argparse.ArgumentParser()
    p.add_argument("--work", default="features", help="Features directory (e.g., .\\features)")
    p.add_argument("--report-dir", default="report", help="Output folder for reports (default: report)")
    p.add_argument("--date", default=None, help="Target/end date YYYY-MM-DD (default: latest)")
    p.add_argument("--top-pct", type=float, default=0.05, help="Top share for users/hosts (e.g., 0.05 = 5%)")
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--n-neighbors", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)
    # Разбираем аргументы из CLI.
    args = p.parse_args()

    # Определяем рабочие каталоги.
    work: Path = Path(args.work)
    report_dir: Path = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = work.parent / report_dir
    # Загружаем таблицы признаков.
    users: pd.DataFrame = core.load_features(work, "users")
    hosts: pd.DataFrame = core.load_features(work, "hosts")

    # Определяем целевую дату (по умолчанию — последняя доступная).
    target: str = args.date or core.pick_latest_date(users, hosts)
    target = str(pd.to_datetime(target, errors="raise").date())

    # Запускаем отчёты по дням/неделям/месяцам.
    out_day = _run_one(
        work,
        report_dir,
        "day",
        target,
        args.top_pct,
        args.contamination,
        args.n_estimators,
        args.n_neighbors,
        args.random_state,
    )
    # Выводим путь к дневному отчёту.
    print(f"[+] day: {out_day}")
    out_week = _run_one(
        work,
        report_dir,
        "week",
        target,
        args.top_pct,
        args.contamination,
        args.n_estimators,
        args.n_neighbors,
        args.random_state,
    )
    # Выводим путь к недельному отчёту.
    print(f"[+] week: {out_week}")
    out_month = _run_one(
        work,
        report_dir,
        "month",
        target,
        args.top_pct,
        args.contamination,
        args.n_estimators,
        args.n_neighbors,
        args.random_state,
    )
    # Выводим путь к месячному отчёту.
    print(f"[+] month: {out_month}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
