#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
visualize_reports.py (interactive)

При запуске показывает инструкцию, параметры и 3 примера,
после чего пользователь вводит значения параметров.

Также поддерживается неинтерактивный режим через CLI (для автоматизации/Docker):
  python visualize_reports.py --work .\\work --scope day --date 2025-12-31
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Optional, Tuple

import pandas as pd

import viz_core as core


def print_manual() -> None:
    print("\n=== Визуализация отчётов аномалий (SOC) ===\n")
    print("Скрипт строит графики на основе features_users_clean.csv и features_hosts_clean.csv.")
    print("Графики сохраняются в work/reports/<scope>_<date>/\n")
    print("Параметры:")
    print("  --work           Папка work (обязательно), например .\\work")
    print("  --scope          day | week | month")
    print("  --date           Дата YYYY-MM-DD (для week/month — конец окна). По умолчанию — последняя дата в данных.")
    print("  --top            TOP-N сущностей для графика за сутки (только scope=day). По умолчанию 20.")
    print("  --contamination  Доля аномалий (0..0.5], по умолчанию 0.05.")
    print("  --n-estimators   Кол-во деревьев Isolation Forest, по умолчанию 300.")
    print("  --n-neighbors    LOF n_neighbors, по умолчанию 20 (типично 10–30).")
    print("\nПримеры:")
    print(r"  1) Отчёт за сутки по последней дате:     python visualize_reports.py --work .\work --scope day")
    print(r"  2) Отчёт за неделю до 2025-12-31:        python visualize_reports.py --work .\work --scope week --date 2025-12-31")
    print(r"  3) Отчёт за месяц до последней даты:     python visualize_reports.py --work .\work --scope month")
    print("\nИнтерактивный режим: запустите без --scope, и скрипт спросит значения.\n")


def _prompt(msg: str, default: Optional[str] = None) -> str:
    if default is not None:
        v = input(f"{msg} [по умолчанию: {default}]: ").strip()
        return v if v else default
    return input(f"{msg}: ").strip()


def interactive_collect() -> Tuple[str, str, str, int, float, int, int, int]:
    work = _prompt("Введите путь к папке work", ".\\work")
    scope = _prompt("Выберите scope (day/week/month)", "day").lower()

    date = _prompt("Введите дату YYYY-MM-DD (пусто = последняя дата в данных)", "").strip()
    date = date if date else ""

    top = int(_prompt("TOP-N для отчёта за сутки (актуально только для scope=day)", "20"))
    contamination = float(_prompt("contamination (0..0.5]", "0.05"))
    n_estimators = int(_prompt("IsolationForest n_estimators", "300"))
    n_neighbors = int(_prompt("LOF n_neighbors", "20"))
    random_state = int(_prompt("random_state", "42"))

    return work, scope, date, top, contamination, n_estimators, n_neighbors, random_state


def run(work: Path,
        scope: str,
        date: Optional[str],
        top: int,
        contamination: float,
        n_estimators: int,
        n_neighbors: int,
        random_state: int) -> None:
    users = core.load_features(work, "users")
    hosts = core.load_features(work, "hosts")

    available = core.available_dates(users, hosts)
    if not available:
        raise ValueError("Не найдены даты в features_*_clean.csv")

    target = date if date else available[-1]
    target = str(pd.to_datetime(target, errors="raise").date())

    out_base = core.ensure_dir(work / "reports")

    if scope == "day":
        out = core.ensure_dir(out_base / f"day_{target}")
        su = core.score_day(users, target, contamination, n_estimators, n_neighbors, random_state)
        sh = core.score_day(hosts, target, contamination, n_estimators, n_neighbors, random_state)

        su.to_csv(out / f"day_scores_users_{target}.csv", index=False)
        sh.to_csv(out / f"day_scores_hosts_{target}.csv", index=False)

        core.save_top_bar(out, su, f"TOP {top} аномалий пользователей за {target}", f"top_users_{target}.png", top)
        core.save_top_bar(out, sh, f"TOP {top} аномалий хостов за {target}", f"top_hosts_{target}.png", top)
        core.save_severity_pie(out, su, f"Серьёзность (пользователи) {target}", f"severity_users_{target}.png")
        core.save_severity_pie(out, sh, f"Серьёзность (хосты) {target}", f"severity_hosts_{target}.png")

        print(f"[+] Готово. Папка отчёта: {out}")
        return

    window = 7 if scope == "week" else 30
    if target not in available:
        raise ValueError(f"Дата {target} отсутствует в данных. Последняя дата: {available[-1]}")

    idx = available.index(target)
    start = max(0, idx - (window - 1))
    dates = available[start:idx + 1]

    out = core.ensure_dir(out_base / f"{scope}_{target}")

    tu = core.build_trend(users, dates, contamination, n_estimators, n_neighbors, random_state)
    th = core.build_trend(hosts, dates, contamination, n_estimators, n_neighbors, random_state)

    tu.to_csv(out / f"trend_users_{scope}_{target}.csv", index=False)
    th.to_csv(out / f"trend_hosts_{scope}_{target}.csv", index=False)

    core.save_trend_line(out, tu, f"Динамика TOTAL (пользователи) - {scope} до {target}", f"trend_users_total_{scope}_{target}.png", "total")
    core.save_trend_line(out, th, f"Динамика TOTAL (хосты) - {scope} до {target}", f"trend_hosts_total_{scope}_{target}.png", "total")

    core.save_trend_line(out, tu, f"Динамика CRITICAL (пользователи) - {scope} до {target}", f"trend_users_critical_{scope}_{target}.png", "critical")
    core.save_trend_line(out, th, f"Динамика CRITICAL (хосты) - {scope} до {target}", f"trend_hosts_critical_{scope}_{target}.png", "critical")

    core.save_trend_stacked(out, tu, f"Серьёзность по дням (пользователи) - {scope}", f"trend_users_severity_{scope}_{target}.png")
    core.save_trend_stacked(out, th, f"Серьёзность по дням (хосты) - {scope}", f"trend_hosts_severity_{scope}_{target}.png")

    print(f"[+] Готово. Папка отчёта: {out}")


def main() -> int:
    p = argparse.ArgumentParser(add_help=True)
    p.add_argument("--work", default=None, help="Work directory (e.g., .\\work)")
    p.add_argument("--scope", default=None, choices=["day", "week", "month"], help="Report scope")
    p.add_argument("--date", default=None, help="Target date YYYY-MM-DD (optional)")
    p.add_argument("--top", type=int, default=20, help="TOP-N for day report")
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--n-neighbors", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    print_manual()

    if args.scope is None:
        work_s, scope, date_s, top, cont, n_est, n_nei, rs = interactive_collect()
        work = Path(work_s)
        date = date_s if date_s else None
        run(work, scope, date, top, cont, n_est, n_nei, rs)
        return 0

    if not args.work:
        raise ValueError("--work is required in non-interactive mode (when --scope is provided)")

    run(Path(args.work), args.scope, args.date, args.top,
        args.contamination, args.n_estimators, args.n_neighbors, args.random_state)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
