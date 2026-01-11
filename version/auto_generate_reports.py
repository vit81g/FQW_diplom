#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_generate_reports.py

Автоматически формирует отчёты:
- сутки (day)
- неделя (week)
- месяц (month)

Предназначено для финального "мастер-скрипта" или Docker.

Usage:
  python auto_generate_reports.py --work .\work
  python auto_generate_reports.py --work .\work --date 2025-12-31 --top 20
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import viz_core as core


def run_scope(work: Path,
              scope: str,
              target: str,
              top: int,
              contamination: float,
              n_estimators: int,
              n_neighbors: int,
              random_state: int) -> None:
    users = core.load_features(work, "users")
    hosts = core.load_features(work, "hosts")
    available = core.available_dates(users, hosts)
    if target not in available:
        raise ValueError(f"Дата {target} отсутствует в данных. Последняя дата: {available[-1]}")

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
        print(f"[+] day: {out}")
        return

    window = 7 if scope == "week" else 30
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
    print(f"[+] {scope}: {out}")


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, help="Work directory, e.g., .\\work")
    p.add_argument("--date", default=None, help="Target/end date YYYY-MM-DD (default: latest date in features)")
    p.add_argument("--top", type=int, default=20, help="TOP-N for day charts")
    p.add_argument("--contamination", type=float, default=0.05)
    p.add_argument("--n-estimators", type=int, default=300)
    p.add_argument("--n-neighbors", type=int, default=20)
    p.add_argument("--random-state", type=int, default=42)
    args = p.parse_args()

    work = Path(args.work)

    users = core.load_features(work, "users")
    hosts = core.load_features(work, "hosts")
    target = args.date or core.pick_latest_date(users, hosts)
    target = str(pd.to_datetime(target, errors="raise").date())

    run_scope(work, "day", target, args.top, args.contamination, args.n_estimators, args.n_neighbors, args.random_state)
    run_scope(work, "week", target, args.top, args.contamination, args.n_estimators, args.n_neighbors, args.random_state)
    run_scope(work, "month", target, args.top, args.contamination, args.n_estimators, args.n_neighbors, args.random_state)

    print("[*] Done. All reports generated.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
