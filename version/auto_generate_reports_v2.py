#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
auto_generate_reports.py

Generate all reports in one run:
- day
- week
- month

Outputs to: work/reports/<scope>_<date>/
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

import viz_core as core


def _run_one(work: Path,
             scope: str,
             target: str,
             top: int,
             contamination: float,
             n_estimators: int,
             n_neighbors: int,
             random_state: int) -> Path:
    users = core.load_features(work, "users")
    hosts = core.load_features(work, "hosts")
    available = core.available_dates(users, hosts)
    if target not in available:
        raise ValueError(f"Date {target} not present. Last available: {available[-1]}")

    out_base = core.ensure_dir(work / "reports")

    if scope == "day":
        out = core.ensure_dir(out_base / f"day_{target}")
        su = core.score_day(users, target, contamination, n_estimators, n_neighbors, random_state)
        sh = core.score_day(hosts, target, contamination, n_estimators, n_neighbors, random_state)

        su.to_csv(out / f"day_scores_users_{target}.csv", index=False)
        sh.to_csv(out / f"day_scores_hosts_{target}.csv", index=False)

        core.save_top_bar(out, su, f"TOP {top} users anomalies for {target}", f"top_users_{target}.png", top)
        core.save_top_bar(out, sh, f"TOP {top} hosts anomalies for {target}", f"top_hosts_{target}.png", top)
        core.save_severity_pie(out, su, f"Users severity {target}", f"severity_users_{target}.png")
        core.save_severity_pie(out, sh, f"Hosts severity {target}", f"severity_hosts_{target}.png")
        return out

    window = 7 if scope == "week" else 30
    idx = available.index(target)
    start = max(0, idx - (window - 1))
    dates = available[start:idx + 1]

    out = core.ensure_dir(out_base / f"{scope}_{target}")

    tu = core.build_trend(users, dates, contamination, n_estimators, n_neighbors, random_state)
    th = core.build_trend(hosts, dates, contamination, n_estimators, n_neighbors, random_state)

    tu.to_csv(out / f"trend_users_{scope}_{target}.csv", index=False)
    th.to_csv(out / f"trend_hosts_{scope}_{target}.csv", index=False)

    core.save_trend_line(out, tu, f"Users TOTAL trend ({scope}) to {target}", f"trend_users_total_{scope}_{target}.png", "total")
    core.save_trend_line(out, th, f"Hosts TOTAL trend ({scope}) to {target}", f"trend_hosts_total_{scope}_{target}.png", "total")

    core.save_trend_line(out, tu, f"Users CRITICAL trend ({scope}) to {target}", f"trend_users_critical_{scope}_{target}.png", "critical")
    core.save_trend_line(out, th, f"Hosts CRITICAL trend ({scope}) to {target}", f"trend_hosts_critical_{scope}_{target}.png", "critical")

    core.save_trend_stacked(out, tu, f"Users severity per day ({scope})", f"trend_users_severity_{scope}_{target}.png")
    core.save_trend_stacked(out, th, f"Hosts severity per day ({scope})", f"trend_hosts_severity_{scope}_{target}.png")
    return out


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, help="Work directory (e.g., .\\work)")
    p.add_argument("--date", default=None, help="Target/end date YYYY-MM-DD (default: latest)")
    p.add_argument("--top", type=int, default=20)
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

    out_day = _run_one(work, "day", target, args.top, args.contamination, args.n_estimators, args.n_neighbors, args.random_state)
    print(f"[+] day: {out_day}")
    out_week = _run_one(work, "week", target, args.top, args.contamination, args.n_estimators, args.n_neighbors, args.random_state)
    print(f"[+] week: {out_week}")
    out_month = _run_one(work, "month", target, args.top, args.contamination, args.n_estimators, args.n_neighbors, args.random_state)
    print(f"[+] month: {out_month}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
