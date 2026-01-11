#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
pipeline_runner.py

One-command runner for the full offline SOC pipeline:
  normalize → build_features → preprocess → train/score → explain → soc_report → visualize(day/week/month)

This script orchestrates existing project scripts via subprocess to avoid tight coupling.

Expected scripts in the same folder (or provide --scripts-dir):
  - python_script.py
  - build_features_v2.py
  - preprocess_features.py
  - train_anomaly_models.py
  - explain_anomalies.py
  - soc_report_anomalies.py
  - auto_generate_reports.py

Usage examples:
  python pipeline_runner.py --data .\data --work .\work
  python pipeline_runner.py --data .\data --work .\work --date 2025-12-31
  python pipeline_runner.py --data .\data --work .\work --skip-normalize
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path
from typing import List, Optional


def _run(cmd: List[str], cwd: Optional[Path] = None) -> None:
    print("[*] RUN:", " ".join(cmd))
    r = subprocess.run(cmd, cwd=str(cwd) if cwd else None)
    if r.returncode != 0:
        raise SystemExit(r.returncode)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", required=True, help="Input folder with SIEM exports (*.tsv/*.txt)")
    ap.add_argument("--work", required=True, help="Work folder (outputs)")
    ap.add_argument("--date", default=None, help="Target/end date YYYY-MM-DD (optional, default: latest available)")
    ap.add_argument("--top", type=int, default=30, help="TOP anomalies (train/explain/report)")
    ap.add_argument("--top-charts", type=int, default=20, help="TOP for day bar charts (visualize)")
    ap.add_argument("--contamination", type=float, default=0.05)
    ap.add_argument("--n-estimators", type=int, default=300)
    ap.add_argument("--n-neighbors", type=int, default=20)
    ap.add_argument("--random-state", type=int, default=42)
    ap.add_argument("--chunksize", type=int, default=200_000)

    ap.add_argument("--scripts-dir", default=None, help="Directory where scripts are located (default: this file dir)")

    ap.add_argument("--skip-normalize", action="store_true", help="Skip python_script.py")
    ap.add_argument("--skip-features", action="store_true", help="Skip build_features_v2.py")
    ap.add_argument("--skip-preprocess", action="store_true", help="Skip preprocess_features.py")
    ap.add_argument("--skip-train", action="store_true", help="Skip train_anomaly_models.py")
    ap.add_argument("--skip-explain", action="store_true", help="Skip explain_anomalies.py")
    ap.add_argument("--skip-soc-report", action="store_true", help="Skip soc_report_anomalies.py")
    ap.add_argument("--skip-visualize", action="store_true", help="Skip auto_generate_reports.py")

    args = ap.parse_args()

    data_dir = Path(args.data)
    work_dir = Path(args.work)
    if not data_dir.exists():
        raise FileNotFoundError(f"Data dir not found: {data_dir}")
    work_dir.mkdir(parents=True, exist_ok=True)

    scripts_dir = Path(args.scripts_dir) if args.scripts_dir else Path(__file__).resolve().parent

    py = sys.executable

    # 1) normalize
    if not args.skip_normalize:
        _run(
            [py, str(scripts_dir / "python_script.py"), "--data", str(data_dir), "--work", str(work_dir), "--chunksize", str(args.chunksize)],
            cwd=scripts_dir,
        )

    # 2) build features
    if not args.skip_features:
        _run([py, str(scripts_dir / "build_features_v2.py"), "--work", str(work_dir)], cwd=scripts_dir)

    # 3) preprocess
    if not args.skip_preprocess:
        _run([py, str(scripts_dir / "preprocess_features.py"), "--work", str(work_dir)], cwd=scripts_dir)

    # 4) train/score
    if not args.skip_train:
        cmd = [
            py, str(scripts_dir / "train_anomaly_models.py"),
            "--work", str(work_dir),
            "--top", str(args.top),
            "--contamination", str(args.contamination),
            "--n-estimators", str(args.n_estimators),
            "--n-neighbors", str(args.n_neighbors),
            "--random-state", str(args.random_state),
        ]
        if args.date:
            cmd += ["--date", args.date]
        _run(cmd, cwd=scripts_dir)

    # 5) explain
    if not args.skip_explain:
        cmd = [py, str(scripts_dir / "explain_anomalies.py"), "--work", str(work_dir), "--top-features", "5"]
        if args.date:
            cmd += ["--date", args.date]
        _run(cmd, cwd=scripts_dir)

    # 6) SOC report (L1)
    if not args.skip_soc_report:
        cmd = [py, str(scripts_dir / "soc_report_anomalies.py"), "--work", str(work_dir), "--top", str(args.top)]
        if args.date:
            cmd += ["--date", args.date]
        _run(cmd, cwd=scripts_dir)

    # 7) visualize day/week/month
    if not args.skip_visualize:
        cmd = [
            py, str(scripts_dir / "auto_generate_reports.py"),
            "--work", str(work_dir),
            "--top", str(args.top_charts),
            "--contamination", str(args.contamination),
            "--n-estimators", str(args.n_estimators),
            "--n-neighbors", str(args.n_neighbors),
            "--random-state", str(args.random_state),
        ]
        if args.date:
            cmd += ["--date", args.date]
        _run(cmd, cwd=scripts_dir)

    print("[+] Pipeline finished OK.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
