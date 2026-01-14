#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
soc_report.py

Формирование SOC-отчётов с контекстом по аномалиям.

Отчёт включает:
- описание аномалии (top_contributors/признаки),
- информацию о программах, пользователях, хостах и времени,
  которые связаны с аномальной активностью.

Вход:
  - anomalies_*_YYYY-MM-DD_explain.csv (в папке --features-dir или --anomaly-dir)
  - суточные CSV события в --work (userXXX_SIEM_YYYY-MM-DD.csv, host_SIEM_YYYY-MM-DD.csv, *_PAN_YYYY-MM-DD.csv)
  - (опционально) user_mapping.csv для расшифровки userXXX

Выход:
  - SOC-отчёт в Markdown (в папке --report-dir, с датой и временем создания в имени)
  - контекстная таблица anomaly_context_* (в папке --anomaly-dir)

Запуск:
  python soc_report.py --work .\\work --features-dir .\\features --scope day --date 2025-12-31
  python soc_report.py --work .\\work --features-dir .\\features --scope week --date 2025-12-31
  python soc_report.py --work .\\work --features-dir .\\features --scope month --date 2025-12-31
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Iterable, Optional

import pandas as pd


TIME_COL_CANDIDATES: list[str] = [
    "Timestamp", "EventTime", "event_time", "time", "TimeGenerated", "StartTime", "EndTime"
]

PROGRAM_COLS: list[str] = [
    "SourceProcessName",
    "ProcessName",
    "Image",
    "Application",
    "Program",
    "CommandLine",
    "DeviceCustomString4",
]

EVENT_NAME_COLS: list[str] = [
    "Name",
    "DeviceEventClassID",
    "DeviceEventCategory",
]

USER_COLS: list[str] = [
    "SourceUserName",
    "DestinationUserName",
    "UserName",
    "AccountName",
    "SubjectUserName",
]

HOST_COLS: list[str] = [
    "DeviceHostname",
    "Computer",
    "Host",
    "Hostname",
    "DeviceName",
]

ADDR_COLS: list[str] = [
    "DestinationAddress",
    "SourceAddress",
    "DestinationIP",
    "SourceIP",
]


@dataclass(frozen=True)
class AnomalyContext:
    entity_type: str
    entity: str
    date: str
    severity: str
    top_contributors: str
    real_user: str
    event_count: int
    time_range: str
    top_hours: str
    top_programs: str
    top_event_names: str
    top_users: str
    top_hosts: str
    top_addresses: str
    source_files: str


def _find_time_col(columns: Iterable[str]) -> Optional[str]:
    cols = list(columns)
    for c in TIME_COL_CANDIDATES:
        if c in cols:
            return c
    for c in cols:
        lc = c.lower()
        if "time" in lc or "date" in lc:
            return c
    return None


def _top_values(df: pd.DataFrame, cols: Iterable[str], top_k: int) -> str:
    chunks = []
    for c in cols:
        if c in df.columns:
            s = df[c].dropna().astype(str).str.strip()
            s = s[s != ""]
            if not s.empty:
                chunks.append(s)
    if not chunks:
        return ""
    combined = pd.concat(chunks, ignore_index=True)
    counts = combined.value_counts().head(top_k)
    return "; ".join([f"{idx} ({cnt})" for idx, cnt in counts.items()])


def _summarize_events(df: pd.DataFrame) -> dict[str, str | int]:
    if df.empty:
        return {
            "event_count": 0,
            "time_range": "",
            "top_hours": "",
            "top_programs": "",
            "top_event_names": "",
            "top_users": "",
            "top_hosts": "",
            "top_addresses": "",
        }

    time_col = _find_time_col(df.columns)
    time_range = ""
    top_hours = ""
    if time_col:
        ts = pd.to_datetime(df[time_col], errors="coerce")
        if not ts.isna().all():
            time_range = f"{ts.min()} — {ts.max()}"
            hours = ts.dt.hour.value_counts().head(5)
            top_hours = "; ".join([f"{idx:02d}:00 ({cnt})" for idx, cnt in hours.items()])

    return {
        "event_count": int(len(df)),
        "time_range": time_range,
        "top_hours": top_hours,
        "top_programs": _top_values(df, PROGRAM_COLS, 5),
        "top_event_names": _top_values(df, EVENT_NAME_COLS, 5),
        "top_users": _top_values(df, USER_COLS, 5),
        "top_hosts": _top_values(df, HOST_COLS, 5),
        "top_addresses": _top_values(df, ADDR_COLS, 5),
    }


def _load_events(work_dir: Path, entity: str, date: str) -> tuple[pd.DataFrame, list[Path]]:
    patterns = [
        f"{entity}_SIEM_{date}.csv",
        f"{entity}_PAN_{date}.csv",
    ]
    files: list[Path] = []
    for pattern in patterns:
        matches = list(work_dir.glob(pattern))
        files.extend(matches)

    if not files:
        return pd.DataFrame(), []

    frames: list[pd.DataFrame] = []
    for path in files:
        frames.append(pd.read_csv(path, dtype=str, keep_default_na=True))
    return pd.concat(frames, ignore_index=True), files


def _escape_md(value: str) -> str:
    return value.replace("|", "\\|")


def _render_markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    if df.empty:
        return ["_Нет данных._", ""]
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    for _, row in df.iterrows():
        vals = [_escape_md(str(row.get(col, ""))) for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    lines.append("")
    return lines


def _load_user_mapping(work_dir: Path) -> dict[str, str]:
    mapping_path = work_dir / "user_mapping.csv"
    if not mapping_path.exists():
        return {}
    df = pd.read_csv(mapping_path, dtype=str, keep_default_na=True)
    if "userXXX" not in df.columns or "user_name" not in df.columns:
        return {}
    return dict(zip(df["userXXX"].fillna(""), df["user_name"].fillna("")))


def _available_dates(anomaly_dir: Path) -> list[str]:
    dates: list[str] = []
    for path in anomaly_dir.glob("anomalies_all_*_explain.csv"):
        name = path.stem
        parts = name.split("_")
        if len(parts) >= 4:
            dates.append(parts[2])
    return sorted({d for d in dates if d})


def _build_date_window(available: list[str], target: str, scope: str) -> list[str]:
    if scope == "day":
        return [target]

    window = 7 if scope == "week" else 30
    target_dt = pd.to_datetime(target, errors="raise").normalize()
    available_dt = pd.to_datetime(pd.Series(available), errors="coerce")

    mask = (available_dt <= target_dt) & (available_dt >= target_dt - pd.Timedelta(days=window - 1))
    return sorted(available_dt[mask].dt.date.astype("string").tolist())


def _context_for_anomalies(
    anomalies: pd.DataFrame,
    work_dir: Path,
    user_mapping: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    contexts: list[AnomalyContext] = []
    warnings: list[str] = []

    cache: dict[tuple[str, str], tuple[pd.DataFrame, list[Path]]] = {}

    for _, row in anomalies.iterrows():
        entity_type = str(row.get("entity_type", ""))
        entity = str(row.get("entity", ""))
        date = str(row.get("date", ""))
        severity = str(row.get("severity", ""))
        top_contributors = str(row.get("top_contributors", ""))
        real_user = user_mapping.get(entity, "") if entity_type == "user" else ""

        cache_key = (entity, date)
        if cache_key in cache:
            events, files = cache[cache_key]
        else:
            events, files = _load_events(work_dir, entity, date)
            cache[cache_key] = (events, files)

        summary = _summarize_events(events)
        if not files:
            warnings.append(f"Нет файлов событий для {entity} ({date}).")

        contexts.append(
            AnomalyContext(
                entity_type=entity_type,
                entity=entity,
                date=date,
                severity=severity,
                top_contributors=top_contributors,
                real_user=real_user,
                event_count=int(summary["event_count"]),
                time_range=str(summary["time_range"]),
                top_hours=str(summary["top_hours"]),
                top_programs=str(summary["top_programs"]),
                top_event_names=str(summary["top_event_names"]),
                top_users=str(summary["top_users"]),
                top_hosts=str(summary["top_hosts"]),
                top_addresses=str(summary["top_addresses"]),
                source_files="; ".join([p.name for p in files]),
            )
        )

    return pd.DataFrame([c.__dict__ for c in contexts]), warnings


def main() -> int:
    p = argparse.ArgumentParser(description="SOC report generator with anomaly context")
    p.add_argument("--work", required=True, help="Work directory with daily CSV events")
    p.add_argument("--features-dir", default="features", help="Features directory (default: features)")
    p.add_argument("--anomaly-dir", default=None, help="Folder with anomaly data (default: features dir)")
    p.add_argument("--report-dir", default="report", help="Output folder for reports (default: report)")
    p.add_argument("--scope", required=True, choices=["day", "week", "month"], help="Report scope")
    p.add_argument("--date", default=None, help="Target date (end date for week/month)")
    args = p.parse_args()

    work_dir = Path(args.work)
    if not work_dir.exists():
        raise FileNotFoundError(f"Work dir not found: {work_dir}")

    features_dir = Path(args.features_dir)
    if not features_dir.is_absolute():
        features_dir = work_dir.parent / features_dir
    if not features_dir.exists():
        raise FileNotFoundError(f"Features dir not found: {features_dir}")

    anomaly_dir = Path(args.anomaly_dir) if args.anomaly_dir else features_dir
    if not anomaly_dir.is_absolute():
        anomaly_dir = features_dir / anomaly_dir
    if not anomaly_dir.exists():
        raise FileNotFoundError(f"Anomaly dir not found: {anomaly_dir}")

    report_dir = Path(args.report_dir)
    if not report_dir.is_absolute():
        report_dir = features_dir.parent / report_dir
    report_dir.mkdir(parents=True, exist_ok=True)

    available = _available_dates(anomaly_dir)
    if not available:
        raise FileNotFoundError("No anomalies_all_*_explain.csv found in anomaly dir.")

    target = args.date or available[-1]
    target = str(pd.to_datetime(target, errors="raise").date())

    dates = _build_date_window(available, target, args.scope)
    if not dates:
        raise ValueError(f"No anomaly dates found for scope={args.scope} and target={target}")

    user_mapping = _load_user_mapping(work_dir)

    report_rows: list[pd.DataFrame] = []
    warnings: list[str] = []

    for date in dates:
        explain_path = anomaly_dir / f"anomalies_all_{date}_explain.csv"
        if not explain_path.exists():
            warnings.append(f"Отсутствует файл объяснений: {explain_path.name}")
            continue

        anomalies = pd.read_csv(explain_path, dtype=str, keep_default_na=True)
        ctx_df, ctx_warnings = _context_for_anomalies(anomalies, work_dir, user_mapping)
        warnings.extend(ctx_warnings)
        report_rows.append(ctx_df)

    if not report_rows:
        raise FileNotFoundError("No anomaly explain files were loaded for the requested scope.")

    report_df = pd.concat(report_rows, ignore_index=True)

    context_out = anomaly_dir / f"anomaly_context_{args.scope}_{target}.csv"
    report_df.to_csv(context_out, index=False)

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = report_dir / f"soc_report_{args.scope}_{target}_{stamp}.md"

    lines: list[str] = [
        f"# SOC отчёт ({args.scope}) до {target}",
        "",
        f"Источник аномалий: {anomaly_dir}",
        f"Период: {', '.join(dates)}",
        "",
    ]

    severity_counts = report_df["severity"].value_counts().to_dict() if "severity" in report_df.columns else {}
    if severity_counts:
        lines.append("## Сводка по уровню серьёзности")
        for sev, count in severity_counts.items():
            lines.append(f"- {sev}: {count}")
        lines.append("")

    columns = [
        "date",
        "entity_type",
        "entity",
        "real_user",
        "severity",
        "top_contributors",
        "event_count",
        "time_range",
        "top_hours",
        "top_programs",
        "top_event_names",
        "top_users",
        "top_hosts",
        "top_addresses",
    ]

    for date in dates:
        lines.append(f"## Аномалии за {date}")
        subset = report_df[report_df["date"] == date]
        if subset.empty:
            lines.append("_Аномалий не найдено._")
            lines.append("")
            continue
        lines.extend(_render_markdown_table(subset, columns))

    if warnings:
        lines.append("## Предупреждения")
        for w in sorted(set(warnings)):
            lines.append(f"- {w}")
        lines.append("")

    report_path.write_text("\n".join(lines), encoding="utf-8")
    print(f"[+] Wrote: {report_path}")
    print(f"[+] Wrote: {context_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
