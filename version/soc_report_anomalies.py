#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
soc_report_anomalies.py

Make SOC L1-friendly daily anomaly report from *_explain.csv produced by explain_anomalies.py.

Inputs (in --work dir):
  - anomalies_users_YYYY-MM-DD_explain.csv
  - anomalies_hosts_YYYY-MM-DD_explain.csv
  (or it can use anomalies_all_YYYY-MM-DD_explain.csv if present)

Outputs (in --work dir):
  - soc_users_YYYY-MM-DD.csv
  - soc_hosts_YYYY-MM-DD.csv
  - soc_all_YYYY-MM-DD.csv

Goal:
- Translate technical feature names into human-readable RU labels
- Convert z-scores into qualitative statements (e.g., "значительно выше нормы")
- Add short triage hint and suggested next steps for SOC L1

Usage:
  python soc_report_anomalies.py --work .\work --date 2025-12-31
  python soc_report_anomalies.py --work .\work            (date defaults to latest *_explain available)
"""

from __future__ import annotations

import argparse
import math
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


DATE_RE = re.compile(r"(anomalies_(users|hosts|all)_(\d{4}-\d{2}-\d{2})_explain\.csv)$", re.IGNORECASE)

# RU-friendly labels for your current feature set
FEATURE_LABELS: Dict[str, str] = {
    "siem_events_total": "Количество SIEM-событий за сутки",
    "siem_unique_destination_addr": "Уникальные DestinationAddress в SIEM",
    "siem_unique_source_process": "Уникальные SourceProcessName в SIEM",
    "siem_unique_event_class": "Уникальные DeviceEventClassID в SIEM",
    "siem_unique_category": "Уникальные категории (DeviceEventCategory) в SIEM",
    "siem_unique_name": "Уникальные Name в SIEM",
    "siem_unique_hours": "Число активных часов в SIEM",
    "siem_night_share": "Доля ночной активности в SIEM (00:00–05:59)",
    "siem_business_share": "Доля активности в рабочие часы в SIEM (09:00–17:59)",

    "pan_events_total": "Количество сетевых PAN-событий за сутки",
    "pan_unique_destination_addr": "Уникальные DestinationAddress в PAN",
    "pan_unique_event_class": "Уникальные DeviceEventClassID в PAN",
    "pan_unique_category": "Уникальные категории (DeviceEventCategory) в PAN",
    "pan_unique_name": "Уникальные Name в PAN",
    "pan_unique_hours": "Число активных часов в PAN",
    "pan_night_share": "Доля ночной активности в PAN (00:00–05:59)",
    "pan_business_share": "Доля активности в рабочие часы в PAN (09:00–17:59)",

    "pan_share_of_all_events": "Доля PAN-событий от всех событий (SIEM+PAN)",

    # host-only
    "siem_unique_users": "Уникальные пользователи (Src/Dst) в событиях SIEM",
    "pan_unique_users": "Уникальные пользователи (Src/Dst) в событиях PAN",

    # calendar
    "day_of_week": "День недели (0=Пн ... 6=Вс)",
    "is_weekend": "Выходной (1) / будний (0)",
}

SEVERITY_RU = {
    "critical": "Критично",
    "high": "Высокий",
    "medium": "Средний",
    "low": "Низкий",
}


def _pick_latest_date(work_dir: Path) -> str:
    dates: List[str] = []
    for p in work_dir.glob("anomalies_*_*_explain.csv"):
        m = DATE_RE.match(p.name)
        if m:
            dates.append(m.group(3))
    if not dates:
        raise FileNotFoundError("No anomalies_*_YYYY-MM-DD_explain.csv found. Run explain_anomalies.py first.")
    return str(pd.to_datetime(pd.Series(dates)).max().date())


def _qual_from_z(z: float) -> str:
    a = abs(z)
    if a >= 3:
        return "значительно"
    if a >= 2:
        return "существенно"
    if a >= 1:
        return "умеренно"
    return "незначительно"


def _direction(z: float) -> str:
    return "выше нормы" if z >= 0 else "ниже нормы"


def _fmt_value(feature: str, v: float) -> str:
    # shares -> percent
    if feature.endswith("_share") or feature.endswith("_night_share") or feature.endswith("_business_share"):
        return f"{v*100:.1f}%"
    if feature in ("pan_share_of_all_events",):
        return f"{v*100:.1f}%"
    # counts -> integer
    return f"{int(round(v))}"


def _triage_hint_from_feature(feature: str, z: float) -> str:
    f = feature
    up = z >= 0

    if "night_share" in f and up:
        return "Ночная активность выше обычной"
    if "business_share" in f and not up:
        return "Меньше активности в рабочие часы, чем обычно (возможна смещённая активность)"
    if "unique_source_process" in f and up:
        return "Появились новые процессы/утилиты (проверьте подозрительные запуски)"
    if "unique_destination_addr" in f and up:
        return "Рост новых адресов назначения (проверьте внешние подключения/DNS/прокси)"
    if "unique_event_class" in f and up:
        return "Появились новые типы событий (DeviceEventClassID)"
    if "unique_category" in f and up:
        return "Появились новые категории событий"
    if "pan_share_of_all_events" in f and up:
        return "Сетевые события (PAN) занимают большую долю, чем обычно"
    if "siem_events_total" in f and up:
        return "Резкий рост общего числа SIEM-событий"
    if "pan_events_total" in f and up:
        return "Резкий рост сетевых PAN-событий"
    if f in ("siem_unique_users", "pan_unique_users") and up:
        return "Много разных пользователей на одном хосте (проверьте общий доступ/админские сессии)"
    if "unique_hours" in f and up:
        return "Активность распределилась по большему числу часов"
    return "Отклонение поведенческого профиля от нормы"


def _suggest_steps(entity_type: str, main_hints: List[str]) -> str:
    steps = []

    # universal
    steps.append("Открыть сырые события за сутки (суточный CSV) и проверить контекст (время, источник, назначения, процессы).")
    steps.append("Сверить с плановыми работами/обновлениями/сканированиями и известными сервисными аккаунтами.")
    steps.append("Проверить EDR/MDR по хосту/пользователю: алерты, изоляции, подозрительные процессы, persistence.")

    # feature-driven add-ons
    hint_text = " ".join(main_hints).lower()
    if "ночн" in hint_text:
        steps.append("Проверить несанкционированные входы/действия ночью (logon, RDP/SMB, административные команды).")
    if "адрес" in hint_text or "подключ" in hint_text or "сетев" in hint_text:
        steps.append("Проверить новые внешние/нетипичные DestinationAddress, домены, прокси-логи, категории трафика.")
    if "процесс" in hint_text or "утилит" in hint_text:
        steps.append("Проверить новые/редкие процессы: путь запуска, родительский процесс, подпись, хеш, командную строку (если есть).")
    if "пользоват" in hint_text and entity_type == "host":
        steps.append("Проверить список пользователей, работавших с хостом, и наличие lateral movement/подборов паролей.")

    # Keep short for L1
    return " ".join(f"{i+1}) {s}" for i, s in enumerate(steps[:5]))


def _build_soc_table(df: pd.DataFrame, kind: str, top_features: int) -> pd.DataFrame:
    # Determine available contributor columns
    # contrib{i}_feature, contrib{i}_z, contrib{i}_value, contrib{i}_baseline_median
    rows = []
    for _, r in df.iterrows():
        entity_type = r.get("entity_type", "user" if kind == "users" else "host")
        entity = r.get("entity")
        date = r.get("date")
        sev = str(r.get("severity", "low")).lower()
        sev_ru = SEVERITY_RU.get(sev, sev)

        reasons = []
        hints = []
        for i in range(1, top_features + 1):
            fcol = f"contrib{i}_feature"
            zcol = f"contrib{i}_z"
            vcol = f"contrib{i}_value"
            bcol = f"contrib{i}_baseline_median"
            if fcol not in df.columns:
                continue
            f = r.get(fcol)
            if pd.isna(f):
                continue

            z = float(r.get(zcol, 0.0))
            v = float(r.get(vcol, 0.0))
            b = float(r.get(bcol, 0.0))

            label = FEATURE_LABELS.get(str(f), str(f))
            qual = _qual_from_z(z)
            dirr = _direction(z)

            # Format value/baseline
            v_s = _fmt_value(str(f), v)
            b_s = _fmt_value(str(f), b)
            reasons.append(f"{label}: {qual} {dirr} (сегодня {v_s}, обычно {b_s})")
            hints.append(_triage_hint_from_feature(str(f), z))

        # Compose single-line summary for L1
        main_hint = hints[0] if hints else "Отклонение поведенческого профиля от нормы"
        summary = f"{sev_ru}: {main_hint}"

        steps = _suggest_steps("host" if entity_type == "host" else "user", hints)

        out = {
            "date": date,
            "entity_type": "Пользователь" if str(entity_type).lower() == "user" else "Хост",
            "entity": entity,
            "severity": sev_ru,
            "priority_rank": r.get("rank_combined", ""),
            "combined_score": r.get("score_combined_norm", ""),
            "summary": summary,
            "reason_1": reasons[0] if len(reasons) > 0 else "",
            "reason_2": reasons[1] if len(reasons) > 1 else "",
            "reason_3": reasons[2] if len(reasons) > 2 else "",
            "reason_4": reasons[3] if len(reasons) > 3 else "",
            "reason_5": reasons[4] if len(reasons) > 4 else "",
            "recommended_steps": steps,
        }
        rows.append(out)

    out_df = pd.DataFrame(rows)

    # Sort: severity then rank
    sev_order = {"Критично": 0, "Высокий": 1, "Средний": 2, "Низкий": 3}
    out_df["_sev_order"] = out_df["severity"].map(sev_order).fillna(9).astype(int)
    out_df["_rank"] = pd.to_numeric(out_df["priority_rank"], errors="coerce").fillna(1e9)
    out_df = out_df.sort_values(["_sev_order", "_rank"], ascending=[True, True], kind="mergesort").drop(columns=["_sev_order", "_rank"])

    return out_df


def _load_explain(work_dir: Path, kind: str, date: str) -> pd.DataFrame:
    path = work_dir / f"anomalies_{kind}_{date}_explain.csv"
    if not path.exists():
        raise FileNotFoundError(f"Missing explain file: {path}")
    df = _read_csv(path, dtype=str)

    # normalize
    if "date" in df.columns:
        df["date"] = pd.to_datetime(df["date"], errors="coerce").dt.date.astype("string")
    return df


def main() -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--work", required=True, help="Work directory")
    p.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: latest available explain date)")
    p.add_argument("--top-features", type=int, default=3, help="How many top features to convert into SOC reasons (3-5)")
    args = p.parse_args()

    work_dir = Path(args.work)
    if not work_dir.exists():
        raise FileNotFoundError(f"Work dir not found: {work_dir}")

    target_date = args.date or _pick_latest_date(work_dir)
    target_date = str(pd.to_datetime(target_date, errors="raise").date())

    k = int(args.top_features)
    if k < 1 or k > 5:
        raise ValueError("--top-features must be between 1 and 5 for SOC report")

    users_ex = _load_explain(work_dir, "users", target_date)
    hosts_ex = _load_explain(work_dir, "hosts", target_date)

    users_soc = _build_soc_table(users_ex, "users", k)
    hosts_soc = _build_soc_table(hosts_ex, "hosts", k)

    out_users = work_dir / f"soc_users_{target_date}.csv"
    out_hosts = work_dir / f"soc_hosts_{target_date}.csv"
    out_all = work_dir / f"soc_all_{target_date}.csv"

    users_soc.to_csv(out_users, index=False)
    hosts_soc.to_csv(out_hosts, index=False)

    all_soc = pd.concat([users_soc, hosts_soc], ignore_index=True)
    all_soc.to_csv(out_all, index=False)

    print(f"[+] Wrote: {out_users} (rows={len(users_soc)})")
    print(f"[+] Wrote: {out_hosts} (rows={len(hosts_soc)})")
    print(f"[+] Wrote: {out_all} (rows={len(all_soc)})")
    print("[*] Done.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
