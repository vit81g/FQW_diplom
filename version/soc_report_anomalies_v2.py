#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
soc_report_anomalies.py

SOC-friendly (L1) report generator.

Reads:
  - anomalies_users_YYYY-MM-DD_explain.csv
  - anomalies_hosts_YYYY-MM-DD_explain.csv
  - (optional) anomalies_all_YYYY-MM-DD_explain.csv

Writes (to --work):
  - soc_report_YYYY-MM-DD.csv   (compact table)
  - soc_report_YYYY-MM-DD.md    (human readable)
  - soc_report_YYYY-MM-DD.json  (metadata: inputs, row counts)

Design goals:
  - Compact, readable, actionable for L1.
  - No speculation: recommendations are generic "what to check" playbook hints,
    not assertions that an incident occurred.

Usage:
  python soc_report_anomalies.py --work .\work --date 2025-12-31
  python soc_report_anomalies.py --work .\work               (date defaults to latest anomalies_*_explain available)
"""

from __future__ import annotations

import argparse
import json
import re
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import pandas as pd


DATE_RE = re.compile(r"anomalies_(users|hosts|all)_(\d{4}-\d{2}-\d{2})_explain\.csv$", re.IGNORECASE)

SEV_RU = {"critical": "Критично", "high": "Высокий", "medium": "Средний", "low": "Низкий"}

# Feature -> (RU label, L1 what-to-check hints)
FEATURE_CATALOG: Dict[str, Tuple[str, List[str]]] = {
    "siem_events_total": (
        "Объём SIEM-событий за сутки",
        [
            "Сопоставить с типовым профилем пользователя/хоста за последние 7–30 дней.",
            "Проверить всплеск по источникам/категориям (категории, классы событий, новые источники).",
        ],
    ),
    "siem_unique_destination_addr": (
        "Уникальные DestinationAddress (SIEM)",
        [
            "Проверить нетипичные/внешние адреса назначения и их принадлежность (ASN/репутация/гео) по вашим средствам.",
            "Проверить признаки сканирования/бокового перемещения: много адресов за короткое время.",
        ],
    ),
    "siem_unique_source_process": (
        "Уникальные SourceProcessName (SIEM)",
        [
            "Проверить появление новых/нетипичных процессов, запускающих сетевую активность или аутентификацию.",
            "Проверить командные интерпретаторы/скрипты (powershell/cmd/wscript) в вашей телеметрии EDR/Sysmon (если есть).",
        ],
    ),
    "siem_unique_event_class": (
        "Уникальные DeviceEventClassID (SIEM)",
        [
            "Проверить, какие классы событий добавились, и связано ли это с изменениями инфраструктуры/обновлениями.",
            "Приоритетно разобрать классы, относящиеся к аутентификации/администрированию/сетевым соединениям.",
        ],
    ),
    "siem_unique_category": (
        "Уникальные DeviceEventCategory (SIEM)",
        [
            "Проверить появление новых категорий (например, аутентификация/доступ/сеть's traffic) относительно базового профиля.",
        ],
    ),
    "siem_unique_name": (
        "Уникальные Name (SIEM)",
        [
            "Проверить новые типы событий (Name) и их первоисточник.",
        ],
    ),
    "siem_unique_hours": (
        "Число уникальных часов активности (SIEM)",
        [
            "Проверить равномерное распределение активности по суткам (в т.ч. ночные часы).",
        ],
    ),
    "siem_night_share": (
        "Доля SIEM-событий ночью (00:00–05:59)",
        [
            "Проверить соответствие графику работы/регламентам, наличие работ/скриптов по расписанию.",
            "Проверить последние успешные/неуспешные входы и изменения привилегий/групп в ночное время.",
        ],
    ),
    "siem_business_share": (
        "Доля SIEM-событий в рабочие часы (09:00–17:59)",
        [
            "Если доля резко снизилась — проверить смещение активности на вечер/ночь.",
        ],
    ),
    "pan_events_total": (
        "Объём PAN-OS TRAFFIC событий за сутки",
        [
            "Проверить всплеск сетевой активности/соединений и соответствие ожидаемым сервисам.",
            "Проверить новые направления трафика (DestinationAddress) и изменения политик/маршрутизации.",
        ],
    ),
    "pan_unique_destination_addr": (
        "Уникальные DestinationAddress (PAN-OS)",
        [
            "Проверить внешние адреса назначения, нетипичные порты/приложения (по данным NGFW).",
            "Проверить возможные признаки сканирования/эксфильтрации (много адресов/сессий).",
        ],
    ),
    "pan_unique_event_class": (
        "Уникальные DeviceEventClassID (PAN-OS)",
        [
            "Проверить новые классы событий/лог-типов PAN-OS, коррелировать с политиками.",
        ],
    ),
    "pan_unique_category": (
        "Уникальные DeviceEventCategory (PAN-OS)",
        [
            "Проверить новые категории/типы трафика, привязать к правилам безопасности.",
        ],
    ),
    "pan_unique_name": (
        "Уникальные Name (PAN-OS)",
        [
            "Проверить нетипичные значения Name (если кроме TRAFFIC присутствуют другие лог-типы).",
        ],
    ),
    "pan_unique_hours": (
        "Число уникальных часов активности (PAN-OS)",
        [
            "Проверить распределение сетевой активности по суткам (в т.ч. ночью).",
        ],
    ),
    "pan_night_share": (
        "Доля PAN-OS трафика ночью (00:00–05:59)",
        [
            "Проверить ночные исходящие соединения, соответствие регламентам.",
        ],
    ),
    "pan_business_share": (
        "Доля PAN-OS трафика в рабочие часы (09:00–17:59)",
        [
            "Если доля резко снизилась — проверить смещение на вечер/ночь.",
        ],
    ),
    "pan_share_of_all_events": (
        "Доля PAN-OS трафика среди всех событий",
        [
            "Проверить, не изменился ли источник данных/фильтрация (возможен перекос в сторону сетевых событий).",
        ],
    ),
    "siem_unique_users": (
        "Уникальные пользователи на хосте (SIEM)",
        [
            "Проверить нетипичных пользователей на хосте (админские входы, новые учётные записи, сервисные учётки).",
        ],
    ),
    "pan_unique_users": (
        "Уникальные пользователи на хосте (PAN-OS)",
        [
            "Проверить, какие пользователи генерировали трафик с хоста (если PAN-логи обогащены пользователем).",
        ],
    ),
}


def _pick_date_from_workdir(work_dir: Path) -> str:
    dates: List[str] = []
    for p in work_dir.glob("anomalies_*_*_explain.csv"):
        m = DATE_RE.match(p.name)
        if m:
            dates.append(m.group(2))
    if not dates:
        raise FileNotFoundError("No anomalies_*_explain.csv found. Run explain_anomalies.py first.")
    dt = pd.to_datetime(pd.Series(dates), errors="coerce")
    return str(dt.max().date())


def _read_explain(work_dir: Path, kind: str, date: str) -> pd.DataFrame:
    p = work_dir / f"anomalies_{kind}_{date}_explain.csv"
    if not p.exists():
        return pd.DataFrame()
    return pd.read_csv(p, dtype=str, keep_default_na=True)


def _feature_to_ru(feature: str) -> str:
    if feature in FEATURE_CATALOG:
        return FEATURE_CATALOG[feature][0]
    return feature  # fallback: raw name


def _hints_for_features(features: List[str], entity_type: str) -> List[str]:
    hints: List[str] = []
    for f in features:
        if f in FEATURE_CATALOG:
            hints.extend(FEATURE_CATALOG[f][1])
    # Deduplicate preserving order
    out: List[str] = []
    seen = set()
    for h in hints:
        if h not in seen:
            out.append(h)
            seen.add(h)
    # Add a short generic close-out line
    if entity_type == "user":
        tail = "При необходимости — запросить подтверждение активности у владельца учётной записи и/или руководителя."
    else:
        tail = "При необходимости — проверить изменения на хосте (патчи/настройки/задачи по расписанию) и владельца системы."
    if tail not in seen:
        out.append(tail)
    return out[:6]  # keep compact for L1


def _extract_top_features(row: pd.Series) -> List[str]:
    feats: List[str] = []
    # Prefer structured columns if present
    for i in range(1, 11):
        c = f"contrib{i}_feature"
        if c in row and pd.notna(row[c]):
            v = str(row[c]).strip()
            if v:
                feats.append(v)
    if feats:
        return feats[:5]
    # Fallback parse "top_contributors"
    tc = str(row.get("top_contributors", "") or "")
    if not tc:
        return []
    # pattern "feature:+z (v=..., base=...)"
    parts = [p.strip() for p in tc.split(";") if p.strip()]
    for p in parts:
        if ":" in p:
            feats.append(p.split(":", 1)[0].strip())
    return feats[:5]


def _format_explanation(row: pd.Series, top_features: List[str]) -> str:
    # Build short RU explanation list: "FeatureRU: z=..., value vs baseline"
    items: List[str] = []
    for i, f in enumerate(top_features, start=1):
        zc = f"contrib{i}_z"
        vc = f"contrib{i}_value"
        bc = f"contrib{i}_baseline_median"
        z = row.get(zc, None)
        v = row.get(vc, None)
        b = row.get(bc, None)

        label = _feature_to_ru(f)

        def _num(x):
            try:
                return float(x)
            except Exception:
                return None

        zf = _num(z)
        vf = _num(v)
        bf = _num(b)

        if zf is not None and vf is not None and bf is not None:
            sign = "+" if zf >= 0 else ""
            items.append(f"{label}: z={sign}{zf:.2f}, v={vf:.0f}, base={bf:.0f}")
        else:
            items.append(f"{label}")
    return " | ".join(items)


def _severity_ru(sev: str) -> str:
    return SEV_RU.get((sev or "").strip().lower(), sev or "")


def build_soc_table(df: pd.DataFrame) -> pd.DataFrame:
    if df.empty:
        return df

    rows = []
    for _, row in df.iterrows():
        et = str(row.get("entity_type", "")).strip() or "unknown"
        ent = str(row.get("entity", "")).strip()
        date = str(row.get("date", "")).strip()
        sev = str(row.get("severity", "")).strip().lower()
        sev_ru = _severity_ru(sev)

        top_feats = _extract_top_features(row)
        explain_short = _format_explanation(row, top_feats)
        hints = _hints_for_features(top_feats, "user" if et == "user" else "host")

        rows.append(
            {
                "date": date,
                "entity_type": et,
                "entity": ent,
                "severity_ru": sev_ru,
                "explain_short": explain_short,
                "what_to_check": " ; ".join(hints),
                "rank_combined": row.get("rank_combined", ""),
                "score_combined_norm": row.get("score_combined_norm", ""),
            }
        )

    out = pd.DataFrame(rows)

    # Sort by severity order then rank
    sev_order = {"Критично": 0, "Высокий": 1, "Средний": 2, "Низкий": 3}
    out["_sev_order"] = out["severity_ru"].map(sev_order).fillna(9).astype(int)
    out["rank_combined_num"] = pd.to_numeric(out["rank_combined"], errors="coerce").fillna(1e9)
    out = out.sort_values(["_sev_order", "rank_combined_num"], kind="mergesort").drop(columns=["_sev_order", "rank_combined_num"])

    # Keep compact column order
    cols = ["date", "entity_type", "entity", "severity_ru", "explain_short", "what_to_check", "score_combined_norm", "rank_combined"]
    for c in list(out.columns):
        if c not in cols:
            cols.append(c)
    return out[cols]


def write_markdown(work_dir: Path, date: str, table: pd.DataFrame, top_n: int) -> Path:
    p = work_dir / f"soc_report_{date}.md"
    lines: List[str] = []
    lines.append(f"# SOC Report (L1) — {date}")
    lines.append("")
    if table.empty:
        lines.append("_Нет данных: отсутствуют anomalies_*_explain.csv или они пусты._")
        p.write_text("\n".join(lines), encoding="utf-8")
        return p

    # Summary counts
    counts = table["severity_ru"].value_counts().to_dict()
    lines.append("## Сводка")
    lines.append("")
    lines.append(f"- Всего сущностей в отчёте: **{len(table)}**")
    for k in ["Критично", "Высокий", "Средний", "Низкий"]:
        if k in counts:
            lines.append(f"- {k}: **{int(counts[k])}**")
    lines.append("")

    # Top N table
    view = table.head(top_n).copy()

    lines.append(f"## ТОП-{min(top_n, len(view))} аномалий")
    lines.append("")
    lines.append("| # | Тип | Сущность | Критичность | Краткое объяснение (TOP отклонения) | Что проверить (L1) |")
    lines.append("|---:|---|---|---|---|---|")
    for i, r in enumerate(view.itertuples(index=False), start=1):
        et = getattr(r, "entity_type")
        ent = getattr(r, "entity")
        sev = getattr(r, "severity_ru")
        ex = getattr(r, "explain_short")
        wt = getattr(r, "what_to_check")
        # sanitize pipes
        ex = str(ex).replace("|", "¦")
        wt = str(wt).replace("|", "¦")
        lines.append(f"| {i} | {et} | {ent} | {sev} | {ex} | {wt} |")

    lines.append("")
    lines.append("## Примечания")
    lines.append("")
    lines.append("- Отчёт носит характер подсказки для первичного триажа (L1) и не является подтверждением инцидента.")
    lines.append("- Рекомендуется коррелировать с первичными логами в SIEM/NGFW/EDR и контекстом (изменения, работы, инциденты).")
    lines.append("")

    p.write_text("\n".join(lines), encoding="utf-8")
    return p


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--work", required=True, help="Work directory")
    ap.add_argument("--date", default=None, help="Target date YYYY-MM-DD (default: latest anomalies_*_explain date)")
    ap.add_argument("--top", type=int, default=30, help="How many rows to show in MD top table")
    args = ap.parse_args()

    work_dir = Path(args.work)
    if not work_dir.exists():
        raise FileNotFoundError(f"Work dir not found: {work_dir}")

    date = args.date or _pick_date_from_workdir(work_dir)
    date = str(pd.to_datetime(date, errors="raise").date())

    # Prefer combined file if present; else merge users+hosts
    df_all = _read_explain(work_dir, "all", date)
    inputs = []
    if not df_all.empty:
        inputs.append(f"anomalies_all_{date}_explain.csv")
        df = df_all
    else:
        du = _read_explain(work_dir, "users", date)
        dh = _read_explain(work_dir, "hosts", date)
        if not du.empty:
            inputs.append(f"anomalies_users_{date}_explain.csv")
        if not dh.empty:
            inputs.append(f"anomalies_hosts_{date}_explain.csv")
        df = pd.concat([du, dh], ignore_index=True) if (not du.empty or not dh.empty) else pd.DataFrame()

    out_table = build_soc_table(df)

    out_csv = work_dir / f"soc_report_{date}.csv"
    out_json = work_dir / f"soc_report_{date}.json"
    out_md = write_markdown(work_dir, date, out_table, int(args.top))

    out_table.to_csv(out_csv, index=False)

    meta = {
        "date": date,
        "inputs": inputs,
        "rows_in": int(len(df)),
        "rows_out": int(len(out_table)),
        "top_shown_in_md": int(min(int(args.top), len(out_table))),
        "version": "1.0",
    }
    out_json.write_text(json.dumps(meta, ensure_ascii=False, indent=2), encoding="utf-8")

    print(f"[+] Wrote: {out_csv}")
    print(f"[+] Wrote: {out_md}")
    print(f"[+] Wrote: {out_json}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
