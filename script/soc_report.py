#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
soc_report.py

Формирование SOC-отчётов с контекстом по аномалиям.

Отчёт включает:
- человекочитаемое описание аномалии (действия пользователя/хоста),
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
    description: str
    real_user: str
    event_count: int
    time_range: str
    top_hours: str
    programs_used: str
    top_event_names: str
    top_users: str
    top_hosts: str
    top_addresses: str
    source_files: str


def _find_time_col(columns: Iterable[str]) -> Optional[str]:
    """Ищет колонку времени в таблице событий."""
    # Приводим входные колонки к списку для повторных проходов.
    cols = list(columns)
    # Сначала проверяем заранее известные варианты.
    for c in TIME_COL_CANDIDATES:
        if c in cols:
            return c
    # Если стандартных имён нет, ищем по ключевым словам.
    for c in cols:
        lc = c.lower()
        if "time" in lc or "date" in lc:
            return c
    # Если ничего не нашли, возвращаем None.
    return None


def _top_values(df: pd.DataFrame, cols: Iterable[str], top_k: int) -> str:
    """Возвращает топ-значения по набору колонок в формате 'значение (кол-во)'."""
    # Готовим контейнер для собранных значений.
    chunks: list[pd.Series] = []
    # Проходим по каждой колонке и собираем валидные значения.
    for c in cols:
        if c in df.columns:
            # Очищаем пропуски и пробелы, приводим к строкам.
            s = df[c].dropna().astype(str).str.strip()
            # Убираем пустые строки.
            s = s[s != ""]
            # Если что-то осталось, добавляем в список.
            if not s.empty:
                chunks.append(s)
    # Если значений нет — возвращаем пустую строку.
    if not chunks:
        return ""
    # Объединяем все значения в один Series для подсчёта.
    combined = pd.concat(chunks, ignore_index=True)
    # Считаем частоты и берём top_k.
    counts = combined.value_counts().head(top_k)
    # Формируем строку для вывода.
    return "; ".join([f"{idx} ({cnt})" for idx, cnt in counts.items()])


def _summarize_events(df: pd.DataFrame) -> dict[str, object]:
    """Собирает агрегированные метрики по событиям для отчёта."""
    # Если событий нет — возвращаем пустую структуру.
    if df.empty:
        return {
            "event_count": 0,
            "time_range": "",
            "top_hours": "",
            "hour_counts": [],
            "programs_used": "",
            "top_event_names": "",
            "top_users": "",
            "top_hosts": "",
            "top_addresses": "",
        }

    # Ищем колонку времени, чтобы определить диапазон активности.
    time_col = _find_time_col(df.columns)
    time_range = ""
    top_hours = ""
    hour_counts: list[tuple[int, int]] = []
    if time_col:
        # Конвертируем время в datetime.
        ts = pd.to_datetime(df[time_col], errors="coerce")
        # Если есть корректные даты — вычисляем диапазон и часы пик.
        if not ts.isna().all():
            time_range = f"{ts.min()} — {ts.max()}"
            hours = ts.dt.hour.value_counts().head(5)
            hour_counts = [(int(idx), int(cnt)) for idx, cnt in hours.items()]
            top_hours = "; ".join([f"{idx:02d}:00 ({cnt})" for idx, cnt in hours.items()])

    # Возвращаем итоговую структуру агрегатов.
    return {
        "event_count": int(len(df)),
        "time_range": time_range,
        "top_hours": top_hours,
        "hour_counts": hour_counts,
        "programs_used": _top_values(df, PROGRAM_COLS, 5),
        "top_event_names": _top_values(df, EVENT_NAME_COLS, 5),
        "top_users": _top_values(df, USER_COLS, 5),
        "top_hosts": _top_values(df, HOST_COLS, 5),
        "top_addresses": _top_values(df, ADDR_COLS, 5),
    }


def _load_events(work_dir: Path, entity: str, date: str) -> tuple[pd.DataFrame, list[Path]]:
    """Загружает дневные события по сущности и возвращает объединённый DataFrame."""
    # Формируем шаблоны имён файлов.
    patterns = [
        f"{entity}_SIEM_{date}.csv",
        f"{entity}_PAN_{date}.csv",
    ]
    # Список всех найденных файлов.
    files: list[Path] = []
    # Ищем файлы по каждому шаблону.
    for pattern in patterns:
        matches = list(work_dir.glob(pattern))
        files.extend(matches)

    # Если файлов нет — возвращаем пустой фрейм.
    if not files:
        return pd.DataFrame(), []

    # Читаем каждый файл в отдельный DataFrame.
    frames: list[pd.DataFrame] = []
    for path in files:
        frames.append(pd.read_csv(path, dtype=str, keep_default_na=True))
    # Объединяем все части в один DataFrame.
    return pd.concat(frames, ignore_index=True), files


def _escape_md(value: str) -> str:
    """Экранирует символы Markdown-таблицы."""
    # Экранируем вертикальные разделители.
    return value.replace("|", "\\|")


def _render_markdown_table(df: pd.DataFrame, columns: list[str]) -> list[str]:
    """Рендерит DataFrame в Markdown-таблицу."""
    # Если данных нет — возвращаем простую заглушку.
    if df.empty:
        return ["_Нет данных._", ""]
    # Формируем заголовок и разделитель.
    header = "| " + " | ".join(columns) + " |"
    divider = "| " + " | ".join(["---"] * len(columns)) + " |"
    lines = [header, divider]
    # Для каждой строки формируем строку таблицы.
    for _, row in df.iterrows():
        vals = [_escape_md(str(row.get(col, ""))) for col in columns]
        lines.append("| " + " | ".join(vals) + " |")
    # Добавляем пустую строку после таблицы.
    lines.append("")
    return lines


def _load_user_mapping(work_dir: Path) -> dict[str, str]:
    """Загружает карту соответствий userXXX → реальное имя (если файл есть)."""
    # Путь к файлу сопоставлений.
    mapping_path = work_dir / "user_mapping.csv"
    # Если файла нет — возвращаем пустой словарь.
    if not mapping_path.exists():
        return {}
    # Читаем CSV с сопоставлениями.
    df = pd.read_csv(mapping_path, dtype=str, keep_default_na=True)
    # Проверяем наличие нужных колонок.
    if "userXXX" not in df.columns or "user_name" not in df.columns:
        return {}
    # Строим словарь userXXX → user_name.
    return dict(zip(df["userXXX"].fillna(""), df["user_name"].fillna("")))


def _available_dates(anomaly_dir: Path) -> list[str]:
    """Собирает доступные даты по файлам anomalies_all_*_explain.csv."""
    # Список найденных дат.
    dates: list[str] = []
    # Проходим по файлам с объяснениями.
    for path in anomaly_dir.glob("anomalies_all_*_explain.csv"):
        name = path.stem
        parts = name.split("_")
        if len(parts) >= 4:
            dates.append(parts[2])
    # Возвращаем уникальные даты в отсортированном виде.
    return sorted({d for d in dates if d})


def _build_date_window(available: list[str], target: str, scope: str) -> list[str]:
    """Формирует список дат для отчёта (день/неделя/месяц)."""
    # Для дневного отчёта возвращаем только целевую дату.
    if scope == "day":
        return [target]

    # Определяем размер окна в днях.
    window = 7 if scope == "week" else 30
    # Нормализуем целевую дату.
    target_dt = pd.to_datetime(target, errors="raise").normalize()
    # Преобразуем доступные даты в datetime.
    available_dt = pd.to_datetime(pd.Series(available), errors="coerce")

    # Оставляем даты, попадающие в заданное окно.
    mask = (available_dt <= target_dt) & (available_dt >= target_dt - pd.Timedelta(days=window - 1))
    return sorted(available_dt[mask].dt.date.astype("string").tolist())


def _format_hour_list(hours: list[tuple[int, int]]) -> str:
    """Формирует строку с перечислением часов активности."""
    # Если часов нет — возвращаем пустую строку.
    if not hours:
        return ""
    # Преобразуем список часов в человекочитаемый формат.
    return "; ".join([f"{hour:02d}:00 ({count})" for hour, count in hours])


def _detect_anomalous_hours(hours: list[tuple[int, int]]) -> str:
    """Определяет подозрительные часы активности (ночь/поздний вечер)."""
    # Задаём диапазоны, которые считаем подозрительными.
    suspicious_hours = {0, 1, 2, 3, 4, 5, 22, 23}
    # Оставляем только подозрительные часы.
    suspicious = [(hour, count) for hour, count in hours if hour in suspicious_hours]
    # Если таких часов нет — возвращаем пустую строку.
    if not suspicious:
        return ""
    # Формируем строку с подозрительными часами.
    return _format_hour_list(suspicious)


def _build_description(
    entity_type: str,
    entity: str,
    summary: dict[str, object],
) -> str:
    """Строит человекочитаемое описание активности для SOC L1."""
    # Извлекаем значения из summary.
    event_count = int(summary.get("event_count", 0))
    time_range = str(summary.get("time_range", ""))
    top_hours = str(summary.get("top_hours", ""))
    programs_used = str(summary.get("programs_used", ""))
    top_event_names = str(summary.get("top_event_names", ""))
    top_addresses = str(summary.get("top_addresses", ""))
    hour_counts = summary.get("hour_counts", [])
    # Определяем подозрительные часы.
    suspicious_hours = _detect_anomalous_hours(hour_counts if isinstance(hour_counts, list) else [])

    # Готовим части описания.
    parts: list[str] = []
    # Добавляем общий контекст по событиям.
    parts.append(f"События: {event_count}.")
    # Добавляем диапазон времени.
    if time_range:
        parts.append(f"Диапазон времени: {time_range}.")
    # Добавляем часы пик.
    if top_hours:
        parts.append(f"Пиковые часы: {top_hours}.")
    # Отмечаем подозрительное время работы.
    if suspicious_hours:
        parts.append(f"Аномальное время работы (ночь/поздний вечер): {suspicious_hours}.")
    # Добавляем информацию о ПО.
    if programs_used:
        parts.append(f"Запуск ПО: {programs_used}.")
    # Добавляем типы/ID событий.
    if top_event_names:
        parts.append(f"События/ID: {top_event_names}.")
    # Добавляем сетевую активность.
    if top_addresses:
        parts.append(f"Сетевые адреса/сайты: {top_addresses}.")

    # Собираем итоговую строку.
    description = " ".join(parts)
    # Если описания нет — возвращаем резервную строку.
    return description if description else f"Аномальная активность {entity_type} {entity} без детализации."


def _context_for_anomalies(
    anomalies: pd.DataFrame,
    work_dir: Path,
    user_mapping: dict[str, str],
) -> tuple[pd.DataFrame, list[str]]:
    """Строит контекст для списка аномалий."""
    # Итоговый список контекстов.
    contexts: list[AnomalyContext] = []
    # Список предупреждений.
    warnings: list[str] = []

    # Кэш, чтобы не читать одни и те же файлы по несколько раз.
    cache: dict[tuple[str, str], tuple[pd.DataFrame, list[Path]]] = {}

    # Проходим по каждой аномалии.
    for _, row in anomalies.iterrows():
        # Считываем базовые поля аномалии.
        entity_type: str = str(row.get("entity_type", ""))
        entity: str = str(row.get("entity", ""))
        date: str = str(row.get("date", ""))
        severity: str = str(row.get("severity", ""))
        # Определяем реального пользователя, если есть сопоставление.
        real_user: str = user_mapping.get(entity, "") if entity_type == "user" else ""

        # Загружаем события по сущности и дате (из кэша или с диска).
        cache_key: tuple[str, str] = (entity, date)
        if cache_key in cache:
            events, files = cache[cache_key]
        else:
            events, files = _load_events(work_dir, entity, date)
            cache[cache_key] = (events, files)

        # Строим агрегированную сводку по событиям.
        summary: dict[str, object] = _summarize_events(events)
        # Если файлов нет — фиксируем предупреждение.
        if not files:
            warnings.append(f"Нет файлов событий для {entity} ({date}).")

        # Формируем человекочитаемое описание.
        description: str = _build_description(entity_type, entity, summary)

        # Добавляем запись в контекст.
        contexts.append(
            AnomalyContext(
                entity_type=entity_type,
                entity=entity,
                date=date,
                severity=severity,
                description=description,
                real_user=real_user,
                event_count=int(summary["event_count"]),
                time_range=str(summary["time_range"]),
                top_hours=str(summary["top_hours"]),
                programs_used=str(summary["programs_used"]),
                top_event_names=str(summary["top_event_names"]),
                top_users=str(summary["top_users"]),
                top_hosts=str(summary["top_hosts"]),
                top_addresses=str(summary["top_addresses"]),
                source_files="; ".join([p.name for p in files]),
            )
        )

    # Возвращаем DataFrame и предупреждения.
    # Преобразуем список датаклассов в DataFrame.
    context_df: pd.DataFrame = pd.DataFrame([c.__dict__ for c in contexts])
    # Переименовываем колонку с описанием в человекочитаемое имя.
    context_df = context_df.rename(columns={"description": "описание", "programs_used": "программы"})
    return context_df, warnings


def main() -> int:
    """Точка входа: готовит SOC-отчёт по выбранному периоду."""
    # Описываем аргументы командной строки.
    p = argparse.ArgumentParser(description="SOC report generator with anomaly context")
    p.add_argument("--work", required=True, help="Work directory with daily CSV events")
    p.add_argument("--features-dir", default="features", help="Features directory (default: features)")
    p.add_argument("--anomaly-dir", default=None, help="Folder with anomaly data (default: features dir)")
    p.add_argument("--report-dir", default="report", help="Output folder for reports (default: report)")
    p.add_argument("--scope", required=True, choices=["day", "week", "month"], help="Report scope")
    p.add_argument("--date", default=None, help="Target date (end date for week/month)")
    # Парсим аргументы.
    args = p.parse_args()

    # Приводим рабочую директорию к Path.
    work_dir: Path = Path(args.work)
    # Проверяем наличие директории.
    if not work_dir.exists():
        raise FileNotFoundError(f"Work dir not found: {work_dir}")

    # Определяем директорию признаков.
    features_dir: Path = Path(args.features_dir)
    # Если путь относительный — строим от родителя work.
    if not features_dir.is_absolute():
        features_dir = work_dir.parent / features_dir
    # Проверяем существование.
    if not features_dir.exists():
        raise FileNotFoundError(f"Features dir not found: {features_dir}")

    # Определяем директорию аномалий.
    anomaly_dir: Path = Path(args.anomaly_dir) if args.anomaly_dir else features_dir
    # Приводим к абсолютному пути при необходимости.
    if not anomaly_dir.is_absolute():
        anomaly_dir = features_dir / anomaly_dir
    # Проверяем существование.
    if not anomaly_dir.exists():
        raise FileNotFoundError(f"Anomaly dir not found: {anomaly_dir}")

    # Определяем директорию отчётов.
    report_dir: Path = Path(args.report_dir)
    # Строим абсолютный путь, если нужно.
    if not report_dir.is_absolute():
        report_dir = features_dir.parent / report_dir
    # Создаём директорию при отсутствии.
    report_dir.mkdir(parents=True, exist_ok=True)

    # Получаем список доступных дат по объяснениям.
    available: list[str] = _available_dates(anomaly_dir)
    # Если дат нет — это ошибка.
    if not available:
        raise FileNotFoundError("No anomalies_all_*_explain.csv found in anomaly dir.")

    # Выбираем целевую дату (последняя, если не задана).
    target: str = args.date or available[-1]
    # Нормализуем формат даты.
    target = str(pd.to_datetime(target, errors="raise").date())

    # Формируем список дат, попадающих в окно отчёта.
    dates: list[str] = _build_date_window(available, target, args.scope)
    # Проверяем, что окно не пустое.
    if not dates:
        raise ValueError(f"No anomaly dates found for scope={args.scope} and target={target}")

    # Загружаем сопоставление userXXX → реальное имя.
    user_mapping: dict[str, str] = _load_user_mapping(work_dir)

    # Список таблиц отчёта по дням.
    report_rows: list[pd.DataFrame] = []
    # Список предупреждений по данным.
    warnings: list[str] = []

    # Формируем отчёт по каждой дате.
    for date in dates:
        # Путь к файлу объяснений.
        explain_path = anomaly_dir / f"anomalies_all_{date}_explain.csv"
        # Если файла нет — фиксируем предупреждение и продолжаем.
        if not explain_path.exists():
            warnings.append(f"Отсутствует файл объяснений: {explain_path.name}")
            continue

        # Загружаем объяснения.
        anomalies = pd.read_csv(explain_path, dtype=str, keep_default_na=True)
        # Формируем контекст по событиям.
        ctx_df, ctx_warnings = _context_for_anomalies(anomalies, work_dir, user_mapping)
        # Добавляем предупреждения.
        warnings.extend(ctx_warnings)
        # Добавляем таблицу в список.
        report_rows.append(ctx_df)

    # Если ни одной таблицы не получено — ошибка.
    if not report_rows:
        raise FileNotFoundError("No anomaly explain files were loaded for the requested scope.")

    # Объединяем таблицы в одну.
    report_df: pd.DataFrame = pd.concat(report_rows, ignore_index=True)

    # Путь для контекстной таблицы.
    context_out: Path = anomaly_dir / f"anomaly_context_{args.scope}_{target}.csv"
    # Сохраняем таблицу контекста.
    report_df.to_csv(context_out, index=False)

    # Формируем имя отчёта с временной меткой.
    stamp: str = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path: Path = report_dir / f"soc_report_{args.scope}_{target}_{stamp}.md"

    # Формируем строки отчёта.
    lines: list[str] = [
        f"# SOC отчёт ({args.scope}) до {target}",
        "",
        f"Источник аномалий: {anomaly_dir}",
        f"Период: {', '.join(dates)}",
        "",
    ]

    # Добавляем сводку по уровням серьёзности.
    severity_counts = report_df["severity"].value_counts().to_dict() if "severity" in report_df.columns else {}
    if severity_counts:
        lines.append("## Сводка по уровню серьёзности")
        for sev, count in severity_counts.items():
            lines.append(f"- {sev}: {count}")
        lines.append("")

    # Колонки, которые попадут в итоговую таблицу.
    columns = [
        "date",
        "entity_type",
        "entity",
        "real_user",
        "severity",
        "описание",
        "event_count",
        "time_range",
        "top_hours",
        "программы",
        "top_event_names",
        "top_users",
        "top_hosts",
        "top_addresses",
    ]

    # Формируем секции по каждому дню.
    for date in dates:
        lines.append(f"## Аномалии за {date}")
        subset = report_df[report_df["date"] == date]
        if subset.empty:
            lines.append("_Аномалий не найдено._")
            lines.append("")
            continue
        lines.extend(_render_markdown_table(subset, columns))

    # Добавляем предупреждения, если они есть.
    if warnings:
        lines.append("## Предупреждения")
        for w in sorted(set(warnings)):
            lines.append(f"- {w}")
        lines.append("")

    # Сохраняем отчёт.
    report_path.write_text("\n".join(lines), encoding="utf-8")
    # Сообщаем о результатах.
    print(f"[+] Wrote: {report_path}")
    print(f"[+] Wrote: {context_out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
