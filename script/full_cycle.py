#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
full_cycle.py

Полный цикл обработки данных и генерации отчётов из одного сценария.

Скрипт выполняет последовательно:
  1) нормализацию исходных выгрузок SIEM (python_script.py);
  2) построение признаков (build_features_v2.py);
  3) очистку признаков (preprocess_features.py);
  4) обучение и скоринг аномалий (train_anomaly_models.py);
  5) объяснение аномалий (explain_anomalies.py);
  6) визуализацию отчётов (visualize_reports.py);
  7) SOC-отчёт с контекстом (soc_report.py).

Интерактивный ввод:
  - выбор диапазона отчёта: день/неделя/месяц/диапазон/за всё время;
  - даты по формату YYYY-MM-DD.
"""

from __future__ import annotations

import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Iterable, List

import pandas as pd

import viz_core as core


def _script_path(base_dir: Path, filename: str) -> Path:
    """Возвращает полный путь до скрипта по имени файла."""
    # Формируем путь относительно директории текущего файла.
    return base_dir / filename


def _run(cmd: list[str]) -> None:
    """Запускает подскрипт и завершает работу при ошибке."""
    # Печатаем команду для наглядного лога.
    print(f"\n[RUN] {' '.join(cmd)}")
    # Запускаем подскрипт и проверяем код возврата.
    subprocess.run(cmd, check=True)


def _parse_date(value: str) -> str:
    """Проверяет формат YYYY-MM-DD и возвращает нормализованную дату."""
    # Преобразуем строку в дату.
    parsed = datetime.strptime(value.strip(), "%Y-%m-%d").date()
    # Возвращаем дату в каноничном строковом виде.
    return str(parsed)


def _prompt_scope() -> str:
    """Запрашивает у пользователя масштаб отчёта."""
    # Справочник допустимых вариантов.
    scopes = {
        "1": "day",
        "2": "week",
        "3": "month",
        "4": "range",
        "5": "all",
    }
    # Показываем пользователю меню выбора.
    print("Выберите диапазон отчёта:")
    print("  1) День")
    print("  2) Неделя")
    print("  3) Месяц")
    print("  4) Диапазон")
    print("  5) За всё время")
    # Цикл ввода до корректного значения.
    while True:
        choice = input("Введите номер варианта: ").strip()
        if choice in scopes:
            return scopes[choice]
        print("Некорректный выбор. Повторите ввод.")


def _prompt_date(label: str) -> str:
    """Запрашивает дату и валидирует формат."""
    # Цикл ввода до корректной даты.
    while True:
        raw = input(f"{label} (YYYY-MM-DD): ").strip()
        try:
            return _parse_date(raw)
        except ValueError:
            print("Неверный формат. Используйте YYYY-MM-DD.")


def _filter_dates(available: Iterable[str], start: str, end: str) -> List[str]:
    """Возвращает отсортированный список дат внутри диапазона."""
    # Приводим вход к списку и сортируем.
    dates = sorted(available)
    # Оставляем только даты в заданных границах.
    return [d for d in dates if start <= d <= end]


def _window_dates(available: list[str], target: str, scope: str) -> list[str]:
    """Формирует список дат для отчётного окна."""
    # Для "day" нужен только один день.
    if scope == "day":
        return [target]
    # Для "week" и "month" используем окно на 7 или 30 дней.
    window = 7 if scope == "week" else 30
    idx = available.index(target)
    start = max(0, idx - (window - 1))
    return available[start: idx + 1]


def _run_reports(
    base_dir: Path,
    work_dir: Path,
    features_dir: Path,
    report_dir: Path,
    scope: str,
    target: str,
    dates_for_anomalies: list[str],
) -> None:
    """Запускает этапы обучения, объяснения и формирования отчётов."""
    # Готовим пути к используемым скриптам.
    train_script = _script_path(base_dir, "train_anomaly_models.py")
    explain_script = _script_path(base_dir, "explain_anomalies.py")
    visualize_script = _script_path(base_dir, "visualize_reports.py")
    soc_script = _script_path(base_dir, "soc_report.py")

    # Сначала готовим аномалии и объяснения по всем нужным датам.
    for day in dates_for_anomalies:
        # Обучаем модели и считаем скоринг для выбранной даты.
        _run([sys.executable, str(train_script), "--work", str(features_dir), "--date", day])
        # Формируем объяснения для конкретного дня.
        _run([sys.executable, str(explain_script), "--work", str(features_dir), "--date", day])

    # Генерируем визуализации и SOC-отчёт.
    if scope in {"day", "week", "month"}:
        # Визуализация по выбранному масштабу.
        _run(
            [
                sys.executable,
                str(visualize_script),
                "--work",
                str(features_dir),
                "--report-dir",
                str(report_dir),
                "--scope",
                scope,
                "--date",
                target,
            ]
        )
        # SOC-отчёт по выбранному масштабу.
        _run(
            [
                sys.executable,
                str(soc_script),
                "--work",
                str(work_dir),
                "--features-dir",
                str(features_dir),
                "--report-dir",
                str(report_dir),
                "--scope",
                scope,
                "--date",
                target,
            ]
        )
        return

    # Для диапазона и "всё время" делаем суточные отчёты по каждому дню.
    for day in dates_for_anomalies:
        # Визуализация за один день.
        _run(
            [
                sys.executable,
                str(visualize_script),
                "--work",
                str(features_dir),
                "--report-dir",
                str(report_dir),
                "--scope",
                "day",
                "--date",
                day,
            ]
        )
        # SOC-отчёт за один день.
        _run(
            [
                sys.executable,
                str(soc_script),
                "--work",
                str(work_dir),
                "--features-dir",
                str(features_dir),
                "--report-dir",
                str(report_dir),
                "--scope",
                "day",
                "--date",
                day,
            ]
        )


def main() -> int:
    """Точка входа: полный цикл обработки с интерактивным вводом."""
    # Базовая директория скриптов.
    base_dir = Path(__file__).resolve().parent
    # Пути к стандартным каталогам.
    data_dir = base_dir / "data"
    work_dir = base_dir / "work"
    features_dir = base_dir / "features"
    report_dir = base_dir / "report"

    # Запрашиваем у пользователя желаемый диапазон отчёта.
    scope = _prompt_scope()

    # Запрашиваем даты в зависимости от выбранного режима.
    target = ""
    range_start = ""
    range_end = ""
    if scope == "day":
        target = _prompt_date("Введите дату для дня")
    elif scope == "week":
        target = _prompt_date("Введите дату, с которой считается неделя")
    elif scope == "month":
        target = _prompt_date("Введите дату, с которой считается месяц")
    elif scope == "range":
        range_start = _prompt_date("Введите начало диапазона")
        range_end = _prompt_date("Введите конец диапазона")
    else:
        print("Выбран режим: за всё время")

    # Запускаем нормализацию исходных SIEM выгрузок.
    _run([sys.executable, str(_script_path(base_dir, "python_script.py")), "--data", str(data_dir), "--work", str(work_dir)])
    # Строим признаки из суточных событий.
    _run([sys.executable, str(_script_path(base_dir, "build_features_v2.py")), "--work", str(work_dir), "--features-dir", str(features_dir)])
    # Очищаем признаки и приводим их к числовому виду.
    _run([sys.executable, str(_script_path(base_dir, "preprocess_features.py")), "--work", str(features_dir)])

    # Загружаем очищенные признаки для определения доступных дат.
    users = core.load_features(features_dir, "users")
    hosts = core.load_features(features_dir, "hosts")
    available = core.available_dates(users, hosts)
    if not available:
        raise ValueError("Не найдено доступных дат в таблицах признаков.")

    # Подбираем список дат для расчёта аномалий.
    if scope in {"day", "week", "month"}:
        # Если дата не указана — берём последнюю.
        if not target:
            target = core.pick_latest_date(users, hosts)
        # Проверяем, что дата есть в доступных.
        if target not in available:
            raise ValueError(f"Дата {target} отсутствует в доступных данных.")
        dates_for_anomalies = _window_dates(available, target, scope)
    elif scope == "range":
        # Проверяем корректность порядка дат.
        if range_start > range_end:
            raise ValueError("Начальная дата не может быть больше конечной.")
        dates_for_anomalies = _filter_dates(available, range_start, range_end)
        if not dates_for_anomalies:
            raise ValueError("Нет данных внутри заданного диапазона.")
        # Для диапазона целевая дата — последняя в диапазоне (для согласованности логов).
        target = dates_for_anomalies[-1]
    else:
        # Для "всё время" берём все доступные даты.
        dates_for_anomalies = available
        target = available[-1]

    # Запускаем этапы обучения, объяснения и формирования отчётов.
    _run_reports(
        base_dir=base_dir,
        work_dir=work_dir,
        features_dir=features_dir,
        report_dir=report_dir,
        scope=scope,
        target=target,
        dates_for_anomalies=dates_for_anomalies,
    )

    # Сигнализируем об успешном завершении.
    print("\nГотово: полный цикл выполнен успешно.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
