# soc_report.py

## Назначение
Формирует SOC-отчёт по аномалиям с дополнительным контекстом из суточных событий.

## Основные функции и переменные
- `_context_for_anomalies()` — собирает контекст для каждой аномалии.
- `_summarize_events()` — извлекает время, программы, пользователей, хосты и адреса.
- `_build_date_window()` — формирует список дат для day/week/month.

## Входные данные
- `anomalies_all_YYYY-MM-DD_explain.csv` (в `--anomaly-dir`).
- Суточные CSV событий из `python_script.py` в `--work`.
- (Опционально) `user_mapping.csv` для расшифровки userXXX.
- Аргументы: `--work`, `--anomaly-dir`, `--report-dir`, `--scope`, `--date`.

## Выходные данные
- Markdown-отчёт `soc_report_<scope>_<date>.md` в `report/` (по умолчанию внутри `--work`).
- Контекстная таблица `anomaly_context_<scope>_<date>.csv` в `anomaly/`.
