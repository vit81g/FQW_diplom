# auto_generate_reports.py

## Назначение
Запускает генерацию day/week/month отчётов одной командой.

## Основные функции и переменные
- `_run_one()` — построение отчёта для одного масштаба.
- `main()` — CLI и последовательный запуск day/week/month.

## Входные данные
- `features_users_clean.csv`, `features_hosts_clean.csv`.
- Аргументы: `--work`, `--date`, `--top-pct`.

## Выходные данные
- Папки `work/reports/day_<date>/`, `work/reports/week_<date>/`, `work/reports/month_<date>/`.
