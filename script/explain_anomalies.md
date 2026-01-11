# explain_anomalies.py

## Назначение
Формирует объяснения для аномалий по признакам и готовит рекомендации для SOC-аналитика.

## Основные функции и переменные
- `_explain_one_kind()` — вычисляет топ-признаки отклонений для users/hosts.
- `_build_soc_recommendations()` — строит причины и рекомендации для SOC.
- `_write_soc_markdown()` — генерирует Markdown-отчёт.
- `FEATURE_HINTS` — справочник причин/рекомендаций.

## Входные данные
- `features_users_clean.csv`, `features_hosts_clean.csv`.
- `anomalies_users_YYYY-MM-DD.csv`, `anomalies_hosts_YYYY-MM-DD.csv`.
- Аргументы: `--work`, `--date`, `--top-features`.

## Выходные данные
- `anomalies_users_YYYY-MM-DD_explain.csv`.
- `anomalies_hosts_YYYY-MM-DD_explain.csv`.
- `anomalies_all_YYYY-MM-DD_explain.csv`.
- `soc_recommendations_YYYY-MM-DD.csv`.
- `soc_recommendations_YYYY-MM-DD.md`.
