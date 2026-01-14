# explain_anomalies.py

## Назначение
Формирует объяснения для аномалий по признакам.

## Основные функции и переменные
- `_explain_one_kind()` — вычисляет топ-признаки отклонений для users/hosts.
- `FEATURE_HINTS` — справочник причин/рекомендаций.

## Входные данные
- `features_users_clean.csv`, `features_hosts_clean.csv`.
- `anomalies_users_YYYY-MM-DD.csv`, `anomalies_hosts_YYYY-MM-DD.csv`.
- Аргументы: `--work`, `--date`, `--top-features`, `--anomaly-dir`.

## Выходные данные
- `anomalies_users_YYYY-MM-DD_explain.csv`.
- `anomalies_hosts_YYYY-MM-DD_explain.csv`.
- `anomalies_all_YYYY-MM-DD_explain.csv`.
