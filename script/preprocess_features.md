# preprocess_features.py

## Назначение
Очищает таблицы признаков, приводит даты к формату `YYYY-MM-DD`, преобразует признаки к
числовому виду и заполняет пропуски.

## Основные функции и переменные
- `clean_features()` — очистка и нормализация таблиц признаков.
- `main()` — парсинг аргументов и сохранение `features_*_clean.csv`.
- `ID_COLS` — идентификаторы сущности и даты.

## Входные данные
- `features_users.csv`, `features_hosts.csv` из `work/`.
- Аргументы: `--users`, `--hosts`, `--out-dir` или `--work`.

## Выходные данные
- `features_users_clean.csv`, `features_hosts_clean.csv` в `work/` или `--out-dir`.
