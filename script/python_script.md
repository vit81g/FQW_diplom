# python_script.py

## Назначение
Нормализует выгрузки SIEM/PAN, удаляет служебные колонки, анонимизирует пользователей и
разбивает события по датам.

## Основные функции, классы и переменные
- `FileRole` — dataclass для роли входного файла (user/host/unknown).
- `detect_file_role()` и `extract_owner_from_filename()` — определение типа сущности и владельца.
- `normalize_file()` — чтение файла, нормализация времени и разбиение на даты.
- `ensure_user_map_for_df()` и `apply_user_map()` — анонимизация пользователей.
- `DROP_COLS`, `TIME_COL_CANDIDATES`, `USER_COL_HINTS` — ключевые константы.

## Входные данные
- Папка `data/` с файлами `*.tsv`/`*.txt`.
- Аргументы: `--data`, `--work`, `--chunksize`.

## Выходные данные
- Суточные CSV-файлы в `work/`.
- Файл сопоставления `user_mapping.csv`.
