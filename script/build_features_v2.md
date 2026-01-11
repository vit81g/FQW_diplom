# build_features_v2.py

## Назначение
Строит суточные числовые признаки для пользователей и хостов на основе CSV из `work/`.

## Основные функции и переменные
- `build_features()` — основной сбор признаков по файлам.
- `_aggregate_one()` — формирование набора признаков для одной сущности/даты.
- `_extract_entity_from_filename()`, `_extract_date_from_filename()` — парсинг имени файла.
- Константы `COL_*` — названия колонок источника SIEM/PAN.

## Входные данные
- Суточные CSV из `work/` (форматы `userXXX_*_YYYY-MM-DD.csv`, `<hostname>_*_YYYY-MM-DD.csv`).
- Аргумент: `--work`.

## Выходные данные
- `features_users.csv` и `features_hosts.csv` в `work/`.
