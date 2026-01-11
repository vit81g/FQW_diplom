# viz_core.py

## Назначение
Содержит функции загрузки признаков, расчёта скорингов и построения графиков.

## Основные функции и переменные
- `load_features()` — загрузка `features_*_clean.csv`.
- `score_day()` — расчёт скоринга аномалий для заданной даты.
- `build_trend()` — тренд по датам.
- `save_top_bar()`, `save_severity_pie()`, `save_trend_line()`, `save_trend_stacked()` — построение графиков.
- `SEV_RU`, `SEV_COLORS` — подписи и цвета уровней серьёзности.

## Входные данные
- `features_users_clean.csv`, `features_hosts_clean.csv`.

## Выходные данные
- PNG-графики и CSV-резюме в директориях отчётов.
