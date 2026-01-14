# train_anomaly_models.py

## Назначение
Обучает модели Isolation Forest и LOF на исторических днях и вычисляет аномальные
скоринги для выбранной даты.

## Основные функции и переменные
- `run_one()` — основной запуск для users/hosts.
- `_fit_models()` и `_score()` — обучение и расчёт скоринга.
- `_build_report()` — формирование таблицы аномалий.
- `_write_outputs()` — сохранение `anomalies_*_YYYY-MM-DD.csv`.

## Входные данные
- `features_users_clean.csv`, `features_hosts_clean.csv` в `work/`.
- Аргументы: `--work`, `--date`, `--top`, `--out-dir`, параметры моделей.

## Выходные данные
- `anomalies_users_YYYY-MM-DD.csv` и `anomalies_hosts_YYYY-MM-DD.csv` в `anomaly/`
  (по умолчанию внутри `--work`).
