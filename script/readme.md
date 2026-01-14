# Документация по скриптам

> Дополнительно для каждого скрипта есть отдельный файл `name_script.md`
> с описанием функций, переменных, входов и выходов.

## Общие зависимости

Установите библиотеки (пример для Windows/PowerShell или cmd):

```bash
pip install -U pandas scikit-learn matplotlib numpy
```

> Примечание: исходные выгрузки лежат в `data/`, суточные CSV — в `work/`.
> Признаки, аномалии и объяснения по умолчанию пишутся в `features/`, а графики/отчёты — в `report/`,
> если иное не указано аргументами командной строки.

---

## python_script.py — нормализация исходных SIEM выгрузок

**Назначение:**
- читает все `*.tsv` и `*.txt` из `script/data/`;
- удаляет служебные колонки (`ClusterID`, `ClusterName`, `TenantID`, `TenantName`);
- анонимизирует пользователей в формат `userXXX`;
- разделяет события по дням и сохраняет в `script/work/`;
- отдельно сохраняет PAN-OS TRAFFIC события;
- формирует таблицу соответствий `user_mapping.csv`.

**Запуск:**
```bash
python python_script.py
python python_script.py --data ./data --work ./work --chunksize 200000
python python_script.py --data ./data --work ./work --chunksize 100000 --max-rows 1000000
```

---

## build_features_v2.py — построение признаков

**Назначение:**
- читает суточные CSV из `work/` (форматы `userXXX_SIEM_YYYY-MM-DD.csv`, `userXXX_PAN_YYYY-MM-DD.csv`,
  `<hostname>_SIEM_YYYY-MM-DD.csv`, `<hostname>_PAN_YYYY-MM-DD.csv`);
- собирает числовые признаки для пользователей и хостов;
- сохраняет `features_users.csv` и `features_hosts.csv` в `features/` (или `--features-dir`).

**Запуск:**
```bash
python build_features_v2.py --work ./work --features-dir ./features
python build_features_v2.py --work ./work --features-dir ./features_alt
```

---

## preprocess_features.py — очистка таблиц признаков

**Назначение:**
- приводит `date` к формату `YYYY-MM-DD`;
- преобразует признаки к числовому типу;
- заполняет пропуски в признаках нулями;
- сохраняет `features_users_clean.csv` и `features_hosts_clean.csv`.

**Запуск:**
```bash
python preprocess_features.py --work ./features
# или
python preprocess_features.py --users ./features/features_users.csv --hosts ./features/features_hosts.csv --out-dir ./features
python preprocess_features.py --work ./features --out-users users_clean.csv --out-hosts hosts_clean.csv
```

---

## train_anomaly_models.py — обучение и скоринг аномалий

**Назначение:**
- обучает Isolation Forest и LOF на всех днях, кроме целевого;
- считает аномальные скоры для выбранной даты;
- сохраняет топ-N аномалий и метаданные в `features/` (или `--out-dir`).

**Запуск:**
```bash
python train_anomaly_models.py --work ./features
python train_anomaly_models.py --work ./features --date 2025-12-17 --top 30
python train_anomaly_models.py --work ./features --out-dir ./features --contamination 0.03 --n-neighbors 25
```

---

## explain_anomalies.py — объяснение причин аномалий

**Назначение:**
- для каждой аномалии определяет 3–5 признаков с наибольшим отклонением от исторической базы;
- присваивает уровень серьёзности по рангу внутри дня;
- сохраняет отчёты для users, hosts и общий файл в `features/` (или `--anomaly-dir`).

**Запуск:**
```bash
python explain_anomalies.py --work ./features --date 2025-12-31
python explain_anomalies.py --work ./features
python explain_anomalies.py --work ./features --anomaly-dir ./features --top-features 5
```

---

## visualize_reports.py — генерация графиков (day/week/month)

**Назначение:**
- строит графики аномалий для суток/недели/месяца;
- сохраняет PNG и CSV в `report/<scope>_<date>/`.

**Запуск:**
```bash
python visualize_reports.py --work ./features --scope day --top-pct 0.05
python visualize_reports.py --work ./features --scope week --date 2025-12-31
python visualize_reports.py --work ./features --scope month
python visualize_reports.py --work ./features --report-dir ./report --contamination 0.03
```

---

## auto_generate_reports.py — полный пакет отчётов

**Назначение:**
- запускает генерацию day/week/month отчётов за одну команду;
- сохраняет результаты в `report/`.

**Запуск:**
```bash
python auto_generate_reports.py --work ./features
# или с долей top-аномалий
python auto_generate_reports.py --work ./features --top-pct 0.05
python auto_generate_reports.py --work ./features --report-dir ./report --date 2025-12-31
```

---

## full_cycle.py — полный цикл обработки и отчётов

**Назначение:**
- выполняет все этапы обработки данных (нормализация → признаки → очистка → аномалии → объяснения → визуализация → SOC-отчёт);
- запрашивает у пользователя масштаб отчёта и даты;
- сохраняет результаты в `work/`, `features/` и `report/`.

**Запуск:**
```bash
python full_cycle.py
```

---

## soc_report.py — SOC-отчёт с контекстом

**Назначение:**
- формирует SOC-отчёт по аномалиям;
- добавляет контекст (программы, пользователи, хосты, время, адреса);
- сохраняет Markdown-отчёт в `report/` (имя с датой/временем), контекстные данные — в `features/`.

**Запуск:**
```bash
python soc_report.py --work ./work --features-dir ./features --scope day --date 2025-12-31
python soc_report.py --work ./work --features-dir ./features --scope week --date 2025-12-31
python soc_report.py --work ./work --features-dir ./features --scope month --date 2025-12-31
python soc_report.py --work ./work --features-dir ./features --report-dir ./report --scope day
```

---

## viz_core.py — общие функции визуализации

**Назначение:**
- содержит функции загрузки признаков, расчёта скорингов и построения графиков;
- используется внутри `visualize_reports.py` и `auto_generate_reports.py`.

**Прямой запуск не требуется.**
