pip install -U scikit-learn
pip install pandas
pip install -U matplotlib


Как работает модуль
Вход

Берёт все *.csv в ./work/, кроме:

user_mapping.csv

features_users.csv

features_hosts.csv

Поддерживаемые имена файлов (регистр не важен):

user###_SIEM_YYYY-MM-DD.csv

user###_PAN_YYYY-MM-DD.csv

<hostname>_SIEM_YYYY-MM-DD.csv

<hostname>_PAN_YYYY-MM-DD.csv

Если формат чуть отличается — модуль всё равно пытается вытащить:

SIEM/PAN из имени,

дату как последнее вхождение YYYY-MM-DD,

entity как часть до _SIEM_ / _PAN_.

Выход (в ту же папку work/)

work/features_users.csv

work/features_hosts.csv


Какие признаки формируются (фиксированный набор, без “раздувания” one-hot)

Для каждой сущности (user или host) за сутки строятся числовые признаки отдельно по SIEM и PAN:

SIEM-пакет признаков

siem_events_total

siem_unique_destination_addr

siem_unique_source_process

siem_unique_event_class

siem_unique_category

siem_unique_name

siem_unique_hours

siem_night_share (00:00–05:59)

siem_business_share (09:00–17:59)

PAN-пакет признаков

pan_events_total

pan_unique_destination_addr

pan_unique_event_class

pan_unique_category

pan_unique_name

pan_unique_hours

pan_night_share

pan_business_share

Дополнительно

day_of_week (0=понедельник … 6=воскресенье)

is_weekend

pan_share_of_all_events = pan_events_total / (siem_events_total + pan_events_total)

Для hosts дополнительно

siem_unique_users, pan_unique_users — число уникальных пользователей (по SourceUserName и DestinationUserName, исключая служебные токены и машинные аккаунты ...$).


Запуск

1. 
python_script.py

2. 
python build_features.py --work .\work 

3. 
Из вашей папки проекта (где есть work/):
python preprocess_features.py --work .\work

4. 
python .\train_anomaly_models.py --work .\work

5.
python .\explain_anomalies.py --work .\work

6. 
Отчёт за сутки (по последней дате)
python .\visualize_reports.py --work .\work --scope day --top 20

Отчёт за неделю (7 дней, конец = 2025-12-31)
python .\visualize_reports.py --work .\work --scope week --date 2025-12-31

Отчёт за месяц (30 дней, конец = последняя дата в данных)
python .\visualize_reports.py --work .\work --scope month


Графики будут здесь:

work\reports\day_YYYY-MM-DD\*.png

work\reports\week_YYYY-MM-DD\*.png

work\reports\month_YYYY-MM-DD\*.png

7.
Интерактивно:

python .\visualize_reports.py


Автоматически (сразу day/week/month):

python .\auto_generate_reports.py --work .\work

