# Repetit_ru

# Промышленность

[Блокнот](https://github.com/qqaazz112211/Repetit_ru/blob/main/Repetit_ru.ipynb)

## Заказчик
Сервис по подбору репетиторов Repetit.ru

## Описание проекта

Сервис Repetit.ru работает с большим количеством заявок от клиентов с данными о предмете, желаемой стоимости, возрасте ученика, целью занятий и тд. К сожалению, 7 из 8 не доходят до оплаты, при этом обработка заявки консультантом увеличивает конверсию в оплату на 30%. Проблема в том, что консультантов не хватает на все заявки и получается, что чем больше заявок — тем меньше конверсия из заявки в оплату и консультанты тратят время на бесперспективные заявки.

## Задачи
Разработать модель, которая по имеющейся информации о клиенте и заявке будет предсказывать вероятность оплаты заявки клиентом. Заказчик хочет понять, какие заявки будут оплачены, а какие нет, чтобы одни обрабатывать вручную консультантами, а другие нет. Оценка качества модели будет производиться с использованием precision и ROC-AUC.


## Библиотеки и инструменты

import pandas as pd  
import re  
import plotly.express as px  
import matplotlib.dates as mdates  
import numpy as np  
import pandas as pd  
import matplotlib.pyplot as plt  
import seaborn as sns  
from sklearn.model_selection import train_test_split  
from sklearn.preprocessing import StandardScaler  
from sklearn.utils import shuffle  
from catboost import CatBoostClassifier, Pool  
from sklearn.metrics import precision_score, f1_score, roc_auc_score  
from catboost import CatBoostClassifier  
from sklearn.model_selection import GridSearchCV  
from sklearn.metrics import precision_score, f1_score, roc_auc_score, make_scorer  
from gensim.models import Word2Vec  
from sklearn.metrics import precision_score, recall_score, f1_score, roc_curve, auc, confusion_matrix  
from sklearn.calibration import CalibratedClassifierCV  



## Вывод

По условию заказчика была разработана модель которая по имеющейся информации о клиенте и заявке будет предсказывать вероятность оплаты заявки клиентом. Для достижения поставленной цели были выполнены следущие шаги:

Разведочный анализ (EDA)
Данный раздел включал в себя первичный осмотр предоставленных данных с дальнейшей подготовкой к построению модели
Моделирование
На данном этапе осуществлялась финальная подготовка данных (стандартизация, векторизация), определялись оптимальные гипперпараметры модели
Построение и обучкение модели
Калибровка вероятностей
Каждый шаг сопровождался комментариями и общими выводами. В результате удалось добиться следующих результатов выбранных метрик заказчика:

До калибровки вероятностей:  
Precision для класса 0: 0.8553965473120608  
Precision для класса 1: 0.627294493216281  
Recall для класса 0: 0.9946553440836834  
Recall для класса 1: 0.05078175474867554  
F1-score для класса 0: 0.9197847379867604  
F1-score для класса 1: 0.09395732472655549  
ROC AUC для класса 1: 0.7215379999003063  

Результаты после калибровки вероятностей:  
Precision по классам: [0.86042433 0.55980861]  
Recall по классам: [0.98737492 0.09110967]  
F1 Score по классам: [0.91953864 0.15671392]  
ROC AUC по классам: 0.7182046406550011  
