import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# Загрузка данных
data = pd.read_csv('Тест2.csv',
                   sep=';', encoding='windows-1251')

alpha = 0.05


def regression_model(train_model, target):
    linmodel = LinearRegression()
    fittedModel = data[train_model].values.reshape(-1, len(train_model))
    linmodel.fit(fittedModel, data[target].values.reshape(-1, 1))
    y_pred = linmodel.predict(data[train_model].values.reshape(-1, len(train_model)))
    return y_pred


def get_f_table(alpha, df):
    return stats.f.ppf(1 - alpha, 1, df)


def get_f_real(y_pred, SSR_initial, SSR_full, df):
    SSE_extra = sum((data[target].values.reshape(-1, 1) - y_pred) ** 2)
    MSE_full = SSE_extra / df
    SSR_extra = SSR_full - SSR_initial
    F_real = (SSR_extra) / MSE_full
    return F_real


def forward_selection(alpha, target):
    # Переменные
    n = len(data)
    model_full = []
    k = 0
    df = n - k - 2

    # Корреляция
    corrmat = data.corr()

    # Выбор самой коррелируемой входной переменной с Y
    x_extra = corrmat[target].drop(target).map(abs).idxmax()

    # Регрессионная модель с переменной x_extra
    y_pred = regression_model([x_extra], target)

    # Распределение Фишера
    F_table = get_f_table(alpha, df)

    # Сумма квадратов регрессии
    y_mean = data[target].mean()
    SSR_extra = sum((y_pred - y_mean) ** 2)

    # Вычисление F-критерия
    F_real = get_f_real(y_pred, SSR_initial=0, SSR_full=SSR_extra, df=df)

    if F_real > F_table:
        model_full.append(x_extra)

        pretendents = []
        for value in data.columns.values:
            if value != target and model_full[0] != value:
                pretendents.append(value)

        SSR_initial = SSR_extra
        while len(pretendents) != 0:
            F_buf = 0
            Pretend_buf = ""
            # Проверка значимости при помощи F - критерия
            n = len(data)
            k = 1
            alpha = 0.05
            df = n - k - 2

            F_table = get_f_table(alpha, df)

            for pretindent in pretendents:
                Model_Extra = model_full.copy()
                Model_Extra.append(pretindent)

                y_pred = regression_model(Model_Extra, target)

                # Игрик с домиком минус Игрик среднее
                SSR_full = sum((y_pred - y_mean) ** 2)

                F_real = get_f_real(y_pred, SSR_initial, SSR_full, df)

                if F_real > F_buf:
                    SSR_initial = SSR_full
                    F_buf = F_real
                    Pretend_buf = pretindent

            if F_buf > F_table:
                pretendents.remove(Pretend_buf)
                model_full.append(Pretend_buf)
            else:
                break

        print(model_full)

    else:
        print('Нет значимых переменных')


target = 'Prosrochki'
forward_selection(alpha, target)
