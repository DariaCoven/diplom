import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# Загрузка данных
# data = pd.read_csv('Тест2.csv', sep=';', encoding='windows-1251')

alpha = 0.05


def regression_model(train_model, data, target):
    linmodel = LinearRegression()
    fittedModel = data[train_model].values.reshape(-1, len(train_model))
    linmodel.fit(fittedModel, data[target].values.reshape(-1, 1))
    y_pred = linmodel.predict(data[train_model].values.reshape(-1, len(train_model)))
    return y_pred

def regression_model_fit(train_model, data, target):
    linmodel = LinearRegression()
    fittedModel = data[train_model].values.reshape(-1, len(train_model))
    linmodel.fit(fittedModel, data[target].values.reshape(-1, 1))
    return linmodel


def get_f_table(alpha, df):
    return stats.f.ppf(1 - alpha, 1, df)


def get_f_real(data, target, y_pred, SSR_initial, SSR_full, df):
    SSE_extra = sum((data[target].values.reshape(-1, 1) - y_pred) ** 2)
    MSE_full = SSE_extra / df
    SSR_extra = SSR_full - SSR_initial
    F_real = SSR_extra / MSE_full
    return F_real


def forward_selection(alpha, data, target):
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
    y_pred = regression_model([x_extra], data, target)

    # Распределение Фишера
    F_table = get_f_table(alpha, df)

    # Сумма квадратов регрессии
    y_mean = data[target].mean()
    SSR_extra = sum((y_pred - y_mean) ** 2)

    # Вычисление F-критерия
    F_real = get_f_real(data, target, y_pred, SSR_initial=0, SSR_full=SSR_extra, df=df)

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

            # Todo проверить правильный расчёт К на большей выборке
            k = len(model_full)
            df = n - k - 2

            F_table = get_f_table(alpha, df)

            SSR_initial_buf = 0
            for pretindent in pretendents:
                Model_Extra = model_full.copy()
                Model_Extra.append(pretindent)

                y_pred = regression_model(Model_Extra, data, target)

                SSR_full = sum((y_pred - y_mean) ** 2)

                F_real = get_f_real(data, target, y_pred, SSR_initial, SSR_full, df)

                if F_real > F_buf:
                    SSR_initial_buf = SSR_full
                    F_buf = F_real
                    Pretend_buf = pretindent

            if F_buf > F_table:
                SSR_initial = SSR_initial_buf
                pretendents.remove(Pretend_buf)
                model_full.append(Pretend_buf)
            else:
                break
        return model_full
    else:
        print('Нет значимых переменных')
        return []


def Backward_Elimination(alpha, data, target):
    model_full = data.drop(columns=target).columns.tolist()
    n = len(data)
    y_mean = data[target].mean()

    while len(model_full) != 0:
        F_buf = 0
        Pretend_buf = ""

        k = len(model_full)
        df = n - k - 1

        F_table = get_f_table(alpha, df)

        for pretindent in model_full:

            y_pred = regression_model(model_full, data, target)

            SSR_full = sum((y_pred - y_mean) ** 2)

            model_initial = model_full.copy()
            model_initial.remove(pretindent)

            if len(model_initial) == 0:
                SSR_initial = 0
            else:
                y_pred_initial = regression_model(model_initial, data, target)
                SSR_initial = sum((y_pred_initial - y_mean) ** 2)

            F_real = get_f_real(data, target, y_pred, SSR_initial, SSR_full, df)

            if F_real < F_buf or F_buf == 0:
                F_buf = F_real
                Pretend_buf = pretindent

        if F_buf <= F_table:
            model_full.remove(Pretend_buf)
        else:
            return model_full
    print('Нет значимых переменных')
    return []


def Stepwise(alpha, data, target):
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
    y_pred = regression_model([x_extra], data, target)

    # Распределение Фишера
    F_table = get_f_table(alpha, df)

    # Сумма квадратов регрессии
    y_mean = data[target].mean()
    SSR_extra = sum((y_pred - y_mean) ** 2)

    # Вычисление F-критерия
    F_real = get_f_real(data, target, y_pred, SSR_initial=0, SSR_full=SSR_extra, df=df)

    if F_real > F_table:
        model_full.append(x_extra)

        pretendents = []

        # Добавление переменные которые будем проверять
        for value in data.columns.values:
            if value != target and model_full[0] != value:
                pretendents.append(value)

        SSR_initial = SSR_extra

        while len(pretendents) != 0:
            F_buf = 0
            Pretend_buf = ""

            # Todo проверить правильный расчёт К на большей выборке
            k = len(model_full)
            df = n - k - 2

            F_table = get_f_table(alpha, df)

            SSR_initial_buf = 0
            for pretindent in pretendents:
                Model_Extra = model_full.copy()
                Model_Extra.append(pretindent)

                y_pred = regression_model(Model_Extra, data, target)

                SSR_full = sum((y_pred - y_mean) ** 2)

                F_real = get_f_real(data, target, y_pred, SSR_initial, SSR_full, df)

                if F_real > F_buf:
                    SSR_initial_buf = SSR_full
                    F_buf = F_real
                    Pretend_buf = pretindent

            if F_buf > F_table:
                SSR_initial = SSR_initial_buf
                pretendents.remove(Pretend_buf)
                model_full.append(Pretend_buf)

                # ToDo Запуск исключения

                F_buf1 = 0
                Pretend_buf1 = ""

                k = len(model_full)
                df = n - k - 1

                F_table = get_f_table(alpha, df)

                for pretindent in model_full:

                    y_pred = regression_model(model_full, data, target)

                    SSR_full = sum((y_pred - y_mean) ** 2)

                    model_initial = model_full.copy()
                    model_initial.remove(pretindent)
                    y_pred_initial = regression_model(model_initial, data, target)

                    SSR_initial = sum((y_pred_initial - y_mean) ** 2)

                    F_real = get_f_real(data, target, y_pred, SSR_initial, SSR_full, df)

                    if F_real < F_buf1 or F_buf1 == 0:
                        F_buf1 = F_real
                        Pretend_buf1 = pretindent

                if F_buf1[0] <= F_table:
                    model_full.remove(Pretend_buf1)
                    break

                # ToDO Конец исключения

            else:
                break
        return model_full

    else:
        print('Нет значимых переменных')
        return []

# target = 'Stazh'
#
# print(forward_selection(alpha, data, target))
# print(Backward_Elimination(alpha, data, target))
# print(Stepwise(alpha, data, target))
