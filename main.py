import numpy as np
import pandas as pd
import scipy.stats as stats
from sklearn.linear_model import LinearRegression

# Загрузка данных
data = pd.read_csv('Тест2.csv',
                   sep=';', encoding='windows-1251')

# Переменные
Model_Full = []
n = len(data)
k = 0
alpha = 0.05
df = n - k - 2

# Корреляция
corrmat = data.corr()
target = 'Prosrochki'

# Выбор самой коррелируемой входной переменной с Y
X_extra = corrmat[target].drop(target).map(abs).idxmax()

# Регрессионная модель с переменной X_extra
linmodel = LinearRegression()
linmodel.fit(data[X_extra].values.reshape(-1, 1), data[target].values.reshape(-1, 1))
y_pred = linmodel.predict(data[X_extra].values.reshape(-1, 1))

# Распределение Фишера
F_table = stats.f.ppf(1 - alpha, 1, df)

# Сумма квадратов регрессии
y_mean = data[target].mean()
SSR_extra = sum((y_pred - y_mean) ** 2)

# Гамма статистика
SSE_extra = sum((data[target].values.reshape(-1, 1) - y_pred) ** 2)
MSE_full = SSE_extra / df
F_real = SSR_extra / MSE_full

if F_real > F_table:
    Model_Full.append(X_extra)

    pretendents = []
    for value in data.columns.values:
        if value != target and Model_Full[0] != value:
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
        F_table = stats.f.ppf(1 - alpha, 1, df)

        for pretindent in pretendents:
            linmodel = LinearRegression()
            Model_Extra = Model_Full.copy()
            Model_Extra.append(pretindent)

            fittedModel = data[Model_Extra].values.reshape(-1, len(Model_Extra))
            linmodel.fit(fittedModel, data[target].values.reshape(-1, 1))
            y_pred = linmodel.predict(data[Model_Extra].values.reshape(-1, len(Model_Extra)))

            # Игрик с домиком минус Игрик среднее
            y_mean = data[target].mean()
            SSR_full = sum((y_pred - y_mean) ** 2)

            # Гамма статистика
            SSE_extra = sum((data[target].values.reshape(-1, 1) - y_pred) ** 2)

            MSE_full = SSE_extra / df
            SSR_extra = SSR_full-SSR_initial
            F_real = (SSR_extra) / MSE_full

            if F_real > F_buf:
                SSR_initial = SSR_full
                F_buf = F_real
                Pretend_buf = pretindent

        if F_buf > F_table:
            pretendents.remove(Pretend_buf)
            Model_Full.append(Pretend_buf)
        else:
            break

    print(Model_Full)

else:
    print('Нет значимых переменных')