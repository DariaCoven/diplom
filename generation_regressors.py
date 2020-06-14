import pandas as pd
import numpy as np


def x_log(data):
    newdata = {}
    for value in data.columns.values:
        x2 = np.log(data[value])
        newdata[value + ' lnX'] = x2.tolist()

    return pd.DataFrame(newdata)


def x_exp(data):
    newdata = {}
    for value in data.columns.values:
        x2 = np.exp(data[value])
        newdata[value + ' exp^x'] = x2.tolist()

    return pd.DataFrame(newdata)


def x_pow(data, _pow):
    newdata = {}
    for value in data.columns.values:
        xpow = data[value] ** _pow
        newdata[value + ' x^' + str(_pow)] = xpow.tolist()

    return pd.DataFrame(newdata)
