import io
import sys
from typing import List

import pandas as pd
import statsmodels.api as sm
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from statsmodels.regression.linear_model import RegressionResults

import main_window
import utils
from generation_regressors import stepen_x, x_log, x_pow
from main import forward_selection, Backward_Elimination, Stepwise, regression_model_fit


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        self.fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = self.fig.add_subplot(111)
        self.axes.set_xlabel('Y наблюдение')
        self.axes.set_ylabel('Y прогноз')

        super(MplCanvas, self).__init__(self.fig)


class App(QtWidgets.QMainWindow, main_window.Ui_MainWindow):
    DIRECT = 0
    REVERSE = 1
    INCLUDE = 2

    CHOICES_FOR_ALGORITHM = {
        DIRECT: 'Прямой отбор',
        REVERSE: 'Обратное исключение',
        INCLUDE: 'Включение и исключение',
    }

    REGRESSOR_TYPES = ['x^2']

    ALPHA_NUMBERS = ['0.001', '0.01', '0.05']

    FUNC_MAP = {
        DIRECT: forward_selection,
        REVERSE: Backward_Elimination,
        INCLUDE: Stepwise,
    }

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.set_values_for_combo_box_from_dict(self.cb_algorithm, self.CHOICES_FOR_ALGORITHM)
        self.set_values_for_combo_box_from_list(self.cb_alpha, self.ALPHA_NUMBERS)

        self.pb_build_model.setEnabled(False)
        self.pb_build_model.clicked.connect(self.build_model)
        self.pb_clear_graph.clicked.connect(self.clear_graphics)

        self.action_open_file.triggered.connect(self.load_data_from_csv_file)
        self.action_export.triggered.connect(self.export_data)

        self.data = None
        self.last_resolve = {}
        # Построение графика
        self.sc = MplCanvas(self, width=5, height=4, dpi=100)
        self.sc.axes.plot([], [])
        self.vl_graph.addWidget(self.sc)

        toolbar = NavigationToolbar(self.sc, self)
        self.vl_graph.addWidget(toolbar)
        # Окончание построение графика

        self.quality_data = None
        self.params_data = None

    def clear_graphics(self):
        self.sc.axes.clear()
        self.sc.axes.set_xlabel('Y наблюдение')
        self.sc.axes.set_ylabel('Y прогноз')
        self.sc.draw()

    def export_data(self):
        filename = self.show_save_file_dialog(filter='*.xlsx')[0]
        if not filename:
            return
        excel_builder = utils.ExcelBuilder()
        excel_builder.add_table(self.params_data[0], self.params_data[1])
        excel_builder.add_table(['Название', 'Значение'], self.quality_data)
        buf = io.BytesIO()
        self.sc.fig.savefig(buf, format='png')
        excel_builder.add_image(buf)
        excel_builder.save(filename)

    def load_data_from_csv_file(self) -> None:
        filename = self.show_file_dialog(filter='*.csv')
        if not filename:
            return
        try:
            self.data = pd.read_csv(
                filename,
                sep=';',
                decimal='.'
            )
        except FileNotFoundError:
            pass

        self.set_values_for_combo_box_from_list(self.cb_target_variable, self.data.columns.values)
        self.pb_build_model.setEnabled(True)

    def show_file_dialog(self, **kwargs):
        filename = QtWidgets.QFileDialog.getOpenFileName(self, 'Открыть файл', **kwargs)[0]
        return filename

    def show_save_file_dialog(self, **kwargs):
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Выберите файл', **kwargs)
        return filename

    # noinspection PyMethodMayBeStatic
    def set_values_for_combo_box_from_dict(self, combo_box: QtWidgets.QComboBox, map_items: dict) -> None:
        combo_box.clear()
        for k, v in map_items.items():
            combo_box.addItem(v, k)

    # noinspection PyMethodMayBeStatic
    def set_values_for_combo_box_from_list(self, combo_box: QtWidgets.QComboBox, items: list) -> None:
        combo_box.clear()
        combo_box.addItems(items)

    # noinspection PyMethodMayBeStatic
    def set_values_for_list_view(self, list_view: QtWidgets.QListWidget, items: list) -> None:
        list_view.clear()
        list_view.addItems(items)

    def build_model(self):
        func = self.get_function_on_type(self.cb_algorithm.currentData())
        alpha = float(self.cb_alpha.currentText())
        target_variable = self.cb_target_variable.currentText()

        # Генерация регрессоров
        print(target_variable)
        oldRegressors = self.data.copy().drop(target_variable, 1)
        generatedData = self.data.copy()

        if self.cb_x2.isChecked():
            data_x_square = x_pow(oldRegressors, 2)
            print(data_x_square)
            generatedData = generatedData.join(data_x_square)

        if self.cb_1x.isChecked():
            data_x_cub = x_pow(oldRegressors, 3)
            generatedData = generatedData.join(data_x_cub)

        if self.cb_lnx.isChecked():
            data_x_log = x_log(oldRegressors)
            generatedData = generatedData.join(data_x_log)

        if self.cb_ex.isChecked():
            data_x_exp = stepen_x(oldRegressors)
            generatedData = generatedData.join(data_x_exp)

        results = func(alpha, generatedData, target_variable)

        # Построение регрессии
        if results:
            linmodel = regression_model_fit(results, generatedData, target_variable)
            y_pred = linmodel.predict(generatedData[results].values.reshape(-1, len(results)))

            # Результаты регрессионного анализа
            factors_metrics = {}

            factors_metrics['R^2'] = linmodel.score(
                generatedData[results].values.reshape(-1, len(results)),
                generatedData[target_variable].values.reshape(-1, 1)
            )

            factors_metrics['regression_a0'] = linmodel.intercept_[0]

            # Регрессия из либы
            X = generatedData[results]
            Y = generatedData[target_variable]
            X = sm.add_constant(X)

            model = sm.OLS(Y, X).fit()

            print(model.summary())

            print(model.rsquared)

            print(linmodel.coef_)

            for num, regressor in enumerate(results):
                factors_metrics[regressor] = (
                    {'b-coef': 1,
                     'regression_coef': linmodel.coef_[0][num],
                     'corr_with_y': generatedData.corr()[regressor][target_variable],
                     }
                )

            print(factors_metrics)

            if max(y_pred) > max(generatedData[target_variable]):
                xmax = max(y_pred)
            else:
                xmax = max(generatedData[target_variable])

            if min(y_pred) < min(generatedData[target_variable]):
                xmin = min(y_pred)
            else:
                xmin = min(generatedData[target_variable])

            generatedData.to_csv(r'generated_data.csv', index=False, sep = ";")

            self.sc.axes.scatter(generatedData[target_variable], y_pred, s = 7)
            self.sc.axes.plot([xmin, xmax], [xmin, xmax])

            self.sc.draw()

            self.set_values_for_list_view(self.lw_results, results)

            quality_data = self.build_quality_table(model)
            self.writing_table(self.table_2,
                               headers=['Название', 'Значение'],
                               rows=quality_data)
            params_data = self.build_results_analyze(model, factors_metrics)
            self.writing_table(self.table_1, params_data[0], params_data[1])

            self.quality_data = quality_data
            self.params_data = params_data

        else:
            self.set_values_for_list_view(self.lw_results, ['Нет значимых переменных'])

        self.last_resolve = {
            'alpha': alpha,
            'target_variable': target_variable,
            'results': results,
            'algorithm': self.cb_algorithm.currentText(),
        }
        self.action_export.setEnabled(True)

    def get_function_on_type(self, func_type: int):
        return self.FUNC_MAP[func_type]

    # noinspection PyMethodMayBeStatic
    def build_quality_table(self, model: RegressionResults):
        rows = [
            ['Коэффицент детерминации', "{:.4f}".format(model.rsquared)],
            ['F статистика', "{:.4f}".format(model.fvalue)],
            ['Число степеней свободы', model.df_model]
        ]
        return rows

    # noinspection PyMethodMayBeStatic
    def build_results_analyze(self, model: RegressionResults, metrics: dict):
        headers = ['Факторы', 'Коэффициент регрессии', 't-критерий', 'P>|t|', 'Корреляция с Y']
        rows = []
        for i in range(len(model.params)):
            regressor = model.params.axes[0].values[i]
            rows.append([
                regressor,
                "{:.4f}".format(model.params[i]),
                "{:.4f}".format(model.tvalues[i]),
                "{:.4f}".format(model.pvalues[i]),
                "{:.4f}".format(metrics[regressor]['corr_with_y']) if regressor != 'const' else '-',

            ])
        return headers, rows

    # noinspection PyMethodMayBeStatic
    def writing_table(self, table: QtWidgets.QTableWidget, headers: list, rows: List[list]):
        table.clear()
        table.setColumnCount(len(headers))
        table.setRowCount(len(rows))

        table.setHorizontalHeaderLabels(headers)
        for i, row in enumerate(rows):
            for j, column in enumerate(row):
                table.setItem(i, j, QtWidgets.QTableWidgetItem(str(column)))
        table.resizeRowsToContents()


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    app.exec_()
