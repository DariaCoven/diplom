import sys

import pandas as pd
from PyQt5 import QtWidgets
from matplotlib.backends.backend_qt5agg import (FigureCanvasQTAgg as FigureCanvas,
                                                NavigationToolbar2QT as NavigationToolbar)
from matplotlib.figure import Figure
from matplotlib.pyplot import savefig

import main_window
import utils
from main import forward_selection, Backward_Elimination, Stepwise


class MplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)


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

    ALPHA_NUMBERS = ['0.01', '0.05']

    FUNC_MAP = {
        DIRECT: forward_selection,
        REVERSE: Backward_Elimination,
        INCLUDE: Stepwise,
    }

    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.set_values_for_combo_box_from_dict(self.cb_algorithm, self.CHOICES_FOR_ALGORITHM)
        self.set_values_for_combo_box_from_list(self.cb_regressor_type, self.REGRESSOR_TYPES)
        self.set_values_for_combo_box_from_list(self.cb_alpha, self.ALPHA_NUMBERS)

        self.pb_build_model.setEnabled(False)
        self.pb_build_model.clicked.connect(self.build_model)

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

    def export_data(self):
        filename = self.show_save_file_dialog()[0]
        if not filename:
            return
        
        image_filename = f'{filename.split(".")[0]}.png'
        data = {
            **self.last_resolve,
            'graphic': image_filename
        }

        utils.save_to_file(filename, data)
        self.sc.print_png(image_filename)

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

    def show_save_file_dialog(self):
        filename = QtWidgets.QFileDialog.getSaveFileName(self, 'Выберите файл', filter='*.json')
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

        results = func(alpha, self.data, target_variable)

        if results:
            self.set_values_for_list_view(self.lw_results, results)

        self.last_resolve = {
            'alpha': alpha,
            'target_variable': target_variable,
            'results': results,
            'algorithm': self.cb_algorithm.currentText(),
            'regressor_type': self.cb_regressor_type.currentText()
        }
        self.action_export.setEnabled(True)

    def get_function_on_type(self, func_type: int):
        return self.FUNC_MAP[func_type]


if __name__ == '__main__':
    app = QtWidgets.QApplication(sys.argv)
    window = App()
    window.show()
    app.exec_()
