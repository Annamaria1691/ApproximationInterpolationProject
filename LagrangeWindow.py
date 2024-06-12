import sys

import numpy as np
import sympy as sp
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QLineEdit, QMessageBox, \
    QApplication
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class LagrangeWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Interpolare și Aproximare cu polinoame Lagrange')
        self.setGeometry(150, 150, 1300, 600)
        self.initUI()

    def initUI(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("background-color: #eefafb;")
        main_layout = QHBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(10)
        left_layout = QVBoxLayout()
        main_layout.addLayout(left_layout)
        title = QLabel('Interpolare și Aproximare cu polinoame Lagrange', self)
        title.setFont(QFont('Roboto', 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("background-color: #eefafb;")
        left_layout.addWidget(title)
        self.inputXValues = QLineEdit(self)
        self.inputXValues.setPlaceholderText('Introduceti valorile lui x (separate de virgula)')
        self.inputXValues.setFont(QFont('Arial', 14))
        left_layout.addWidget(self.inputXValues)
        self.inputYValues = QLineEdit(self)
        self.inputYValues.setPlaceholderText('Introduceti valorile lui y (separate de virgula)')
        self.inputYValues.setFont(QFont('Arial', 14))
        left_layout.addWidget(self.inputYValues)
        self.buttonGenerateInterpolare = QPushButton('Genereaza Interpolarea Lagrange', self)
        self.buttonGenerateInterpolare.setFont(QFont('Arial', 14))
        self.buttonGenerateInterpolare.setStyleSheet("""
                            QPushButton {
                                background-color: #B3E5FC;
                                border-radius: 10px;
                                padding: 15px;
                                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
                                border: 1px solid #ccc;
                                color:#35403c;
                            }
                            QPushButton:hover {
                                background-color: #81D4FA;
                            }
                            QPushButton:pressed {
                                background-color: #4FC3F7;
                                border-style: inset;
                            }
                        """)
        self.buttonGenerateInterpolare.clicked.connect(self.plot_lagrange_interpolation)
        left_layout.addWidget(self.buttonGenerateInterpolare)
        self.resultingFunctionLabel = QLabel(self)
        self.resultingFunctionLabel.setFont(QFont('Arial', 14))
        self.resultingFunctionLabel.setStyleSheet("background-color: #eefafb;")
        left_layout.addWidget(self.resultingFunctionLabel)
        self.inputXValuesApprox = QLineEdit(self)
        self.inputXValuesApprox.setPlaceholderText('Introduceri numarul de puncte pentru interpolare')
        self.inputXValuesApprox.setFont(QFont('Arial', 14))
        left_layout.addWidget(self.inputXValuesApprox)
        self.buttonGenerateAproximare = QPushButton('Genereaza Punctele de Interpolare', self)
        self.buttonGenerateAproximare.setFont(QFont('Arial', 14))
        self.buttonGenerateAproximare.setStyleSheet("""
                            QPushButton {
                                background-color: #B3E5FC;
                                border-radius: 10px;
                                padding: 15px;
                                box-shadow: 2px 2px 5px rgba(0, 0, 0, 0.3);
                                border: 1px solid #ccc;
                                color:#35403c;
                            }
                            QPushButton:hover {
                                background-color: #81D4FA;
                            }
                            QPushButton:pressed {
                                background-color: #4FC3F7;
                                border-style: inset;
                            }
                        """)
        self.buttonGenerateAproximare.clicked.connect(self.animate_lagrange)
        left_layout.addWidget(self.buttonGenerateAproximare)
        self.canvas = MplCanvas(self, width=12, height=10, dpi=100)
        self.canvas.setFixedSize(700, 600)
        main_layout.addWidget(self.canvas)

    def plot_lagrange_interpolation(self):
        x_values = self.inputXValues.text()
        y_values = self.inputYValues.text()
        try:
            x = np.array([float(val) for val in x_values.split(',')])
            y = np.array([float(val) for val in y_values.split(',')])
        except ValueError:
            QMessageBox.warning(self, 'Input Error', 'Introduceti numere valide pentru x si y')
            return
        if len(x) != len(y):
            QMessageBox.warning(self, 'Input Error', 'Introduceti o cifra pentru una din coordonate')
            return
        t = np.linspace(min(x), max(x), 1000)
        lagrange_poly = self.lagrange_interpolation(x, y, t)
        self.display_lagrange_function(x, y)
        self.canvas.axes.clear()
        self.canvas.axes.plot(t, lagrange_poly, label='Iterpolarea Lagrange')
        self.canvas.axes.scatter(x, y, color='red', label='Punctele Initiale')
        self.canvas.axes.legend()
        self.canvas.axes.set_title('Interpolare Lagrange')
        self.canvas.draw()

    def plot_interpolated_points(self):
        x_values = self.inputXValues.text()
        y_values = self.inputYValues.text()
        x_values_approx = self.inputXValuesApprox.text()
        try:
            x = np.array([float(val) for val in x_values.split(',')])
            y = np.array([float(val) for val in y_values.split(',')])
            num_points = int(
                x_values_approx) if x_values_approx.strip() else 100  # Default to 100 points if not provided
        except ValueError:
            QMessageBox.warning(self, 'Input Error', 'Introduceti valori numerice valide')
            return
        if len(x) != len(y):
            QMessageBox.warning(self, 'Input Error', 'Introduceti o cifra pentru una din coordonate')
            return
        t = np.linspace(min(x), max(x), num_points)
        lagrange_poly = self.lagrange_interpolation(x, y, t)
        self.canvas.axes.clear()
        self.canvas.axes.plot(t, lagrange_poly, label='Aproximare Lagrange')
        self.canvas.axes.scatter(x, y, color='red', label='Puncte Initiale')
        self.canvas.axes.scatter(t, lagrange_poly, color='blue', marker='x', label='Puncte Interpolate')
        self.canvas.axes.legend()
        self.canvas.axes.set_title('Aproximare Lagrange')
        self.canvas.draw()

    def lagrange_interpolation(self, x, y, t):
        x_sym = sp.symbols('x')
        L = 0
        for i in range(len(x)):
            term = y[i]
            for j in range(len(x)):
                if i != j:
                    term *= (x_sym - x[j]) / (x[i] - x[j])
            L += term
        lagrange_poly_func = sp.lambdify(x_sym, L, modules='numpy')
        return lagrange_poly_func(t)

    def display_lagrange_function(self, x, y):
        x_sym = sp.symbols('x')
        L = 0
        for i in range(len(x)):
            term = y[i]
            for j in range(len(x)):
                if i != j:
                    term *= (x_sym - x[j]) / (x[i] - x[j])
            L += term
        L_simplified = sp.simplify(L)
        L_rounded = sp.N(L_simplified, 2)
        self.resultingFunctionLabel.setText(f"f(x) = {L_rounded}")

    def animate_lagrange(self):
        if not self.validate_inputs():
            return
        x_values = self.inputXValues.text()
        y_values = self.inputYValues.text()
        x_values_approx = self.inputXValuesApprox.text()
        try:
            self.x = np.array([float(val) for val in x_values.split(',')])
            self.y = np.array([float(val) for val in y_values.split(',')])
            self.num_points = int(
                x_values_approx) if x_values_approx.strip() else 100  # Default to 100 points if not provided
        except ValueError:
            QMessageBox.warning(self, 'Input Error', 'Introduceti valori numerice valide')
            return
        if len(self.x) != len(self.y):
            QMessageBox.warning(self, 'Input Error', 'Introduceti o cifra pentru una din coordonate')
            return
        self.t = np.linspace(min(self.x), max(self.x), self.num_points)
        self.current_point_index = 0
        self.timer = QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start()

    def update_animation(self):
        if self.current_point_index >= len(self.t):
            self.timer.stop()
            return
        lagrange_poly = self.lagrange_interpolation(self.x, self.y, self.t[:self.current_point_index + 1])
        self.canvas.axes.clear()
        self.canvas.axes.plot(self.t[:self.current_point_index + 1], lagrange_poly, label='Aproximare Lagrange')
        self.canvas.axes.scatter(self.x, self.y, color='red', label='Puncte Initiale')
        self.canvas.axes.scatter(self.t[:self.current_point_index + 1], lagrange_poly, color='blue', marker='x',
                                 label='Puncte Interpolate')
        self.canvas.axes.legend()
        self.canvas.axes.set_title('Aproximare Lagrange')
        self.canvas.draw()
        self.current_point_index += 1

    def validate_inputs(self):
        if not self.inputXValues.text().strip():
            QMessageBox.warning(self, 'Input Error', 'Introduceti valori pentru x')
            return False
        if not self.inputYValues.text().strip():
            QMessageBox.warning(self, 'Input Error', 'Introduceti valori pentru y')
            return False
        return True


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=12, height=10, dpi=100):
        fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = LagrangeWindow()
    window.show()
    sys.exit(app.exec_())
