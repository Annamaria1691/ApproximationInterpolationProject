import math
import sys

import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QMainWindow, QVBoxLayout, QHBoxLayout, QWidget, QPushButton, QLabel, QLineEdit, QMessageBox, \
    QApplication
from matplotlib import pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas


class BernsteinWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Aproximare cu polinoame Bernstein')
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
        title = QLabel('Aproximare cu polinoame Bernstein', self)
        title.setFont(QFont('Roboto', 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("background-color: #eefafb;")
        left_layout.addWidget(title)
        self.inputFunction = QLineEdit(self)
        self.inputFunction.setPlaceholderText('Introduceti o functie f(x) (e.x.: sin(x),x**2)')
        self.inputFunction.setFont(QFont('Arial', 14))
        left_layout.addWidget(self.inputFunction)
        self.inputDegree = QLineEdit(self)
        self.inputDegree.setPlaceholderText('Introduceti gradul polinomului (e.x.:2,3,6) ')
        self.inputDegree.setFont(QFont('Arial', 14))
        left_layout.addWidget(self.inputDegree)
        self.inputIntervalA = QLineEdit(self)
        self.inputIntervalA.setPlaceholderText('Introduceti valoarea de inceput a intervalului (a)')
        self.inputIntervalA.setFont(QFont('Arial', 14))
        left_layout.addWidget(self.inputIntervalA)
        self.inputIntervalB = QLineEdit(self)
        self.inputIntervalB.setPlaceholderText('Introduceti valoarea de sfarsit a intervalulului (b)')
        self.inputIntervalB.setFont(QFont('Arial', 14))
        left_layout.addWidget(self.inputIntervalB)
        self.buttonGenerate = QPushButton('Genereaza Aproximarea Bernstein', self)
        self.buttonGenerate.setFont(QFont('Arial', 14))
        self.buttonGenerate.setStyleSheet("""
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
        self.buttonGenerate.clicked.connect(self.plot_bernstein)
        left_layout.addWidget(self.buttonGenerate)
        self.canvas = MplCanvas(self, width=12, height=10, dpi=100)
        main_layout.addWidget(self.canvas)
        right_layout = QVBoxLayout()
        main_layout.addLayout(right_layout)
        degrees = [2, 3, 5, 10]
        for degree in degrees:
            button = QPushButton(f'Grad {degree}', self)
            button.setFont(QFont('Arial', 14))
            button.setStyleSheet("""
                QPushButton {
                    background-color: #B3E5FC;
                    border-radius: 10px;
                    padding: 10px;
                    margin: 5px;
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
            button.clicked.connect(lambda _, d=degree: self.plot_bernstein_with_degree(d))
            right_layout.addWidget(button)
        self.buttonAnimate = QPushButton('Animatie', self)
        self.buttonAnimate.setFont(QFont('Arial', 14))
        self.buttonAnimate.setStyleSheet("""
            QPushButton {
                background-color: #B3E5FC;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
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
        self.buttonAnimate.clicked.connect(self.animate_plot)
        right_layout.addWidget(self.buttonAnimate)
        self.buttonAbsoluteError = QPushButton('Eroarea absolută', self)
        self.buttonAbsoluteError.setFont(QFont('Arial', 14))
        self.buttonAbsoluteError.setStyleSheet("""
            QPushButton {
                background-color: #B3E5FC;
                border-radius: 10px;
                padding: 10px;
                margin: 5px;
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
        self.buttonAbsoluteError.clicked.connect(self.show_absolute_error)
        right_layout.addWidget(self.buttonAbsoluteError)
        right_layout.addStretch()

    def plot_bernstein(self):
        if not self.validate_inputs():
            return
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        function_str = self.inputFunction.text()
        degree = int(self.inputDegree.text())
        a = float(self.inputIntervalA.text()) if self.inputIntervalA.text().strip() else 0
        b = float(self.inputIntervalB.text()) if self.inputIntervalB.text().strip() else 1
        if not self.is_continuous(function_str, a, b):
            QMessageBox.warning(self, 'Input Error', 'Funcția nu este continuă.')
            return
        self.calculate_and_plot(function_str, degree)

    def plot_bernstein_with_degree(self, degree):
        if not self.validate_inputs():
            return
        # Stop the animation if it's running
        if hasattr(self, 'timer') and self.timer.isActive():
            self.timer.stop()
        function_str = self.inputFunction.text()
        a = float(self.inputIntervalA.text()) if self.inputIntervalA.text().strip() else 0
        b = float(self.inputIntervalB.text()) if self.inputIntervalB.text().strip() else 1
        if not self.is_continuous(function_str, a, b):
            QMessageBox.warning(self, 'Input Error', 'Funcția nu este continuă.')
            return
        self.calculate_and_plot(function_str, degree)

    def validate_inputs(self):
        if not self.inputFunction.text().strip():
            QMessageBox.warning(self, 'Input Error', 'Introduceti functia lui f(x).')
            return False
        if not self.inputDegree.text().strip():
            QMessageBox.warning(self, 'Input Error', 'Introduceti gradul polinomului.')
            return False
        if not self.inputDegree.text().strip().isdigit() or int(self.inputDegree.text().strip()) <= 0:
            QMessageBox.warning(self, 'Input Error', 'Introduceti o valoare pozitiva pentru gradul polinomului.')
            return False
        return True

    def is_continuous(self, function_str, a, b):
        try:
            func = eval("lambda x: " + self.add_np_prefix(function_str))
            x = np.linspace(a, b, 1000)
            y = func(x)
            differences = np.diff(y)
            if np.any(np.abs(differences) > 1e6):
                return False
            return True
        except Exception as e:
            return False

    def add_np_prefix(self, function_str):
        import re
        function_str = re.sub(r'\b(sin|cos|tan|exp|log|sqrt|pi|abs|arcsin|arccos|arctan)\b', r'np.\1', function_str)
        return function_str

    def calculate_and_plot(self, function_str, degree):
        a = float(self.inputIntervalA.text()) if self.inputIntervalA.text().strip() else 0
        b = float(self.inputIntervalB.text()) if self.inputIntervalB.text().strip() else 1
        num_points = 100
        x = np.linspace(a, b, num_points)
        try:
            func = eval("lambda x: " + self.add_np_prefix(function_str))
            y = func(x)
        except Exception as e:
            QMessageBox.warning(self, 'Evaluation Error', f'Eroare la verificarea functiei: {e}')
            return
        t = np.linspace(a, b, 1000)
        self.bernstein_poly = np.zeros_like(t)
        for k in range(degree + 1):
            binom = math.comb(degree, k)
            self.bernstein_poly += binom * ((t - a) / (b - a)) ** k * ((1 - (t - a) / (b - a)) ** (degree - k)) * func(
                a + k * (b - a) / degree)
        self.original_func_values = func(t)
        self.canvas.axes.clear()
        self.canvas.axes.plot(t, self.original_func_values, label='Functia f(x)', color='red')
        self.canvas.axes.plot(t, self.bernstein_poly, label='Aproximarea Bernstein', color='blue')
        self.canvas.axes.legend()
        self.canvas.axes.set_title('Aproximare Bernstein')
        self.canvas.draw()

    def animate_plot(self):
        if not self.validate_inputs():
            return
        self.current_degree = 1
        self.max_degree = 10
        self.timer = QTimer()
        self.timer.setInterval(500)
        self.timer.timeout.connect(self.update_animation)
        self.timer.start()

    def update_animation(self):
        function_str = self.inputFunction.text()
        if self.current_degree > self.max_degree:
            self.current_degree = 1
        self.calculate_and_plot(function_str, self.current_degree)
        self.current_degree += 1

    def show_absolute_error(self):
        error = np.max(np.abs(self.original_func_values - self.bernstein_poly))
        QMessageBox.information(self, 'Eroarea absolută', f'Eroarea absolută este: {error}')


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=12, height=10, dpi=100):
        fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = BernsteinWindow()
    window.show()
    sys.exit(app.exec_())
