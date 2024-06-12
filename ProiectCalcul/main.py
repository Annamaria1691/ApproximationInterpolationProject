import sys

import matplotlib.pyplot as plt
import numpy as np
from PyQt5.QtCore import Qt, QTimer
from PyQt5.QtGui import QFont
from PyQt5.QtWidgets import QApplication, QMainWindow, QVBoxLayout, QWidget, QPushButton, QLabel, QHBoxLayout
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.collections import LineCollection

from BernsteinWindow import BernsteinWindow
from LagrangeWindow import LagrangeWindow


class MainApp(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle('Calculator - Bernstein & Lagrange')
        self.setGeometry(100, 100, 800, 470)
        self.initUI()

    def initUI(self):
        central_widget = QWidget(self)
        self.setCentralWidget(central_widget)
        central_widget.setStyleSheet("background-color: #eefafb;")
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(10)
        title_layout = QHBoxLayout()
        title_layout.setAlignment(Qt.AlignCenter)
        title = QLabel('Metode de Aproximare și Interpolare', self)
        title.setFont(QFont('Roboto', 20, QFont.Bold))
        title.setAlignment(Qt.AlignCenter)
        title.setStyleSheet("background-color: #eefafb;")
        main_layout.addWidget(title)
        main_layout.addLayout(title_layout)
        self.canvas = MplCanvas(self, width=5, height=3, dpi=100)
        self.canvas.setStyleSheet("background-color: #E0F7FA;")
        main_layout.addWidget(self.canvas)
        main_layout.addSpacing(7)
        self.timer = QTimer()
        self.timer.setInterval(50)
        self.timer.timeout.connect(self.update_plot)
        self.timer.start()
        button_layout = QHBoxLayout()
        self.buttonBernstein = QPushButton('Aproximare Bernstein', self)
        self.buttonBernstein.setFont(QFont('Arial', 15))
        self.buttonBernstein.setStyleSheet("""
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
        self.buttonBernstein.clicked.connect(self.open_bernstein_window)
        button_layout.addWidget(self.buttonBernstein)
        self.buttonLagrange = QPushButton('Interpolare Lagrange', self)
        self.buttonLagrange.setFont(QFont('Arial', 15))
        self.buttonLagrange.setStyleSheet("""
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
        self.buttonLagrange.clicked.connect(self.open_lagrange_window)
        button_layout.addWidget(self.buttonLagrange)
        main_layout.addLayout(button_layout)
        self.xdata = np.linspace(0, 2 * np.pi, 100)
        self.phase = 0
        self.ydata = np.sin(self.xdata + self.phase)
        self.line, = self.canvas.axes.plot(self.xdata, self.ydata)

    def update_plot(self):
        self.phase += 0.1
        self.ydata = np.sin(self.xdata + self.phase)
        points = np.array([self.xdata, self.ydata]).T.reshape(-1, 1, 2)
        segments = np.concatenate([points[:-1], points[1:]], axis=1)
        lc = LineCollection(segments, cmap='rainbow', norm=plt.Normalize(0, 2 * np.pi))
        lc.set_array(self.xdata)
        lc.set_linewidth(10)
        self.canvas.axes.clear()
        self.canvas.axes.add_collection(lc)
        self.canvas.axes.set_xlim(0, 2 * np.pi)
        self.canvas.axes.set_ylim(-1.5, 1.5)
        self.canvas.draw()

    def open_bernstein_window(self):
        self.bernstein_window = BernsteinWindow()
        self.bernstein_window.show()

    def open_lagrange_window(self):
        self.lagrange_window = LagrangeWindow()
        self.lagrange_window.show()


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig, self.axes = plt.subplots(figsize=(width, height), dpi=dpi)
        super().__init__(fig)
        self.setParent(parent)
        self.axes.set_xlim(0, 2 * np.pi)
        self.axes.set_ylim(-1.5, 1.5)
        self.axes.set_facecolor('#f5f5ec')


if __name__ == '__main__':
    app = QApplication(sys.argv)
    main_app = MainApp()
    main_app.show()
    sys.exit(app.exec_())
