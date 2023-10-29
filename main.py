import sys
from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.QtWebEngineWidgets import QWebEngineView
from MainWindow import Ui_MainWindow
from convert import ConvertWindow
from create_training_data import CreateWindow
from preprocess import PreprocessingWindow
from map import PredictWindow
from post_process import PostProcessWindow
from annot import AntWindow
from output import OutputWindow
from HelpWindow import Ui_HelpWindow


class HelpWindow(QMainWindow, Ui_HelpWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self) 
        self.textEdit.setReadOnly(True)

class MainWindow(QMainWindow, Ui_MainWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)
        self.sub_windows = []
        self.preprocess_window = None
        self.convert_window = None
        self.split_window = None
        self.predict_window = None
        self.annot_window = None
        self.postprocess_window = None
        self.output_window = None
        self.preprocessButton.clicked.connect(self.open_preprocess_window)
        self.convertButton.clicked.connect(self.open_convert_window)
        self.createButton.clicked.connect(self.open_create_window)
        self.predictButton.clicked.connect(self.open_predict_window)
        # self.postprocessButton.clicked.connect(self.open_postprocess_window)
        self.antButton.clicked.connect(self.open_annot_window)
        self.antButton_2.clicked.connect(self.open_annot_window)
        self.outputButton.clicked.connect(self.open_output_window)
        self.outputButton_2.clicked.connect(self.open_output_window)

        self.actionHelp.triggered.connect(self.open_help_window)

    def open_preprocess_window(self):
        self.preprocess_window = PreprocessingWindow()
        self.sub_windows.append(self.preprocess_window)
        self.preprocess_window.show()

    def open_convert_window(self):
        self.convert_window = ConvertWindow()
        self.sub_windows.append(self.convert_window)
        self.convert_window.show()

    def open_create_window(self):
        self.create_window = CreateWindow()
        self.sub_windows.append(self.create_window)
        self.create_window.show()

    def open_predict_window(self):
        self.predict_window = PredictWindow()
        self.sub_windows.append(self.predict_window)
        self.predict_window.show()

    def open_postprocess_window(self):
        self.postprocess_window = PostProcessWindow()
        self.sub_windows.append(self.postprocess_window)
        self.postprocess_window.show()

    def open_annot_window(self):
        self.annot_window = AntWindow()
        self.sub_windows.append(self.annot_window)
        self.annot_window.show()

    def open_output_window(self):
        self.output_window = OutputWindow()
        self.sub_windows.append(self.output_window)
        self.output_window.show()

    def open_help_window(self):
        self.help_window = HelpWindow()
        self.sub_windows.append(self.help_window)
        self.help_window.show()

    def closeEvent(self, event):
        # Close all sub-windows when the main window is closed
        for sub_window in self.sub_windows:
            sub_window.close()


if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()  # Show the main window
    sys.exit(app.exec())