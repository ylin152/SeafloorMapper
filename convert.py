import os
import sys
import subprocess
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from ConvertWindow import Ui_ConvertWindow
from PyQt6.QtCore import QThread, pyqtSignal

from preprocessing import ATL03_h5_to_csv
from preprocessing import split_data_bulk

base_dir = os.path.dirname(os.path.abspath(__file__))


class FolderDialog(QFileDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Folder")
        self.setFileMode(QFileDialog.FileMode.Directory)


class CommandRunner(QThread):
    finished = pyqtSignal(str)  # Signal to notify completion

    def __init__(self):
        super().__init__()
        self.path = None
        self.threshold = 0.5
        self.num_votes = 10
        # self.args = None

    def setCommand(self, args):
        self.path = args[0]

    def run(self):
        try:
            # Get the path to the Python interpreter (sys.executable)
            python_interpreter = sys.executable

            convert_script = os.path.join(base_dir, 'preprocessing', 'ATL03_h5_to_csv.py')
            split_script = os.path.join(base_dir, 'preprocessing', 'split_data_bulk.py')

            commands = [
                f'{python_interpreter} {convert_script} --data_dir {self.path} --removeLand --removeIrrelevant --utm',
                f'{python_interpreter} {split_script} --input_dir {self.path}'
            ]

            output = 'Start pre-processing ' + self.path + '\n'
            self.finished.emit(output)
            output = ''
            # Iterate through the list and run each script with its parameters
            for command in commands:
                result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
                output += result.stdout + '\n'

            self.finished.emit(output + 'Executed successfully!')
            # self.finished.emit("Executed successfully!")
        except subprocess.CalledProcessError as e:
            self.finished.emit("Failed: " + e.stderr)


class ConvertWindow(QMainWindow, Ui_ConvertWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.path = None
        self.threshold = 0.5
        self.num_votes = 10

        self.selectPathButton.clicked.connect(self.openFolderDialog)
        self.convertButton.clicked.connect(self.runCommand)

        # self.command_runner = CommandRunner()
        # self.command_runner.finished.connect(self.showDonePopup)

    def openFolderDialog(self):
        folder_dialog = FolderDialog(self)

        directory = folder_dialog.getExistingDirectory(self, "Select Directory")
        self.folderPath.setPlainText(directory)

    def runCommand(self):
        self.showStartPopup()

        self.path = self.folderPath.toPlainText()

        try:
            ATL03_h5_to_csv.convert(self.path)

            if self.splitCreateBtn.isChecked():
                split_data_bulk.split(self.path, mode='train')

            if self.splitDirectBtn.isChecked():
                split_data_bulk.split(self.path, mode='direct')

            self.showDonePopup('Done')
        except:
            self.showDonePopup('Failed')

    def showStartPopup(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Starting...")
        msg_box.setText("The operation has started.")
        msg_box.exec()

    def showDonePopup(self, result):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(result)
        msg_box.setText(result+'!')
        msg_box.exec()

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = ConvertWindow()
    window.show()  # Show the main window
    sys.exit(app.exec())