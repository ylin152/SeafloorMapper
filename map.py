import os
import sys
import subprocess
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from PredictWindow import Ui_PredictWindow
from PyQt6.QtCore import QThread, pyqtSignal
from predict import predict

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
        self.threshold = args[1]
        self.num_votes = args[2]

    def run(self):
        try:
            # result = subprocess.run(self.command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            # output = result.stdout
            #
            # # Notify that the operation is complete
            # self.finished.emit(output)

            python_interpreter = sys.executable

            predict_script = os.path.join(base_dir, 'predict.py')

            print(os.path.exists(predict_script))

            command = f'{python_interpreter} {predict_script} --data_root {self.path} --conf --threshold {self.threshold} --num_votes {self.num_votes}'

            output = 'Start predict ' + self.path + '\n'
            self.finished.emit(output)
            output = ''
            # Iterate through the list and run each script with its parameters
            result = subprocess.run(command, shell=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
            output += result.stdout + '\n'

            self.finished.emit(output + 'Executed successfully!')
            # self.finished.emit("Executed successfully!")
        except subprocess.CalledProcessError as e:
            self.finished.emit("Failed: " + e.stderr)


class PredictWindow(QMainWindow, Ui_PredictWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.path = None
        self.threshold = 0.5
        self.num_votes = 10

        self.selectPathButton.clicked.connect(self.openFolderDialog)
        self.predictButton.clicked.connect(self.runCommand)

        # self.command_runner = CommandRunner()
        # self.command_runner.finished.connect(self.showDonePopup)

    def openFolderDialog(self):
        folder_dialog = FolderDialog(self)
        # options = QFileDialog.Options()
        # options |= QFileDialog.Option.ShowDirsOnly

        directory = folder_dialog.getExistingDirectory(self, "Select Directory")
        self.folderPath.setPlainText(directory)

        # if folder_dialog.exec() == QFileDialog.Accepted:
        #     selected_folder = folder_dialog.selectedFiles()[0]
        #     self.folderPath.setPlainText(selected_folder)

    # def run_bash(self):
    #     self.path = self.folderPath.toPlainText()
    #     self.threshold = self.thresSpinBox.value()
    #     self.num_votes = self.predsSpinBox.value()
    #     res = subprocess.run(['bash'], self.script, self.path, self.threshold, self.num_votes)
    #     if res.returncode != 0:
    #         print('Failed')
    #     else:
    #         print('Successful')

    def runCommand(self):
        self.showStartPopup()  # Show "start..." popup

        # retrieve arguments
        # script = os.path.join(base_dir, 'predict_script.sh')
        self.path = self.folderPath.toPlainText()
        self.threshold = self.thresSpinBox.value()
        self.num_votes = self.predsSpinBox.value()
        # args = [self.path, str(self.threshold), str(self.num_votes)]
        # # Pass the command to the CommandRunner thread
        # self.command_runner.setCommand(args)
        # # Start the command runner thread
        # self.command_runner.start()

        try:
            predict(data_root=self.path, threshold=self.threshold, num_votes=self.num_votes)
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
    window = PredictWindow()
    window.show()  # Show the main window
    sys.exit(app.exec())