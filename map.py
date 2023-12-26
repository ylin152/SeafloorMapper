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
    started = pyqtSignal()
    finished = pyqtSignal(str)  # Signal to notify completion

    def __init__(self):
        super().__init__()
        self.path = None
        self.threshold = 0.5
        self.num_votes = 10
        self.subfolders = False

    def setCommand(self, args):
        self.path = args[0]
        self.threshold = args[1]
        self.num_votes = args[2]
        self.subfolders = args[3]

    def run(self):
        try:
            self.started.emit()
            predict(self.path, gpu='0', threshold=self.threshold, num_votes=self.num_votes, subfolders=self.subfolders)
            self.finished.emit("Executed successfully!")
        except subprocess.CalledProcessError as e:
            self.finished.emit("Failed: " + e.stderr)
    
    def stop(self):
        self.terminate()
        self.finished.emit("Terminated")

class PredictWindow(QMainWindow, Ui_PredictWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.path = None
        self.threshold = 0.5
        self.num_votes = 10
        self.subfolders = False

        self.selectPathButton.clicked.connect(self.openFolderDialog)
        self.predictButton.clicked.connect(self.runCommand)

        self.command_runner = CommandRunner()
        self.command_runner.started.connect(self.showStartPopup)
        self.command_runner.finished.connect(self.showDonePopup)

    def openFolderDialog(self):
        folder_dialog = FolderDialog(self)
        directory = folder_dialog.getExistingDirectory(self, "Select Directory")
        self.folderPath.setPlainText(directory)

    def runCommand(self):
        if self.folderPath.toPlainText() != "":
            # retrieve arguments
            self.path = self.folderPath.toPlainText()
            self.threshold = self.thresSpinBox.value()
            self.num_votes = self.predsSpinBox.value()
            if self.subfoldersButton.isChecked():
                self.subfolders = True
            else:
                self.subfolders = False
            args = [self.path, self.threshold, self.num_votes, self.subfolders]
            # Pass the command to the CommandRunner thread
            self.command_runner.setCommand(args)
            # Start the command runner thread
            self.command_runner.start()

    def showStartPopup(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Starting...")
        msg_box.setText("The operation has started.")
        msg_box.exec()

    def showDonePopup(self, result):
        msg_box = QMessageBox()
        msg_box.setWindowTitle("Done")
        msg_box.setText(result)
        msg_box.exec()

    def closeEvent(self, event):
        # Terminate the CommandRunner when the main window is closed
        if self.command_runner.isRunning():
            self.command_runner.stop()
            self.command_runner.wait()  # Wait for the thread to terminate
        event.accept() # Close the window

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = PredictWindow()
    window.show()  # Show the main window
    sys.exit(app.exec())