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
    started = pyqtSignal() # Signal to notify initiation
    finished = pyqtSignal(str)  # Signal to notify completion

    def __init__(self):
        super().__init__()
        self.path = None
        self.mode = 'train'
        # self.args = None

    def setCommand(self, args):
        self.path = args[0]
        self.mode = args[1]

    def run(self):
        try:
            self.started.emit()
            ATL03_h5_to_csv.convert(self.path)
            split_data_bulk.split(self.path, mode=self.mode)
            self.finished.emit("Executed successfully!")
        except subprocess.CalledProcessError as e:
            self.finished.emit("Failed: " + e.stderr)

    def stop(self):
        self.terminate()
        self.finished.emit("Terminated")


class ConvertWindow(QMainWindow, Ui_ConvertWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.path = None
        self.mode = 'train'

        self.selectPathButton.clicked.connect(self.openFolderDialog)
        self.convertButton.clicked.connect(self.runCommand)

        self.command_runner = CommandRunner()
        self.command_runner.started.connect(self.showStartPopup)
        self.command_runner.finished.connect(self.showDonePopup)

    def openFolderDialog(self):
        folder_dialog = FolderDialog(self)

        directory = folder_dialog.getExistingDirectory(self, "Select Directory")
        self.folderPath.setPlainText(directory)

    def runCommand(self):
        if self.folderPath.toPlainText() != "":
            self.path = self.folderPath.toPlainText()
            if self.splitCreateBtn.isChecked():
                self.mode = 'train'
            elif self.splitDirectBtn.isChecked():
                self.mode = 'direct'
            
            # retrieve arguments                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            args = [self.path, self.mode]
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
    window = ConvertWindow()
    window.show()  # Show the main window
    sys.exit(app.exec())