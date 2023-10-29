import os
import sys
import shutil
import subprocess
from PyQt6.QtWidgets import QApplication, QMainWindow, QFileDialog, QMessageBox
from CreateWindow import Ui_CreateWindow
from PyQt6.QtCore import QThread, pyqtSignal

base_dir = os.path.dirname(os.path.abspath(__file__))


class FolderDialog(QFileDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Folder")
        self.setFileMode(QFileDialog.FileMode.Directory)


class CreateWindow(QMainWindow, Ui_CreateWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.path = None

        self.selectPathButton.clicked.connect(self.openFolderDialog)
        self.generateButton.clicked.connect(self.runCommand)

        # self.command_runner = CommandRunner()
        # self.command_runner.finished.connect(self.showDonePopup)

    def openFolderDialog(self):
        folder_dialog = FolderDialog(self)

        directory = folder_dialog.getExistingDirectory(self, "Select Directory")
        self.folderPath.setPlainText(directory)

    def runCommand(self):
        self.showStartPopup()

        try:
            self.path = self.folderPath.toPlainText()
            dirname = os.path.dirname(self.path)
            dst_dirname = os.path.join(dirname, 'input_data')
            if not os.path.exists(dst_dirname):
                os.mkdir(dst_dirname)
            for file in os.listdir(self.path):
                src_file = os.path.join(self.path, file)
                dst_file = os.path.join(dst_dirname, file) 
                shutil.copy(src_file, dst_file)
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
    window = CreateWindow()
    window.show()  # Show the main window
    sys.exit(app.exec())