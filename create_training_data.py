import os
import sys
import shutil
import pandas as pd
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

class CommandRunner(QThread):
    started = pyqtSignal()
    finished = pyqtSignal(str)  # Signal to notify completion

    def __init__(self):
        super().__init__()
        self.path = None

    def setCommand(self, args):
        self.path = args[0]
        self.from_pred = args[1]

    def run(self):
        try:
            self.started.emit()
            dirname = os.path.dirname(self.path)
            dst_dirname = os.path.join(dirname, 'input_data')
            if not os.path.exists(dst_dirname):
                os.mkdir(dst_dirname)
            for file in os.listdir(self.path):
                if not self.from_pred:
                    dst_file = os.path.join(dst_dirname, file) 
                    src_file = os.path.join(self.path, file)
                    shutil.copy(src_file, dst_file)
                else:
                    file_new = os.path.splitext(file)[0] + '.txt'
                    dst_file = os.path.join(dst_dirname, file_new) 
                    df = pd.read_csv(os.path.join(self.path, file))
                    df = df.drop(columns=['prob'])
                    # df.rename(columns={'pred':'annot'})
                    df.to_csv(dst_file, header=False, index=False, sep=' ')
            
            self.finished.emit("Executed successfully!")
        except subprocess.CalledProcessError as e:
            self.finished.emit("Failed: " + e.stderr)

    def stop(self):
        self.terminate()
        self.finished.emit("Terminated")


class CreateWindow(QMainWindow, Ui_CreateWindow):
    def __init__(self):
        super().__init__()
        self.setupUi(self)

        self.path = None
        self.from_pred = False

        self.selectPathButton.clicked.connect(self.openFolderDialog)
        self.generateButton.clicked.connect(self.runCommand)

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
            if self.fromPredButton.isChecked():
                self.from_pred = True
            else:
                self.from_pred = False

            # retrieve arguments                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                
            args = [self.path, self.from_pred]
            # Pass the command to the CommandRunner thread
            self.command_runner.setCommand(args)
            # Start the command runner thread
            self.command_runner.start()

        # try:
        #     self.path = self.folderPath.toPlainText()
        #     dirname = os.path.dirname(self.path)
        #     dst_dirname = os.path.join(dirname, 'input_data')
        #     if not os.path.exists(dst_dirname):
        #         os.mkdir(dst_dirname)
        #     for file in os.listdir(self.path):
        #         dst_file = os.path.join(dst_dirname, file) 
        #         if not self.from_pred:
        #             src_file = os.path.join(self.path, file)
        #             shutil.copy(src_file, dst_file)
        #         else:
        #             df = pd.read_csv(file)
        #             df.drop(columns=['prob'])
        #             df.rename(columns={'pred':'annot'})
        #             df.to_csv(dst_file, header=False, index=False)

        #     self.showDonePopup('Done')
        # except:
        #     self.showDonePopup('Failed')

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

    def closeEvent(self, event):
        # Terminate the CommandRunner when the main window is closed
        if self.command_runner.isRunning():
            self.command_runner.stop()
            self.command_runner.wait()  # Wait for the thread to terminate
        event.accept() # Close the window

if __name__ == "__main__":
    app = QApplication(sys.argv)
    window = CreateWindow()
    window.show()  # Show the main window
    sys.exit(app.exec())