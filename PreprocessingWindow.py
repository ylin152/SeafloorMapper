# Form implementation generated from reading ui file 'PreprocessingWindow.ui'
#
# Created by: PyQt6 UI code generator 6.5.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_PreprocessingWindow(object):
    def setupUi(self, PreprocessingWindow):
        PreprocessingWindow.setObjectName("PreprocessingWindow")
        PreprocessingWindow.resize(461, 155)
        self.centralwidget = QtWidgets.QWidget(parent=PreprocessingWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout.setObjectName("verticalLayout")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setSpacing(12)
        self.gridLayout.setObjectName("gridLayout")
        self.folderPath = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        self.folderPath.setReadOnly(True)
        self.folderPath.setObjectName("folderPath")
        self.gridLayout.addWidget(self.folderPath, 0, 1, 1, 1)
        self.selectPathButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.selectPathButton.setObjectName("selectPathButton")
        self.gridLayout.addWidget(self.selectPathButton, 0, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.preprocessButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.preprocessButton.setObjectName("preprocessButton")
        self.verticalLayout.addWidget(self.preprocessButton)
        PreprocessingWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=PreprocessingWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 461, 21))
        self.menubar.setObjectName("menubar")
        PreprocessingWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=PreprocessingWindow)
        self.statusbar.setObjectName("statusbar")
        PreprocessingWindow.setStatusBar(self.statusbar)

        self.retranslateUi(PreprocessingWindow)
        QtCore.QMetaObject.connectSlotsByName(PreprocessingWindow)

    def retranslateUi(self, PreprocessingWindow):
        _translate = QtCore.QCoreApplication.translate
        PreprocessingWindow.setWindowTitle(_translate("PreprocessingWindow", "Data Preprocessing"))
        self.selectPathButton.setText(_translate("PreprocessingWindow", "Select path"))
        self.preprocessButton.setText(_translate("PreprocessingWindow", "Preprocess"))
