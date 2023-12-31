# Form implementation generated from reading ui file 'PredictWindow.ui'
#
# Created by: PyQt6 UI code generator 6.5.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_PredictWindow(object):
    def setupUi(self, PredictWindow):
        PredictWindow.setObjectName("PredictWindow")
        PredictWindow.resize(461, 258)
        PredictWindow.setMaximumSize(QtCore.QSize(16777215, 339))
        self.centralwidget = QtWidgets.QWidget(parent=PredictWindow)
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
        self.thresLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.thresLabel.setObjectName("thresLabel")
        self.gridLayout.addWidget(self.thresLabel, 1, 0, 1, 1)
        self.numOfPredsLabel = QtWidgets.QLabel(parent=self.centralwidget)
        self.numOfPredsLabel.setObjectName("numOfPredsLabel")
        self.gridLayout.addWidget(self.numOfPredsLabel, 2, 0, 1, 1)
        self.thresSpinBox = QtWidgets.QDoubleSpinBox(parent=self.centralwidget)
        self.thresSpinBox.setDecimals(1)
        self.thresSpinBox.setMinimum(0.1)
        self.thresSpinBox.setMaximum(0.9)
        self.thresSpinBox.setSingleStep(0.1)
        self.thresSpinBox.setProperty("value", 0.5)
        self.thresSpinBox.setObjectName("thresSpinBox")
        self.gridLayout.addWidget(self.thresSpinBox, 1, 1, 1, 1)
        self.predsSpinBox = QtWidgets.QSpinBox(parent=self.centralwidget)
        self.predsSpinBox.setMinimum(1)
        self.predsSpinBox.setMaximum(50)
        self.predsSpinBox.setProperty("value", 10)
        self.predsSpinBox.setObjectName("predsSpinBox")
        self.gridLayout.addWidget(self.predsSpinBox, 2, 1, 1, 1)
        self.subfoldersButton = QtWidgets.QRadioButton(parent=self.centralwidget)
        self.subfoldersButton.setObjectName("subfoldersButton")
        self.gridLayout.addWidget(self.subfoldersButton, 3, 0, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)
        self.predictButton = QtWidgets.QPushButton(parent=self.centralwidget)
        self.predictButton.setObjectName("predictButton")
        self.verticalLayout.addWidget(self.predictButton)
        PredictWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=PredictWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 461, 21))
        self.menubar.setObjectName("menubar")
        PredictWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=PredictWindow)
        self.statusbar.setObjectName("statusbar")
        PredictWindow.setStatusBar(self.statusbar)

        self.retranslateUi(PredictWindow)
        QtCore.QMetaObject.connectSlotsByName(PredictWindow)

    def retranslateUi(self, PredictWindow):
        _translate = QtCore.QCoreApplication.translate
        PredictWindow.setWindowTitle(_translate("PredictWindow", "Model Prediction"))
        self.selectPathButton.setText(_translate("PredictWindow", "Select path"))
        self.thresLabel.setText(_translate("PredictWindow", "Threshold"))
        self.numOfPredsLabel.setText(_translate("PredictWindow", "Number of Predictions"))
        self.subfoldersButton.setText(_translate("PredictWindow", "Subfolders"))
        self.predictButton.setText(_translate("PredictWindow", "Predict"))
