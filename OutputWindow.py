# Form implementation generated from reading ui file 'OutputWindow.ui'
#
# Created by: PyQt6 UI code generator 6.5.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_OutputWindow(object):
    def setupUi(self, OutputWindow):
        OutputWindow.setObjectName("OutputWindow")
        OutputWindow.resize(479, 446)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Preferred, QtWidgets.QSizePolicy.Policy.Minimum)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(OutputWindow.sizePolicy().hasHeightForWidth())
        OutputWindow.setSizePolicy(sizePolicy)
        self.centralwidget = QtWidgets.QWidget(parent=OutputWindow)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayout_2 = QtWidgets.QVBoxLayout(self.centralwidget)
        self.verticalLayout_2.setObjectName("verticalLayout_2")
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.inputPathBtn = QtWidgets.QPushButton(parent=self.centralwidget)
        self.inputPathBtn.setObjectName("inputPathBtn")
        self.gridLayout.addWidget(self.inputPathBtn, 0, 0, 1, 1)
        self.inputPath = QtWidgets.QPlainTextEdit(parent=self.centralwidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Policy.Expanding, QtWidgets.QSizePolicy.Policy.Expanding)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.inputPath.sizePolicy().hasHeightForWidth())
        self.inputPath.setSizePolicy(sizePolicy)
        self.inputPath.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.WidgetWidth)
        self.inputPath.setReadOnly(True)
        self.inputPath.setObjectName("inputPath")
        self.gridLayout.addWidget(self.inputPath, 0, 1, 1, 1)
        self.verticalLayout_2.addLayout(self.gridLayout)
        self.groupBox = QtWidgets.QGroupBox(parent=self.centralwidget)
        self.groupBox.setObjectName("groupBox")
        self.gridLayout_3 = QtWidgets.QGridLayout(self.groupBox)
        self.gridLayout_3.setObjectName("gridLayout_3")
        self.pointsGroup = QtWidgets.QGroupBox(parent=self.groupBox)
        self.pointsGroup.setObjectName("pointsGroup")
        self.verticalLayout_3 = QtWidgets.QVBoxLayout(self.pointsGroup)
        self.verticalLayout_3.setObjectName("verticalLayout_3")
        self.allRadioBtn = QtWidgets.QRadioButton(parent=self.pointsGroup)
        self.allRadioBtn.setChecked(True)
        self.allRadioBtn.setObjectName("allRadioBtn")
        self.verticalLayout_3.addWidget(self.allRadioBtn)
        self.sfRadioBtn = QtWidgets.QRadioButton(parent=self.pointsGroup)
        self.sfRadioBtn.setObjectName("sfRadioBtn")
        self.verticalLayout_3.addWidget(self.sfRadioBtn)
        self.gridLayout_3.addWidget(self.pointsGroup, 0, 1, 1, 2)
        self.rcorrBox = QtWidgets.QCheckBox(parent=self.groupBox)
        self.rcorrBox.setObjectName("rcorrBox")
        self.gridLayout_3.addWidget(self.rcorrBox, 0, 0, 1, 1)
        self.gridLayout_2 = QtWidgets.QGridLayout()
        self.gridLayout_2.setObjectName("gridLayout_2")
        self.outputPathButton = QtWidgets.QPushButton(parent=self.groupBox)
        self.outputPathButton.setObjectName("outputPathButton")
        self.gridLayout_2.addWidget(self.outputPathButton, 0, 0, 1, 1)
        self.outputPath = QtWidgets.QPlainTextEdit(parent=self.groupBox)
        self.outputPath.setLineWrapMode(QtWidgets.QPlainTextEdit.LineWrapMode.NoWrap)
        self.outputPath.setReadOnly(False)
        self.outputPath.setObjectName("outputPath")
        self.gridLayout_2.addWidget(self.outputPath, 0, 1, 1, 1)
        self.gridLayout_3.addLayout(self.gridLayout_2, 2, 0, 1, 3)
        self.trackGroup = QtWidgets.QGroupBox(parent=self.groupBox)
        self.trackGroup.setObjectName("trackGroup")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.trackGroup)
        self.verticalLayout.setObjectName("verticalLayout")
        self.sepTrackRadioBtn = QtWidgets.QRadioButton(parent=self.trackGroup)
        self.sepTrackRadioBtn.setChecked(True)
        self.sepTrackRadioBtn.setObjectName("sepTrackRadioBtn")
        self.verticalLayout.addWidget(self.sepTrackRadioBtn)
        self.oneTrackRadioBtn = QtWidgets.QRadioButton(parent=self.trackGroup)
        self.oneTrackRadioBtn.setObjectName("oneTrackRadioBtn")
        self.verticalLayout.addWidget(self.oneTrackRadioBtn)
        self.gridLayout_3.addWidget(self.trackGroup, 1, 0, 1, 2)
        self.formatGroup = QtWidgets.QGroupBox(parent=self.groupBox)
        self.formatGroup.setObjectName("formatGroup")
        self.verticalLayout_4 = QtWidgets.QVBoxLayout(self.formatGroup)
        self.verticalLayout_4.setObjectName("verticalLayout_4")
        self.csvRadioBtn = QtWidgets.QRadioButton(parent=self.formatGroup)
        self.csvRadioBtn.setChecked(True)
        self.csvRadioBtn.setObjectName("csvRadioBtn")
        self.verticalLayout_4.addWidget(self.csvRadioBtn)
        self.txtRadioBtn = QtWidgets.QRadioButton(parent=self.formatGroup)
        self.txtRadioBtn.setObjectName("txtRadioBtn")
        self.verticalLayout_4.addWidget(self.txtRadioBtn)
        self.gridLayout_3.addWidget(self.formatGroup, 1, 2, 1, 1)
        self.verticalLayout_2.addWidget(self.groupBox)
        self.outputBtn = QtWidgets.QPushButton(parent=self.centralwidget)
        self.outputBtn.setObjectName("outputBtn")
        self.verticalLayout_2.addWidget(self.outputBtn)
        OutputWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtWidgets.QMenuBar(parent=OutputWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 479, 21))
        self.menubar.setObjectName("menubar")
        OutputWindow.setMenuBar(self.menubar)
        self.statusbar = QtWidgets.QStatusBar(parent=OutputWindow)
        self.statusbar.setObjectName("statusbar")
        OutputWindow.setStatusBar(self.statusbar)

        self.retranslateUi(OutputWindow)
        QtCore.QMetaObject.connectSlotsByName(OutputWindow)

    def retranslateUi(self, OutputWindow):
        _translate = QtCore.QCoreApplication.translate
        OutputWindow.setWindowTitle(_translate("OutputWindow", "Output Data"))
        self.inputPathBtn.setText(_translate("OutputWindow", "Select input path"))
        self.groupBox.setTitle(_translate("OutputWindow", "Setting"))
        self.pointsGroup.setTitle(_translate("OutputWindow", "Category"))
        self.allRadioBtn.setText(_translate("OutputWindow", "All points"))
        self.sfRadioBtn.setText(_translate("OutputWindow", "Seafloor points only"))
        self.rcorrBox.setText(_translate("OutputWindow", "Refraction Correction"))
        self.outputPathButton.setText(_translate("OutputWindow", "Set output path (optional)"))
        self.trackGroup.setTitle(_translate("OutputWindow", "Track"))
        self.sepTrackRadioBtn.setText(_translate("OutputWindow", "Separate tracks"))
        self.oneTrackRadioBtn.setText(_translate("OutputWindow", "Combine tracks to one"))
        self.formatGroup.setTitle(_translate("OutputWindow", "Format"))
        self.csvRadioBtn.setText(_translate("OutputWindow", "csv"))
        self.txtRadioBtn.setText(_translate("OutputWindow", "txt"))
        self.outputBtn.setText(_translate("OutputWindow", "Output"))
