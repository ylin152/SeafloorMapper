# Form implementation generated from reading ui file 'PreviewDialog.ui'
#
# Created by: PyQt6 UI code generator 6.5.2
#
# WARNING: Any manual changes made to this file will be lost when pyuic6 is
# run again.  Do not edit this file unless you know what you are doing.


from PyQt6 import QtCore, QtGui, QtWidgets


class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(534, 406)
        self.verticalLayout = QtWidgets.QVBoxLayout(Dialog)
        self.verticalLayout.setObjectName("verticalLayout")
        self.tableWidget = QtWidgets.QTableWidget(parent=Dialog)
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)
        self.verticalLayout.addWidget(self.tableWidget)
        self.gridLayout = QtWidgets.QGridLayout()
        self.gridLayout.setObjectName("gridLayout")
        self.label = QtWidgets.QLabel(parent=Dialog)
        self.label.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label.setObjectName("label")
        self.gridLayout.addWidget(self.label, 0, 0, 1, 1)
        self.comboBox_lat = QtWidgets.QComboBox(parent=Dialog)
        self.comboBox_lat.setObjectName("comboBox_lat")
        self.gridLayout.addWidget(self.comboBox_lat, 0, 3, 1, 1)
        self.comboBox_elev = QtWidgets.QComboBox(parent=Dialog)
        self.comboBox_elev.setObjectName("comboBox_elev")
        self.gridLayout.addWidget(self.comboBox_elev, 1, 1, 1, 1)
        self.label_2 = QtWidgets.QLabel(parent=Dialog)
        self.label_2.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_2.setObjectName("label_2")
        self.gridLayout.addWidget(self.label_2, 0, 2, 1, 1)
        self.label_5 = QtWidgets.QLabel(parent=Dialog)
        self.label_5.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_5.setObjectName("label_5")
        self.gridLayout.addWidget(self.label_5, 0, 4, 1, 1)
        self.label_4 = QtWidgets.QLabel(parent=Dialog)
        self.label_4.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_4.setObjectName("label_4")
        self.gridLayout.addWidget(self.label_4, 1, 2, 1, 1)
        self.label_3 = QtWidgets.QLabel(parent=Dialog)
        self.label_3.setAlignment(QtCore.Qt.AlignmentFlag.AlignCenter)
        self.label_3.setObjectName("label_3")
        self.gridLayout.addWidget(self.label_3, 1, 0, 1, 1)
        self.comboBox_lon = QtWidgets.QComboBox(parent=Dialog)
        self.comboBox_lon.setObjectName("comboBox_lon")
        self.gridLayout.addWidget(self.comboBox_lon, 0, 1, 1, 1)
        self.comboBox_label = QtWidgets.QComboBox(parent=Dialog)
        self.comboBox_label.setObjectName("comboBox_label")
        self.gridLayout.addWidget(self.comboBox_label, 1, 3, 1, 1)
        self.comboBox_y = QtWidgets.QComboBox(parent=Dialog)
        self.comboBox_y.setObjectName("comboBox_y")
        self.gridLayout.addWidget(self.comboBox_y, 0, 5, 1, 1)
        self.pushButton = QtWidgets.QPushButton(parent=Dialog)
        self.pushButton.setObjectName("pushButton")
        self.gridLayout.addWidget(self.pushButton, 2, 5, 1, 1)
        self.verticalLayout.addLayout(self.gridLayout)

        self.retranslateUi(Dialog)
        QtCore.QMetaObject.connectSlotsByName(Dialog)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Preview Data"))
        self.label.setText(_translate("Dialog", "Longitude"))
        self.label_2.setText(_translate("Dialog", "Latitude"))
        self.label_5.setText(_translate("Dialog", "Y Distance"))
        self.label_4.setText(_translate("Dialog", "Label"))
        self.label_3.setText(_translate("Dialog", "Elevation"))
        self.pushButton.setText(_translate("Dialog", "OK"))