# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMessageBox


class Ui_Dialog(QtWidgets.QDialog):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setupUi()

    def show_popup_ok(self, title: str, content: str):
        msg = QMessageBox()
        msg.setWindowTitle(title)
        msg.setText(content)
        msg.setStandardButtons(QMessageBox.Ok)
        msg.exec_()

    def setupUi(self):
        self.setObjectName("Dialog")
        self.resize(800, 600)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(self)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")
        self.listView = QtWidgets.QListWidget(self.frame)
        self.listView.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listView.setGeometry(QtCore.QRect(0, 0, 781, 431))
        self.listView.setObjectName("listView")

        self.choose_directory_button = QtWidgets.QPushButton(self.frame)
        self.choose_directory_button.setGeometry(QtCore.QRect(30, 460, 201, 32))
        self.choose_directory_button.setObjectName("choose_directory_button")

        self.selected_delete_button = QtWidgets.QPushButton(self.frame)
        self.selected_delete_button.setGeometry(QtCore.QRect(130, 510, 201, 32))
        self.selected_delete_button.setObjectName("selected_delete_button")

        self.add_file_button = QtWidgets.QPushButton(self.frame)
        self.add_file_button.setGeometry(QtCore.QRect(260, 460, 201, 32))
        self.add_file_button.setObjectName("add_file_button")

        self.dicom_save_button = QtWidgets.QPushButton(self.frame)
        self.dicom_save_button.setGeometry(QtCore.QRect(490, 460, 201, 32))
        self.dicom_save_button.setObjectName("dicom_save_button")
        self.dicom_save_button.clicked.connect(self.controller.file_save)

        self.setting_value_button = QtWidgets.QPushButton(self.frame)
        self.setting_value_button.setGeometry(QtCore.QRect(410, 510, 201, 32))
        self.setting_value_button.setObjectName("setting_value")

        self.change_TFRecord_button = QtWidgets.QPushButton(self.frame)
        self.change_TFRecord_button.setGeometry(QtCore.QRect(300, 530, 201, 32))
        self.change_TFRecord_button.setObjectName("tfRecord")
        self.change_TFRecord_button.clicked.connect(self.controller.change_dicom_to_tfRecord)

        self.listView.itemClicked.connect(self.controller.itemClicked)
        # self.listView.itemDoubleClicked.connect(self.controller.itemDoubleClicked)
        # self.listView.currentItemChanged.connect(self.chkCurrentItemChanged)
        self.choose_directory_button.clicked.connect(self.controller.load_files)
        self.add_file_button.clicked.connect(self.controller.add_files)
        self.selected_delete_button.clicked.connect(self.controller.delete_files)
        self.setting_value_button.clicked.connect(self.controller.openSetting)

        self.horizontalLayout.addWidget(self.frame)
        self.setLayout(self.horizontalLayout)
        self.retranslateUi(self)



    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.choose_directory_button.setText(_translate("Dialog", "폴더 선택"))
        self.selected_delete_button.setText(_translate("Dialog", "선택 삭제"))
        self.add_file_button.setText(_translate("Dialog", "파일 추가"))
        self.setting_value_button.setText(_translate("Dialog", "기본 값 설정"))
        self.dicom_save_button.setText(_translate("Dialog", "자른 파일 저장"))
        self.change_TFRecord_button.setText(_translate("Dialog", "tfRecord저장"))