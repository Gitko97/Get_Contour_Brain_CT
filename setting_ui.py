# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtWidgets


class Ui_Setting(QtWidgets.QDialog):
    def __init__(self, controller):
        super().__init__()
        self.controller = controller
        self.setupUi()

    def setupUi(self):
        self.setObjectName("Dialog")
        self.resize(450, 500)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(self)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")

        self.hu_boundary_value_text = QtWidgets.QPlainTextEdit(self.frame)
        self.hu_boundary_value_text.setGeometry(QtCore.QRect(190, 30, 191, 31))
        self.hu_boundary_value_text.setObjectName("hu_boundary_value_text")
        self.hu_boundary_value_label = QtWidgets.QLabel(self.frame)
        self.hu_boundary_value_label.setGeometry(QtCore.QRect(29, 35, 121, 16))
        self.hu_boundary_value_label.setObjectName("hu_boundary_value_label")

        self.normalized_min_bound_text = QtWidgets.QPlainTextEdit(self.frame)
        self.normalized_min_bound_text.setGeometry(QtCore.QRect(190, 90, 191, 31))
        self.normalized_min_bound_text.setObjectName("normalized_min_bound_text")
        self.normalized_min_bound_label = QtWidgets.QLabel(self.frame)
        self.normalized_min_bound_label.setGeometry(QtCore.QRect(9, 95, 141, 20))
        self.normalized_min_bound_label.setObjectName("normalized_min_bound_label")

        self.normalized_max_bound_label = QtWidgets.QLabel(self.frame)
        self.normalized_max_bound_label.setGeometry(QtCore.QRect(10, 150, 151, 20))
        self.normalized_max_bound_label.setObjectName("normalized_max_bound_label")
        self.normalized_max_bound_text = QtWidgets.QPlainTextEdit(self.frame)
        self.normalized_max_bound_text.setGeometry(QtCore.QRect(181, 145, 191, 31))
        self.normalized_max_bound_text.setObjectName("normalized_max_bound_text")

        self.contour_OTSU_boundary_value_label = QtWidgets.QLabel(self.frame)
        self.contour_OTSU_boundary_value_label.setGeometry(QtCore.QRect(10, 210, 161, 20))
        self.contour_OTSU_boundary_value_label.setObjectName("contour_OTSU_boundary_value_label")
        self.contour_OTSU_boundary_value_text = QtWidgets.QPlainTextEdit(self.frame)
        self.contour_OTSU_boundary_value_text.setGeometry(QtCore.QRect(181, 205, 191, 31))
        self.contour_OTSU_boundary_value_text.setObjectName("contour_OTSU_boundary_value_text")

        self.brain_init_position_label = QtWidgets.QLabel(self.frame)
        self.brain_init_position_label.setGeometry(QtCore.QRect(20, 270, 141, 16))
        self.brain_init_position_label.setObjectName("brain_init_position_label")
        self.brain_init_position_text = QtWidgets.QPlainTextEdit(self.frame)
        self.brain_init_position_text.setGeometry(QtCore.QRect(181, 265, 191, 31))
        self.brain_init_position_text.setObjectName("brain_init_position_text")

        self.saveButton = QtWidgets.QPushButton(self.frame)
        self.saveButton.setGeometry(QtCore.QRect(40, 390, 113, 32))
        self.saveButton.setObjectName("saveButton")
        self.saveButton.clicked.connect(self.controller.value_save)


        self.previewButton = QtWidgets.QPushButton(self.frame)
        self.previewButton.setGeometry(QtCore.QRect(220, 390, 113, 32))
        self.previewButton.setObjectName("previewButton")
        self.previewButton.clicked.connect(self.controller.preview_setting)

        self.resetValueButton = QtWidgets.QPushButton(self.frame)
        self.resetValueButton.setGeometry(QtCore.QRect(130, 310, 113, 32))
        self.resetValueButton.setObjectName("resetValueButton")
        self.resetValueButton.clicked.connect(self.controller.reset_setting_value)

        self.horizontalLayout.addWidget(self.frame)
        self.setLayout(self.horizontalLayout)
        self.retranslateUi(self)
        QtCore.QMetaObject.connectSlotsByName(self)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.hu_boundary_value_label.setText(_translate("Dialog", "hu_boundary_value"))
        self.normalized_min_bound_label.setText(_translate("Dialog", "normalized_min_bound"))
        self.normalized_max_bound_label.setText(_translate("Dialog", "normalized_max_bound"))
        self.contour_OTSU_boundary_value_label.setText(_translate("Dialog", "contour_OTSU_boundary_value"))
        self.brain_init_position_label.setText(_translate("Dialog", "brain_init_position"))
        self.saveButton.setText(_translate("Dialog", "저장"))
        self.previewButton.setText(_translate("Dialog", "미리보기"))
        self.resetValueButton.setText(_translate("Dialog", "초기화"))