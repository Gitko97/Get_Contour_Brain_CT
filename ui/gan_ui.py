# -*- coding: utf-8 -*-

from PyQt5 import QtCore, QtWidgets
from PyQt5.QtWidgets import QMessageBox


class GanUI(QtWidgets.QDialog):
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
        self.resize(800, 800)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.frame = QtWidgets.QFrame(self)
        self.frame.setFrameShape(QtWidgets.QFrame.StyledPanel)
        self.frame.setFrameShadow(QtWidgets.QFrame.Raised)
        self.frame.setObjectName("frame")

        self.generator_layers_view = QtWidgets.QListWidget(self.frame)
        self.generator_layers_view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.generator_layers_view.setGeometry(QtCore.QRect(0, 0, 400, 500))
        self.generator_layers_view.setObjectName("generator_layers")

        self.discriminator_layers_view = QtWidgets.QListWidget(self.frame)
        self.discriminator_layers_view.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.discriminator_layers_view.setGeometry(QtCore.QRect(400, 0, 400, 500))
        self.discriminator_layers_view.setObjectName("discriminator_layers")

        self.listView = QtWidgets.QListWidget(self.frame)
        self.listView.setSelectionMode(QtWidgets.QAbstractItemView.ExtendedSelection)
        self.listView.setGeometry(QtCore.QRect(0, 500, 800, 100))
        self.listView.setObjectName("generator_layers")
        # self.listView.itemClicked.connect(self.controller.itemClicked)

        self.gan_button = QtWidgets.QPushButton(self.frame)
        self.gan_button.setGeometry(QtCore.QRect(30, 605, 201, 32))
        self.gan_button.setObjectName("gan_button")
        self.gan_button.clicked.connect(self.controller.press_gan_button)
        #
        self.generator_button = QtWidgets.QPushButton(self.frame)
        self.generator_button.setGeometry(QtCore.QRect(260, 605, 201, 32))
        self.generator_button.setObjectName("generator_button")
        self.generator_button.clicked.connect(self.controller.press_generator_button)
        #
        self.discriminator_button = QtWidgets.QPushButton(self.frame)
        self.discriminator_button.setGeometry(QtCore.QRect(490, 605, 201, 32))
        self.discriminator_button.setObjectName("discriminator_button")
        self.discriminator_button.clicked.connect(self.controller.press_discriminaotr_button)

        self.horizontalLayout.addWidget(self.frame)
        self.setLayout(self.horizontalLayout)
        self.retranslateUi(self)



    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.gan_button.setText(_translate("Dialog", "GAN"))
        self.generator_button.setText(_translate("Dialog", "Generator"))
        self.discriminator_button.setText(_translate("Dialog", "Discriminator"))