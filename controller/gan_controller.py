import cv2
from matplotlib import pyplot as plt
from Dicom.Get_Contour_Brain_CT.image_utils.preprocessor import Image_PreProcessor
from Dicom.Get_Contour_Brain_CT.ui.gan_ui import GanUI
from Dicom.Get_Contour_Brain_CT.model.gan import GAN
from Dicom.Get_Contour_Brain_CT.model.discriminator import Discriminator
from Dicom.Get_Contour_Brain_CT.model.generator import Generator
import numpy as np

class GanController(object):
    def __init__(self, input_shape=(512, 512, 1)):
        self.ui = GanUI(self)
        self.input_shape = input_shape
        self.discriminator = Discriminator(input_shape=input_shape, name="Discriminator")
        self.generator = Generator(input_shape=(512, 512, 1), name="Generator")
        self.gan = GAN(discriminator=self.discriminator.model, generator=self.generator.model)
        self.set_layer_summary_string()

    def start(self):
        self.ui.show()

    def set_layer_summary_string(self):
        generator_summary = []
        discriminator_summary = []

        self.generator.model.summary(print_fn=lambda x: generator_summary.append(x))
        self.discriminator.model.summary(print_fn=lambda x: discriminator_summary.append(x))
        for text in generator_summary:
            self.ui.generator_layers_view.addItem(text)

        for text in discriminator_summary:
            self.ui.discriminator_layers_view.addItem(text)

    def press_generator_button(self):
        try:
            self.generator.model.predict(np.zeros((1, 512, 512, 1)))
            self.ui.listView.addItem("Generator 동작 성공")
        except Exception as e:
            self.ui.listView.addItem('Generator 예외가 발생했습니다.\n' + e.__str__())

    def press_discriminaotr_button(self):
        try:
            self.discriminator.model.predict(np.zeros((1, 512, 512, 1)), np.zeros(1))
            self.ui.listView.addItem("Discriminator 동작 성공")
        except Exception as e:
            self.ui.listView.addItem('Discriminator 예외가 발생했습니다.\n' + e.__str__())

    def press_gan_button(self):
        try:
            self.gan.gan_model.predict(np.zeros((1, 512, 512, 1)), np.zeros(1))
            self.ui.listView.addItem("GAN 동작 성공")
        except Exception as e:
            self.ui.listView.addItem('GAN 예외가 발생했습니다.\n' + e.__str__())
