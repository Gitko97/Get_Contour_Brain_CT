import cv2
import numpy as np
from matplotlib import pyplot as plt
from Dicom.Get_Contour_Brain_CT.preprocessor import Image_PreProcessor
from Dicom.Get_Contour_Brain_CT.setting_ui import Ui_Setting
from pydicom.pixel_data_handlers import util

class SettingController(object):
    def __init__(self, main_controller, dicom_path):
        self.image_preprocessor = Image_PreProcessor()
        self.main_controller = main_controller
        self.ui = Ui_Setting(self)
        self.dicom_file = self.image_preprocessor.load_scan([dicom_path])

    def get_text_field_value(self):
        brain_init_position = self.ui.brain_init_position_text.toPlainText()
        brain_init_position = brain_init_position.split(',')
        brain_init_position = (int(brain_init_position[0][1:]), int(brain_init_position[1][:-1]))
        setting_value = {
            "hu_boundary_value": int(self.ui.hu_boundary_value_text.toPlainText()),
            "brain_init_position": brain_init_position
        }
        return setting_value

    def reset_setting_value(self):
        self.set_text_field(self.main_controller.default_setting_value)

    def value_save(self):
        setting_value = self.get_text_field_value()
        self.main_controller.set_setting_value(setting_value)
        self.ui.destroy()

    def preview_setting(self):
        setting = self.get_text_field_value()
        images_hu_pixels = self.image_preprocessor.get_pixels_hu(self.dicom_file)
        binary_image = self.image_preprocessor.get_binary_image_with_hu_value(images_hu_pixels[0],
                                                                              hu_boundary_value=setting.get(
                                                                                  "hu_boundary_value"))
        _, contours, hierarchy = self.image_preprocessor.find_dicom_Countour(binary_image)
        brain_contour = self.image_preprocessor.find_brain_contour(contours=contours, hierarchy=hierarchy,
                                                                   init_position=setting.get(
                                                                       "brain_init_position"))
        output = self.image_preprocessor.extract_image_with_contour(image=self.dicom_file[0].pixel_array,
                                                                    brain_contour=brain_contour)
        cv2.circle(output, setting.get("brain_init_position"), 10, (255, 0, 0), 5)
        plt.figure(None, figsize=(5, 3), dpi=200)
        plt.subplot(1, 3, 1)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(images_hu_pixels[0], cmap='bone')
        plt.title("Original Image")
        plt.subplot(1, 3, 2)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(output, cmap='bone')
        plt.title("Crop Image")
        plt.subplot(1, 3, 3)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(binary_image, cmap='bone')
        plt.title("Crop Image")
        plt.show()

    def set_text_field(self, setting_value):
        self.ui.hu_boundary_value_text.setPlainText(str(setting_value.get("hu_boundary_value")))
        self.ui.brain_init_position_text.setPlainText(
            str(setting_value.get("brain_init_position")))

    def start(self):
        self.ui.show()
        self.set_text_field(self.main_controller.setting_value)
