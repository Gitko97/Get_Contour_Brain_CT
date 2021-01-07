import numpy as np
from matplotlib import pyplot as plt
from Dicom.Get_Contour_Brain_CT.preprocessor import Image_PreProcessor
from Dicom.Get_Contour_Brain_CT.setting_ui import Ui_Setting


class SettingController(object):
    def __init__(self,main_controller, dicom_path):
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
            "normalized_min_bound": int(self.ui.normalized_min_bound_text.toPlainText()),
            "normalized_max_bound": int(self.ui.normalized_max_bound_text.toPlainText()),
            "contour_OTSU_boundary_value": int(self.ui.contour_OTSU_boundary_value_text.toPlainText()),
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
        original_hu_image, images_hu_pixels = self.image_preprocessor.get_pixels_hu(self.dicom_file,
                                                                 hu_boundary_value=setting.get(
                                                                     "hu_boundary_value"))
        _, normalized_image = self.image_preprocessor.normalize(images_hu_pixels,
                                                                               min_bound=setting.get(
                                                                                   "normalized_min_bound"),
                                                                               max_bound=setting.get(
                                                                                   "normalized_max_bound"), pixel_mean=self.main_controller.image_mean_pixel)
        contour_image = (normalized_image[0] * 255).astype(np.uint8)  # 100 이라는 경계값으로 contour를 찾기 위해서 0~255로 값 바꿈
        _, contours, hierarchy = self.image_preprocessor.find_dicom_Countour_OTSU(contour_image,
                                                                                  contour_boundary_value=setting.get(
                                                                                      "contour_OTSU_boundary_value"))
        brain_contour = self.image_preprocessor.find_brain_contour(contours=contours, hierarchy=hierarchy,
                                                                   init_position=setting.get(
                                                                       "brain_init_position"))
        output = self.image_preprocessor.extract_image_with_contour(image=original_hu_image[0], brain_contour=brain_contour)
        plt.figure(None, figsize=(5, 3), dpi=200)
        plt.subplot(1, 2, 1)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(self.dicom_file[0].pixel_array, cmap='bone')
        plt.title("Original Image")
        plt.subplot(1, 2, 2)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(output, cmap='bone')
        plt.title("Crop Image")
        plt.show()

    def set_text_field(self, setting_value):
        self.ui.hu_boundary_value_text.setPlainText(str(setting_value.get("hu_boundary_value")))
        self.ui.normalized_min_bound_text.setPlainText(
            str(setting_value.get("normalized_min_bound")))
        self.ui.normalized_max_bound_text.setPlainText(
            str(setting_value.get("normalized_max_bound")))
        self.ui.contour_OTSU_boundary_value_text.setPlainText(
            str(setting_value.get("contour_OTSU_boundary_value")))
        self.ui.brain_init_position_text.setPlainText(
            str(setting_value.get("brain_init_position")))

    def start(self):
        self.ui.show()
        self.set_text_field(self.main_controller.setting_value)
