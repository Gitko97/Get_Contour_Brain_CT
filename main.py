import os
import sys

import numpy as np
from PyQt5 import QtWidgets
from PyQt5.QtWidgets import QFileDialog
from matplotlib import pyplot as plt

from Dicom.Get_Contour_Brain_CT.file_in_out import FileInOut
from Dicom.Get_Contour_Brain_CT.preprocessor import Image_PreProcessor
from Dicom.Get_Contour_Brain_CT.setting_controller import SettingController
from Dicom.Get_Contour_Brain_CT.ui import Ui_Dialog


class Controller(object):
    def __init__(self, file_extension=".dcm"):
        self.file_paths = []
        self.dicom_files = []
        self.original_hu_images = []
        self.images_hu_pixels = []
        self.normalized_image = []
        self.file_io = FileInOut(file_extension)
        self.image_preprocessor = Image_PreProcessor()
        self.image_mean_pixel = None
        self.setting_value = {
            "hu_boundary_value": -70,
            "normalized_min_bound": -1000,
            "normalized_max_bound": 600,
            "contour_OTSU_boundary_value": 100,
            "brain_init_position": (250,145)
        }
        self.default_setting_value = self.setting_value

    def set_setting_value(self, setting_value):
        self.setting_value = setting_value
        self.original_hu_images, self.images_hu_pixels, self.image_mean_pixel, self.normalized_image = self.get_images_pixels(self.dicom_files)

    def start(self):
        app = QtWidgets.QApplication(sys.argv)
        self.ui = Ui_Dialog(self)
        self.ui.show()
        sys.exit(app.exec_())

    def load_files(self):
        # selected_dir = QFileDialog.getExistingDirectory(None, caption='Choose Directory', directory=os.getcwd())
        # self.file_paths = self.file_io.search(selected_dir)
        self.file_paths = self.file_io.search("/Users/joonhyoungjeon/Downloads/S005/CT")

        self.ui.listView.clear()
        for filepath in self.file_paths:
            self.ui.listView.addItem(filepath[1])
        self.dicom_files = self.image_preprocessor.load_scan(self.file_paths)
        self.original_hu_images, self.images_hu_pixels, self.image_mean_pixel, self.normalized_image = self.get_images_pixels(self.dicom_files)
        self.image_shape = self.images_hu_pixels[0].shape

    def add_files(self):
        selected_dir = QFileDialog.getExistingDirectory(None, caption='Choose Directory', directory=os.getcwd())
        searched_files = self.file_io.search(selected_dir)
        self.file_paths.extend(searched_files)
        for filepath in searched_files:
            self.ui.listView.addItem(filepath[1])
            self.dicom_files.extend(self.image_preprocessor.load_scan([filepath]))

        self.original_hu_images, self.images_hu_pixels, self.image_mean_pixel, self.normalized_image = self.get_images_pixels(self.dicom_files)

    def delete_files(self):
        for index, selected_index in enumerate(self.ui.listView.selectedIndexes()):
            self.file_paths.pop(selected_index.row() - index)
            self.dicom_files.pop(selected_index.row() - index)
            self.ui.listView.takeItem(selected_index.row() - index)

        self.original_hu_images, self.images_hu_pixels, self.image_mean_pixel, self.normalized_image = self.get_images_pixels(self.dicom_files)

    def get_images_pixels(self, dicom_files):
        original_hu_image, images_hu_pixels = self.image_preprocessor.get_pixels_hu(dicom_files, hu_boundary_value=self.setting_value.get("hu_boundary_value"))
        image_mean_pixel, normalized_image = self.image_preprocessor.normalize(images_hu_pixels, min_bound=self.setting_value.get("normalized_min_bound"), max_bound=self.setting_value.get("normalized_max_bound"))
        return original_hu_image, images_hu_pixels, image_mean_pixel, normalized_image

    def itemClicked(self):
        selected_index = self.ui.listView.currentRow()
        brain_crop_images = self.dicom_preprocess(self.original_hu_images[selected_index],self.normalized_image[selected_index])
        plt.figure(num=self.ui.listView.currentItem().text(), figsize=(5, 3), dpi=200)
        plt.subplot(1, 3, 1)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(self.original_hu_images[selected_index], cmap='bone')
        plt.title("Original Image")
        plt.subplot(1, 3, 2)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(self.images_hu_pixels[selected_index], cmap='bone')
        plt.title("Normalized Image")
        plt.subplot(1, 3, 3)
        plt.axis('off')
        plt.tight_layout()
        plt.imshow(brain_crop_images[0], cmap='bone')
        plt.title("Crop Image")
        plt.show()


    def dicom_preprocess(self, original_image,normalized_image):
        brain_crop_images = []
        contour_image = (normalized_image * 255).astype(np.uint8)  # 100 이라는 경계값으로 contour를 찾기 위해서 0~255로 값 바꿈
        _, contours, hierarchy = self.image_preprocessor.find_dicom_Countour_OTSU(contour_image, contour_boundary_value=self.setting_value.get("contour_OTSU_boundary_value"))
        brain_contour = self.image_preprocessor.find_brain_contour(contours=contours, hierarchy=hierarchy,
                                                              init_position=self.setting_value.get("brain_init_position"))
        output = self.image_preprocessor.extract_image_with_contour(image=original_image, brain_contour=brain_contour)
        brain_crop_images.append(output)
        return np.array(brain_crop_images)

    def dicom_preprocess_with_original_image(self, normalized_image, index):
        contour_image = (normalized_image * 255).astype(np.uint8)  # 100 이라는 경계값으로 contour를 찾기 위해서 0~255로 값 바꿈
        _, contours, hierarchy = self.image_preprocessor.find_dicom_Countour_OTSU(contour_image, contour_boundary_value=self.setting_value.get("contour_OTSU_boundary_value"))
        brain_contour = self.image_preprocessor.find_brain_contour(contours=contours, hierarchy=hierarchy,
                                                              init_position=self.setting_value.get("brain_init_position"))
        output = self.image_preprocessor.extract_image_with_contour(image=self.dicom_files[index].pixel_array, brain_contour=brain_contour)
        return output

    def openSetting(self):
        selected_index = self.ui.listView.currentRow()
        if selected_index == -1:
            self.ui.show_popup_ok("Warning", "한개의 이미지를 선택 후 동작해주세요")
            return
        setting_controller = SettingController(self, self.file_paths[selected_index])
        setting_controller.start()

    def file_save(self):
        croped_image = [self.dicom_preprocess(file.pixel_array, image) for file, image in zip(self.dicom_files, self.normalized_image)]
        for file, brain in zip(self.dicom_files, croped_image):
            file.PixelData = brain.astype(np.uint16).tobytes()
        self.file_io.save(self.file_paths, self.dicom_files)

if __name__ == '__main__':
    controller = Controller(".dcm")
    controller.start()
